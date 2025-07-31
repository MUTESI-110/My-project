import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import screen_brightness_control as sbc
import os
import queue
import threading
import subprocess
import pygetwindow as gw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, 
                            QPushButton, QComboBox, QSlider, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QMovie, QImage, QPixmap
import sys
import json
from collections import deque

class ThreadManager:
    def __init__(self):
        self.threads = {}
        self.lock = threading.Lock()
        
    def start_thread(self, name, target, daemon=True):
        with self.lock:
            self.stop_thread(name)
            thread = threading.Thread(target=target, name=name, daemon=daemon)
            self.threads[name] = thread
            thread.start()
            
    def stop_thread(self, name):
        with self.lock:
            if name in self.threads:
                if hasattr(self.threads[name], 'stop'):
                    self.threads[name].stop = True
                self.threads[name].join(timeout=1.0)
                if self.threads[name].is_alive():
                    pass
                del self.threads[name]
                
    def stop_all_threads(self):
        with self.lock:
            for name in list(self.threads.keys()):
                self.stop_thread(name)

class CameraWorker(QObject):
    frame_processed = pyqtSignal(object)
    finished = pyqtSignal()
    
    def __init__(self, face_mesh, pose, holistic):
        super().__init__()
        self.running = True
        self.face_mesh = face_mesh
        self.pose = pose
        self.holistic = holistic
        
    def process_feed(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if ret:
                    processed_data = self.process_frame(frame)
                    self.frame_processed.emit(processed_data)
        finally:
            cap.release()
            self.finished.emit()
            
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process all models
        face_results = self.face_mesh.process(image)
        pose_results = self.pose.process(image)
        holistic_results = self.holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return {
            'image': image,
            'face_results': face_results,
            'pose_results': pose_results,
            'holistic_results': holistic_results,
            'timestamp': time.time()
        }
        
    def stop(self):
        self.running = False

class GestureActionExecutor:
    def __init__(self, app):
        self.app = app
        self.action_queue = queue.Queue()
        self.action_thread = threading.Thread(target=self._process_actions, daemon=True)
        self.action_thread.start()
        self.current_action = None
        self.lock = threading.Lock()

    def _process_actions(self):
        while True:
            action_data = self.action_queue.get()
            if action_data is None:
                break
                
            gesture, action_func = action_data
            try:
                with self.lock:
                    self.current_action = gesture
                    action_func()
            except Exception as e:
                self.app.update_status(f"Error executing {gesture}: {str(e)}", "red")
            finally:
                with self.lock:
                    self.current_action = None
                self.action_queue.task_done()

    def enqueue_action(self, gesture, action_func):
        self.action_queue.put((gesture, action_func))

    def shutdown(self):
        self.action_queue.put(None)
        self.action_thread.join()

class GestureControlApp:
    def __init__(self):
        self.thread_manager = ThreadManager()
        self.config = {
            'sensitivity': 0.8,
            'voice_feedback': False,
            'start_on_login': False,
            'recent_apps': [],
            'recent_files': []
        }
        self.load_config()

        # Initialize MediaPipe with proper settings
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # This helps with more accurate landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Enhanced gesture tracking
        self.gesture_history = deque(maxlen=15)
        self.movement_threshold = 0.015
        self.action_executor = GestureActionExecutor(self)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        # Audio control
        self.initialize_audio_control()

        
        
        # Advanced gesture verification
        self.min_gesture_frames = {
            'eye_blink': 2,
            'head_movement': 4,
            'shoulder_movement': 5,
            'mouth_gesture': 3
        }
        self.current_gesture_frames = 0
        self.last_confirmed_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5
        
        # Movement tracking buffers
        self.head_position_buffer = deque(maxlen=15)
        self.eye_state_buffer = deque(maxlen=15)
        self.shoulder_position_buffer = deque(maxlen=15)
        self.mouth_state_buffer = deque(maxlen=15)
        
        # Velocity-based detection
        self.min_velocity_threshold = 0.01
        self.smoothing_factor = 0.2
        
        # Calibration data
        self.calibration_data = {
            'neutral_head_position': (0.5, 0.5),
            'neutral_shoulder_position': 0.5,
            'eye_open_threshold': 0.25,
            'eye_closed_threshold': 0.15
        }

        # Gesture actions
        self.gesture_actions = {
            'head_tilt_left_blink_right': ('Open App Selector', self.open_app_selector),
            'head_tilt_right_blink_left': ('Open File Selector', self.open_file_selector),
            'raise_right_shoulder': ('Volume Up', lambda: self.adjust_volume(5)),
            'lower_right_shoulder': ('Volume Down', lambda: self.adjust_volume(-5)),
            'blink_both': ('Space', lambda: self.press_key('enter')),
            'head_tilt_up': ('Up Arrow', lambda: self.press_key('up')),
            'head_tilt_down': ('Down Arrow', lambda: self.press_key('down')),
            'head_tilt_left': ('Left Arrow', lambda: self.press_key('left')),
            'head_tilt_right': ('Right Arrow', lambda: self.press_key('right')),
            'mouth_open_wide': ('Show Desktop', self.show_desktop),
            'raise_right_shoulder_fast': ('Brightness Up', lambda: self.adjust_brightness(10)),
            'lower_right_shoulder_fast': ('Brightness Down', lambda: self.adjust_brightness(-10)),
            'head_tilt_up_fast': ('Switch Apps', self.switch_apps),
            'blink_both_mouth_open': ('Enter', lambda: self.press_key('space'))
        }
        
        
        # Add these new attributes
        self.last_action_time = 0
        self.action_cooldown = 2.0  # 2 seconds cooldown between actions
        self.calibration_mode = False  # Fix the missing attribute
        self.feedback_label = None  # For visual feedback
        
        # Initialize UI
        self.app = QApplication(sys.argv)
        self.window = QWidget()
        self.setup_ui()
        self.setup_ui_timer()
        
        # Start workers
        self.start_workers()

    def load_config(self):
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def initialize_audio_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))

    def setup_ui(self):
        self.window.setWindowTitle("AI Gesture Control")
        self.window.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.window.setAttribute(Qt.WA_TranslucentBackground)
        
        palette = self.window.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0, 200))
        self.window.setPalette(palette)
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        
        # Title bar
        self.title_bar = QLabel("AI-Driven Gesture Control (Drag to Move)")
        self.title_bar.setStyleSheet("""
            QLabel {
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 5px;
                background-color: rgba(50, 50, 50, 200);
                border-radius: 5px;
            }
        """)
        self.title_bar.mousePressEvent = self.title_bar_pressed
        self.title_bar.mouseMoveEvent = self.title_bar_moved
        self.layout.addWidget(self.title_bar)

        # Add feedback label (top-left corner)
        self.feedback_label = QLabel()
        self.feedback_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # self.feedback_label.setStyleSheet("""
        #     QLabel {
        #         color: white;
        #         font-size: 16px;
        #         font-weight: bold;
        #         background-color: rgba(0, 0, 0, 150);
        #         padding: 5px;
        #         border-radius: 5px;
        #     }
        # """)
        self.feedback_label.setFixedWidth(300)
        self.feedback_label.setWordWrap(True)
        
        # Make sure to add it to the layout
        self.layout.insertWidget(0, self.feedback_label)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.camera_label)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        # self.status_label.setStyleSheet("""
        #     QLabel {
        #         color: lightgreen; 
        #         font-size: 12px;
        #         padding: 5px;
        #         border: 2px solid rgba(0, 255, 0, 100);
        #         border-radius: 5px;
        #     }
        # """)
        self.layout.addWidget(self.status_label)
        
        # Gesture display
        self.gesture_tabs = QComboBox()
        self.gesture_tabs.addItems(["All Gestures", "Navigation", "Editing", "System"])
        self.gesture_tabs.currentIndexChanged.connect(self.update_gesture_display)
        # self.gesture_tabs.setStyleSheet("""
        #     QComboBox {
        #         color: white;
        #         background-color: rgba(70, 70, 70, 200);
        #         border: 1px solid gray;
        #         padding: 3px;
        #         border-radius: 3px;
        #     }
        # """)
        self.layout.addWidget(self.gesture_tabs)
        
        self.gesture_label = QLabel()
        # self.gesture_label.setStyleSheet("""
        #     QLabel {
        #         color: white; 
        #         font-size: 12px;
        #         padding: 5px;
        #         background-color: rgba(40, 40, 40, 150);
        #         border-radius: 5px;
        #     }
        # """)
        self.gesture_label.setWordWrap(True)
        self.update_gesture_display()
        self.layout.addWidget(self.gesture_label)
        
        # Control buttons
        self.button_layout = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.toggle_calibration)
        # self.calibrate_btn.setStyleSheet("""
        #     QPushButton {
        #         color: white;
        #         background-color: rgba(70, 70, 150, 200);
        #         border: 1px solid gray;
        #         padding: 5px;
        #         border-radius: 3px;
        #     }
        #     QPushButton:hover {
        #         background-color: rgba(90, 90, 170, 200);
        #     }
        # """)
        self.button_layout.addWidget(self.calibrate_btn)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.toggle_settings)
        # self.settings_btn.setStyleSheet("""
        #     QPushButton {
        #         color: white;
        #         background-color: rgba(70, 70, 70, 200);
        #         border: 1px solid gray;
        #         padding: 5px;
        #         border-radius: 3px;
        #     }
        #     QPushButton:hover {
        #         background-color: rgba(90, 90, 90, 200);
        #     }
        # """)
        self.button_layout.addWidget(self.settings_btn)
        
        self.layout.addLayout(self.button_layout)
        
        # Settings panel
        self.settings_panel = QWidget()
        self.settings_panel.setVisible(False)
        self.settings_layout = QVBoxLayout()
        
        # Sensitivity slider
        self.sensitivity_label = QLabel(f"Sensitivity: {self.config['sensitivity']:.1f}")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(int(self.config['sensitivity'] * 10))
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        self.settings_layout.addWidget(self.sensitivity_label)
        self.settings_layout.addWidget(self.sensitivity_slider)
        
        # Startup toggle
        self.startup_toggle = QPushButton(
            "Start on Login: " + ("ON" if self.config['start_on_login'] else "OFF"))
        self.startup_toggle.clicked.connect(self.toggle_start_on_login)
        self.settings_layout.addWidget(self.startup_toggle)
        
        self.settings_panel.setLayout(self.settings_layout)
        self.layout.addWidget(self.settings_panel)
        
        self.window.setLayout(self.layout)
        self.window.resize(400, 600)
        self.window.move(100, 100)
        self.window.show()

    def setup_ui_timer(self):
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

    def start_workers(self):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(
            face_mesh=self.face_mesh,
            pose=self.pose,
            holistic=self.holistic
        )
        self.camera_worker.moveToThread(self.camera_thread)
        
        self.camera_worker.frame_processed.connect(self.handle_processed_frame)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_thread.started.connect(self.camera_worker.process_feed)
        self.camera_thread.finished.connect(self.camera_worker.deleteLater)
        
        self.camera_thread.start()

    def update_ui(self):
        current_time = time.strftime("%H:%M:%S")
        self.status_label.setText(f"Last update: {current_time}")

    def handle_processed_frame(self, results):
        # Update camera view
        image = results['image']
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))
        
        # Handle calibration
        if self.calibration_mode:
            self.process_calibration_frame(results)
            return
            
        # Detect gestures
        if results['face_results'].multi_face_landmarks or results['pose_results'].pose_landmarks:
            # Pass all required arguments to detect_gesture
            gesture = self.detect_gesture(
                face_results=results['face_results'],
                pose_results=results['pose_results'],
                holistic_results=results['holistic_results'],
                timestamp=results['timestamp']
            )
            if gesture and gesture in self.gesture_actions:
                self.execute_gesture(gesture)
    def execute_gesture(self, gesture):
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return
            
        if gesture in self.gesture_actions:
            action_name, action_func = self.gesture_actions[gesture]
            
            # Show visual feedback
            self.show_action_feedback(action_name)
            
            # Execute with cooldown
            self.last_action_time = current_time
            self.action_executor.enqueue_action(gesture, action_func)
            self.update_status(f"Executing: {action_name}", "lightgreen")
    
    def show_action_feedback(self, action_name):
        self.feedback_label.setText(f"Action: {action_name}")
        
        # Make the feedback disappear after 2 seconds
        QTimer.singleShot(2000, lambda: self.feedback_label.setText(""))

    def detect_gesture(self, face_results, pose_results, holistic_results, timestamp):
        current_time = timestamp
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return None
            
        current_gesture = None
        
        self._update_movement_buffers(face_results, pose_results, timestamp)
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]            
            left_eye_closed = self._verify_eye_state('left', face_landmarks, timestamp)
            right_eye_closed = self._verify_eye_state('right', face_landmarks, timestamp)            
            head_movement, head_velocity = self._verify_head_movement(timestamp)            
            mouth_open, mouth_duration = self._verify_mouth_state(face_landmarks, timestamp)            
            if (head_movement == 'left' and right_eye_closed and 
                self._check_timing('head_tilt_left_blink_right')):
                current_gesture = 'head_tilt_left_blink_right'
            elif (head_movement == 'right' and left_eye_closed and 
                  self._check_timing('head_tilt_right_blink_left')):
                current_gesture = 'head_tilt_right_blink_left'
            elif left_eye_closed and right_eye_closed and mouth_open:
                current_gesture = 'blink_both_mouth_open'
            elif left_eye_closed and right_eye_closed:
                current_gesture = 'blink_both'
            elif head_movement == 'up' and head_velocity > self.min_velocity_threshold:
                current_gesture = 'head_tilt_up_fast'
            elif head_movement == 'up':
                current_gesture = 'head_tilt_up'
            elif head_movement == 'down':
                current_gesture = 'head_tilt_down'
            elif head_movement == 'left':
                current_gesture = 'head_tilt_left'
            elif head_movement == 'right':
                current_gesture = 'head_tilt_right'
            elif mouth_open and mouth_duration > 0.5:
                current_gesture = 'mouth_open_wide'
        
        if pose_results.pose_landmarks:
            shoulder_movement, shoulder_velocity = self._verify_shoulder_movement(timestamp)
            if shoulder_movement == 'raise_right' and shoulder_velocity > self.min_velocity_threshold:
                current_gesture = 'raise_right_shoulder_fast'
            elif shoulder_movement == 'raise_right':
                current_gesture = 'raise_right_shoulder'
            elif shoulder_movement == 'lower_right' and shoulder_velocity > self.min_velocity_threshold:
                current_gesture = 'lower_right_shoulder_fast'
            elif shoulder_movement == 'lower_right':
                current_gesture = 'lower_right_shoulder'
        
        if current_gesture:
            confirmed_gesture = self._verify_gesture_consistency(current_gesture)
            if confirmed_gesture:
                self.last_gesture_time = current_time
                return confirmed_gesture
        
        return None

    def _update_movement_buffers(self, face_results, pose_results, timestamp):
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            nose = face_landmarks.landmark[1]
            self.head_position_buffer.append((nose.x, nose.y, timestamp))
            
            left_closed = self._calculate_ear(face_landmarks, 'left') < self.calibration_data['eye_closed_threshold']
            right_closed = self._calculate_ear(face_landmarks, 'right') < self.calibration_data['eye_closed_threshold']
            self.eye_state_buffer.append((left_closed, right_closed, timestamp))
            
            upper_lip = face_landmarks.landmark[13].y
            lower_lip = face_landmarks.landmark[14].y
            mouth_open = (lower_lip - upper_lip) > 0.05
            self.mouth_state_buffer.append((mouth_open, timestamp))
        
        if pose_results.pose_landmarks:
            right_shoulder = pose_results.pose_landmarks.landmark[
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            self.shoulder_position_buffer.append((right_shoulder.y, timestamp))

    def _calculate_ear(self, face_landmarks, eye='left'):
        if eye == 'left':
            points = [362, 385, 387, 263, 373, 380]
        else:
            points = [33, 160, 158, 133, 153, 144]
        
        vertical1 = np.linalg.norm(np.array([
            face_landmarks.landmark[points[1]].x - face_landmarks.landmark[points[5]].x,
            face_landmarks.landmark[points[1]].y - face_landmarks.landmark[points[5]].y
        ]))
        vertical2 = np.linalg.norm(np.array([
            face_landmarks.landmark[points[2]].x - face_landmarks.landmark[points[4]].x,
            face_landmarks.landmark[points[2]].y - face_landmarks.landmark[points[4]].y
        ]))
        horizontal = np.linalg.norm(np.array([
            face_landmarks.landmark[points[0]].x - face_landmarks.landmark[points[3]].x,
            face_landmarks.landmark[points[0]].y - face_landmarks.landmark[points[3]].y
        ]))
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def _verify_eye_state(self, eye, face_landmarks, timestamp):
        current_state = self._calculate_ear(face_landmarks, eye) < self.calibration_data['eye_closed_threshold']
        required_frames = max(2, int(self.min_gesture_frames['eye_blink']/2))
        consistent_frames = 0
        
        for state in reversed(self.eye_state_buffer):
            if eye == 'left' and state[0] == current_state:
                consistent_frames += 1
            elif eye == 'right' and state[1] == current_state:
                consistent_frames += 1
            else:
                break
        
        return consistent_frames >= required_frames and current_state

    def _verify_head_movement(self, timestamp):
        if len(self.head_position_buffer) < 2:
            return None, 0
            
        start_x, start_y, start_time = self.head_position_buffer[0]
        end_x, end_y, end_time = self.head_position_buffer[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        dt = end_time - start_time
        
        if dt == 0:
            return None, 0
            
        velocity = np.sqrt(dx**2 + dy**2) / dt
        
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return None, velocity
            
        if abs(dx) > abs(dy):
            return ('left' if dx < 0 else 'right'), velocity
        else:
            return ('up' if dy < 0 else 'down'), velocity

    def _verify_shoulder_movement(self, timestamp):
        if len(self.shoulder_position_buffer) < self.min_gesture_frames['shoulder_movement']:
            return None, 0
            
        start_y, start_time = self.shoulder_position_buffer[0]
        end_y, end_time = self.shoulder_position_buffer[-1]
        dy = end_y - start_y
        dt = end_time - start_time
        
        if dt == 0:
            return None, 0
            
        velocity = abs(dy) / dt
        
        if abs(dy) < self.movement_threshold:
            return None, velocity
            
        return ('raise_right' if dy < 0 else 'lower_right'), velocity

    def _verify_mouth_state(self, face_landmarks, timestamp):
        upper_lip = face_landmarks.landmark[13].y
        lower_lip = face_landmarks.landmark[14].y
        current_state = (lower_lip - upper_lip) > 0.05
        
        if not current_state:
            return False, 0
            
        # Calculate duration of mouth open state
        duration = 0
        for state, state_time in reversed(self.mouth_state_buffer):
            if state:
                duration = timestamp - state_time
            else:
                break
                
        return current_state, duration

    def _verify_gesture_consistency(self, current_gesture):
        if current_gesture == self.last_confirmed_gesture:
            return None
            
        self.current_gesture_frames += 1
        
        if self.current_gesture_frames >= self.min_gesture_frames.get(current_gesture.split('_')[0], 3):
            self.last_confirmed_gesture = current_gesture
            self.current_gesture_frames = 0
            return current_gesture
        
        return None

    def _check_timing(self, gesture_type):
        # Placeholder for timing verification between combined gestures
        return True

    def needs_confirmation(self, gesture):
        confirmation_required = [
            'open_app_selector',
            'open_file_selector',
            'show_desktop'
        ]
        
        if gesture in self.gesture_actions:
            action_id = self.gesture_actions[gesture][0].lower().replace(' ', '_')
            return action_id in confirmation_required
        
        return False

    def request_confirmation(self, gesture):
        self.confirmation_mode = True
        self.pending_action = gesture
        self.confirmation_start_time = time.time()
        
        action_name = self.gesture_actions[gesture][0]
        self.update_status(f"Confirm {action_name}? Look at camera to accept", "yellow")
        
        QTimer.singleShot(3000, self._check_confirmation_timeout)

    def _check_confirmation_timeout(self):
        if self.confirmation_mode:
            self.confirmation_mode = False
            self.pending_action = None
            self.update_status("Confirmation timed out", "red")

    def update_gesture_display(self):
        tab = self.gesture_tabs.currentText()
        gesture_text = "<b>Available Gestures:</b><br>"
        
        gesture_categories = {
            "All Gestures": self.gesture_actions.items(),
            "Navigation": [
                (g, a) for g, a in self.gesture_actions.items() 
                if any(k in a[0].lower() for k in ['arrow', 'scroll', 'tab', 'switch', 'win', 'desktop'])
            ],
            "Editing": [
                (g, a) for g, a in self.gesture_actions.items() 
                if any(k in a[0].lower() for k in ['copy', 'paste', 'space', 'enter'])
            ],
            "System": [
                (g, a) for g, a in self.gesture_actions.items() 
                if any(k in a[0].lower() for k in ['app', 'file', 'volume', 'brightness'])
            ]
        }
        
        for i, (gesture, (desc, _)) in enumerate(gesture_categories[tab], 1):
            gesture_text += f"{i}. {desc}: {gesture.replace('_', ' ')}<br>"
        
        self.gesture_label.setText(gesture_text)

    def update_status(self, message, color="lightgreen"):
        self.status_label.setText(f"Status: {message}")
        # self.status_label.setStyleSheet(f"""
        #     QLabel {{
        #         color: {color}; 
        #         font-size: 12px;
        #         padding: 5px;
        #         border: 2px solid rgba({self._color_to_rgb(color)}, 100);
        #         border-radius: 5px;
        #     }}
        # """)

    def _color_to_rgb(self, color_name):
        colors = {
            "lightgreen": "0, 255, 0",
            "yellow": "255, 255, 0",
            "red": "255, 0, 0",
            "cyan": "0, 255, 255"
        }
        return colors.get(color_name, "0, 255, 0")

    def title_bar_pressed(self, event):
        self.old_pos = event.globalPos()

    def title_bar_moved(self, event):
        delta = QPoint(event.globalPos() - self.old_pos)
        self.window.move(self.window.x() + delta.x(), self.window.y() + delta.y())
        self.old_pos = event.globalPos()

    def toggle_calibration(self):
        self.calibration_mode = not self.calibration_mode
        
        if self.calibration_mode:
            self.update_status("Calibration mode activated", "yellow")
            self.calibrate_btn.setText("Exit Calibration")            
            # CALIBRATION INITIALIZATION
            self.calibration_samples = [] 
            self.calibration_start_time = time.time()
            self.calibration_phase = 0 
            self.update_status("Please look straight ahead (Neutral Position)", "cyan")            
            # Set neutral positions (initialize if not exists)
            if not hasattr(self, 'neutral_head_position'):
                self.neutral_head_position = (0.5, 0.5)  # Default values
                self.neutral_shoulder_position = 0.5
                self.eye_open_threshold = 0.25
                self.eye_closed_threshold = 0.15
                
        else:
            self.update_status("Calibration mode deactivated", "lightgreen")
            self.calibrate_btn.setText("Calibrate")            
            # CALIBRATION CLEANUP
            if hasattr(self, 'calibration_samples') and self.calibration_samples:
                # Calculate averages from collected samples
                neutral_head = np.mean([s['head'] for s in self.calibration_samples if s['phase'] == 0], axis=0)
                neutral_shoulder = np.mean([s['shoulder'] for s in self.calibration_samples if s['phase'] == 0], axis=0)
                
                # Update calibration data
                self.calibration_data.update({
                    'neutral_head_position': neutral_head,
                    'neutral_shoulder_position': neutral_shoulder,
                    'eye_open_threshold': 0.25,  
                    'eye_closed_threshold': 0.15
                })                
            # Clean up temporary attributes
            for attr in ['calibration_samples', 'calibration_start_time', 'calibration_phase']:
                if hasattr(self, attr):
                    delattr(self, attr)
    def process_calibration_frame(self, frame_data):
        if not self.calibration_mode:
            return
            
        current_time = time.time()
        
        # Only sample every 0.5 seconds during calibration
        if current_time - self.calibration_last_sample < 0.5:
            return
            
        self.calibration_last_sample = current_time
        
        sample = {
            'timestamp': current_time,
            'phase': self.calibration_phase,
            'head': self.get_current_head_position(frame_data),
            'shoulder': self.get_current_shoulder_position(frame_data),
            'eye_state': self.get_current_eye_state(frame_data)
        }
        
        self.calibration_samples.append(sample)
        
        # Progress through calibration phases
        if current_time - self.calibration_phase_start > 3.0:  # 3 seconds per phase
            self.calibration_phase += 1
            self.calibration_phase_start = current_time
            
            if self.calibration_phase == 1:
                self.update_status("Please close both eyes", "cyan")
            elif self.calibration_phase == 2:
                self.update_status("Tilt head left", "cyan")
            # Add more phases as needed
            else:
                self.toggle_calibration()  # End calibration

    def toggle_settings(self, show=None):
        if show is None:
            self.settings_mode = not self.settings_mode
        else:
            self.settings_mode = show
        
        self.settings_panel.setVisible(self.settings_mode)
        if self.settings_mode:
            self.settings_btn.setText("Hide Settings")
            self.update_status("Settings panel opened", "lightgreen")
        else:
            self.settings_btn.setText("Settings")
            self.update_status("Settings panel closed", "lightgreen")

    def update_sensitivity(self, value):
        self.config['sensitivity'] = value / 10.0
        self.sensitivity_label.setText(f"Sensitivity: {self.config['sensitivity']:.1f}")
        self.save_config()

    def toggle_start_on_login(self):
        self.config['start_on_login'] = not self.config['start_on_login']
        self.startup_toggle.setText(
            "Start on Login: " + ("ON" if self.config['start_on_login'] else "OFF"))
        self.save_config()

    # Action implementations
    def open_app_selector(self):
        try:
            os.startfile("notepad.exe")
            self.update_status("Opened Notepad", "lightgreen")
        except Exception as e:
            self.update_status(f"Error opening app: {str(e)}", "red")

    def open_file_selector(self):
        try:
            docs_path = os.path.join(os.path.expanduser("~"), "Documents")
            os.startfile(docs_path)
            self.update_status("Opened Documents folder", "lightgreen")
        except Exception as e:
            self.update_status(f"Error opening file: {str(e)}", "red")

    def adjust_volume(self, delta):
        try:
            current = self.volume_control.GetMasterVolumeLevelScalar()
            new_vol = max(0.0, min(1.0, current + delta/100.0))
            self.volume_control.SetMasterVolumeLevelScalar(new_vol, None)
            self.update_status(f"Volume set to {int(new_vol*100)}%", "lightgreen")
        except Exception as e:
            self.update_status(f"Error adjusting volume: {str(e)}", "red")

    def adjust_brightness(self, delta):
        try:
            current = sbc.get_brightness()
            if isinstance(current, list):
                current = current[0]
            new_brightness = max(0, min(100, current + delta))
            sbc.set_brightness(new_brightness)
            self.update_status(f"Brightness set to {new_brightness}%", "lightgreen")
        except Exception as e:
            self.update_status(f"Error adjusting brightness: {str(e)}", "red")

    def press_key(self, key):
        try:
            pyautogui.press(key)
            self.update_status(f"Pressed {key}", "lightgreen")
        except Exception as e:
            self.update_status(f"Error pressing key: {str(e)}", "red")

    def show_desktop(self):
        try:
            pyautogui.hotkey('win', 'd')
            self.update_status("Showing desktop", "lightgreen")
        except Exception as e:
            self.update_status(f"Error showing desktop: {str(e)}", "red")

    def switch_apps(self):
        try:
            pyautogui.hotkey('win', 'tab')
            self.update_status("Switched applications", "lightgreen")
        except Exception as e:
            self.update_status(f"Error switching apps: {str(e)}", "red")

    def run(self):
        try:
            sys.exit(self.app.exec_())
        finally:
            self.cleanup()

    def cleanup(self):
        if self.camera_worker:
            self.camera_worker.stop()
        if self.camera_thread:
            self.camera_thread.quit()
            self.camera_thread.wait()
        self.face_mesh.close()
        self.pose.close()
        self.holistic.close()
        self.action_executor.shutdown()
        self.ui_timer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_app = GestureControlApp()
    gesture_app.run()