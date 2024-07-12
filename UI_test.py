# 바꿔줘야 하는 값 : table x,y,z / delta x,z,yaw / screen 사이즈, 위치 / 초기위치 다시 재조정
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QInputDialog
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from time import sleep
import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import socket
import ast
import re
import math
import pyaudio
import whisper
from collections import deque
import queue
from threading import Thread
import wave
from array import array

### 서버 ###
server_address = "192.168.137.50"
server_port = 12345

### 설정값 ###
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "C:/woogi/CoRoC/code/voice/ffmpeg/bin/output.wav"
SILENCE_THRESHOLD = 2000
SILENCE_SECONDS = 3
screen_width = 1280
screen_height = 720
screen_center_x = screen_width // 2
screen_center_y = screen_height // 2
substring = "vel="
device = 'cuda' if torch.cuda.is_available() else 'cpu'
table_x = -680
table_y = 0
table_z = 200

### 함수 ###
class EllipseWidget(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.label = QLabel(self.text, self)
        self.label.setFont(QFont('Arial', 14))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setStyleSheet("padding-left: 20px; padding-right: 20px;")
        self.setGeometry(390, 1120, 1780, 130)  # 위치 및 크기 설정
        self.update_geometry()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        brush = QBrush(Qt.white)
        painter.setBrush(brush)
        painter.setPen(Qt.black)
        rect = self.rect()
        painter.drawRect(rect)

    def update_geometry(self):
        self.label.setGeometry(0, 0, self.width(), self.height())

    def setText(self, text):
        self.text = text
        self.label.setText(text)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.recording = False
        self.writer = None
        self.recording_index = 1
        
    def run(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, screen_width, screen_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, screen_width, screen_height, rs.format.bgr8, 30)
        self.pipeline.start(config)
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(800, 650, Qt.KeepAspectRatio)
                
                self.change_pixmap_signal.emit(p)
                
                # 녹화 중인 경우 프레임 저장
                if self.recording and self.writer is not None:
                    self.writer.write(color_image)
                
                sleep(0.03)  # 1프레임당 0.03초 (약 30fps)
        finally:
            self.pipeline.stop()
            if self.writer is not None:
                self.writer.release()

    def start_recording(self):
        filename = f"./output_{self.recording_index}.avi"
        self.recording_index += 1
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, 30, (screen_width, screen_height))
        self.recording = True

    def stop_recording(self):
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
class App(QWidget):
    update_voice_result_signal = pyqtSignal(str)
    listen_complete_signal = pyqtSignal(str)

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.initUI()
        self.initVideo()
        self.update_voice_result_signal.connect(self.update_voice_result)
        self.listen_complete_signal.connect(self.process_voice_command)
        self.is_listening = False

    def initUI(self):
        self.setWindowTitle('Control Panel')
        self.setGeometry(0, 0, 2560, 1440)  # 창 크기 조절
        
        label = QLabel('Communicative Robot Cinematographer', self)
        label.setFont(QFont('Lucida Sans Typewriter', 40))
        label.move(530, 80)

        label2 = QLabel('<Control Button>', self)
        label2.setFont(QFont('Lucida Sans Typewriter', 20))
        label2.move(605, 275)

        label3 = QLabel('<실시간 영상>', self)
        label3.setFont(QFont('Lucida Sans Typewriter', 20))
        label3.move(1640, 230)

        btn0 = QPushButton('0: 초기위치', self)
        btn0.setFont(QFont('Arial', 20))  # 버튼 폰트 설정
        btn0.setStyleSheet("background-color: #FFFFFF; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")  # 버튼 스타일 설정
        btn0.clicked.connect(lambda: on_click('0'))
        btn0.move(390, 350)

        btn1 = QPushButton('1: SAM', self)
        btn1.setFont(QFont('Arial', 20))  # 버튼 폰트 설정
        btn1.setStyleSheet("background-color: green; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")  # 버튼 스타일 설정
        btn1.clicked.connect(lambda: on_click('1'))
        btn1.move(840, 350)

        btn2 = QPushButton('2: LLM', self)
        btn2.setFont(QFont('Arial', 20))  # 버튼 폰트 설정
        btn2.setStyleSheet("background-color: skyblue; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")  # 버튼 스타일 설정
        btn2.clicked.connect(lambda: on_click('2'))
        btn2.move(390, 750)

        btn3 = QPushButton('3: 통신 종료', self)
        btn3.setFont(QFont('Arial', 20))  # 버튼 폰트 설정
        btn3.setStyleSheet("background-color: orange; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")  # 버튼 스타일 설정
        btn3.clicked.connect(lambda: on_click('3'))
        btn3.move(840, 750)

        self.record_btn = QPushButton("녹화 시작", self)
        self.record_btn.setFont(QFont('Arial', 18))
        self.record_btn.setStyleSheet("background-color: lightcoral; color: black; padding: 10px; border-radius: 10px; width: 780px; height: 80px")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.move(1370, 960)
        self.is_recording = False
        
        self.video_viewer_label = QLabel(self)
        self.video_viewer_label.setGeometry(1370, 300, 800, 650)  # 비디오 레이블 크기 및 위치 설정
        self.video_viewer_label.setStyleSheet("background-color: black;")  # 비디오 레이블 배경색 설정
        
        self.ellipse_widget = EllipseWidget('음성 인식: ', self)
        
    def initVideo(self):
        self.thread = VideoThread(self.pipeline)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, image):
        self.video_viewer_label.setPixmap(QPixmap.fromImage(image))
        
    def toggle_recording(self):
        if self.is_recording:
            self.thread.stop_recording()
            self.record_btn.setText("녹화 시작")
        else:
            self.thread.start_recording()
            self.record_btn.setText("녹화 중지")
        self.is_recording = not self.is_recording
    
    def update_voice_result(self, text):
        self.ellipse_widget.setText(f'{text}')

    def toggle_listening(self):
        if self.is_listening:
            self.ellipse_widget.setText('음성 인식: ')
            self.is_listening = False
        else:
            self.start_listening()

    def start_listening(self):
        self.ellipse_widget.setText('음성 인식 준비')
        self.is_listening = True
        listener_thread = Thread(target=self.listen_and_transcribe)
        listener_thread.start()

    def listen_and_transcribe(self):
        voice()
        voice_result = model_w.transcribe(WAVE_OUTPUT_FILENAME)
        print(voice_result['text'])
        text = voice_result['text']
        
        self.update_voice_result_signal.emit(f'음성 인식 결과: {text}')
        self.listen_complete_signal.emit(text)

    def process_voice_command(self, text):
        message = '2'
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_address, server_port))
        client_socket.send(message.encode("utf-8"))
        print('연결완료')
        
        cur = client_socket.recv(1024).decode("utf-8")
        cur = str(cur).replace(", ", ",")
        print(cur)
        a = gen(text, cur)
        print(a)
        n, ok = QInputDialog.getInt(None, 'Input Dialog', '원하는 등장 횟수를 입력하세요 (1부터 5까지 가능):', min=1, max=5)
        
        if ok:
            if 1 <= n <= 5:
                result = slice_move(a, substring, n)
                if result is not None:
                    print(f"{result}")
                    dsr_code = output_process(result)
                    output = dsr_code
                    print(output)
                    llm_request = f"{message}&&{output}"
                    client_socket.send(llm_request.encode("utf-8"))
                    client_socket.close()
                else:
                    print(f"{n}번째로 나오는 '0)'를 찾을 수 없습니다.")
            else:
                print("잘못된 입력입니다. 1부터 5까지의 정수를 입력하세요.")
        else:
            client_socket.close()
            print("입력이 취소되었습니다.")
        
        self.is_listening = False

def execute_message(message):
    if message == '0':
        print('1')
    
    elif message == '1':
        print('1')
    elif message == '3':
        print('프로그램 종료')
        pipeline.stop()
        app.quit()
        sys.exit()

def on_click(message):
    if message == '2':
        print('1')
    else:
        execute_message(message)
        
### 메인 ###
if __name__ == '__main__':
    pipeline = rs.pipeline()
    app = QApplication(sys.argv)
    ex = App(pipeline)
    ex.show()
    sys.exit(app.exec_())
