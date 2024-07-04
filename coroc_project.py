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
RATE = 48000
WAVE_OUTPUT_FILENAME = "./model/ffmpeg/bin/output.wav"
SILENCE_THRESHOLD = 1000
SILENCE_SECONDS = 2
screen_width = 1280
screen_height = 720
screen_center_x = screen_width // 2
screen_center_y = screen_height // 2
substring = "vel="
device = 'cuda' if torch.cuda.is_available() else 'cpu'
table_x = -670
table_y = 0
table_z = 350

### 모델 ###
# FastSAM
sam_model = FastSAM('./model/FastSAM-s.pt')
# LLM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
model = AutoModelForCausalLM.from_pretrained("./model/5B_model_0607", quantization_config=bnb_config)
PROMPT_DICT = {
    "prompt_input": (
        "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    )
}
model.eval()
model.config.use_cache=True
# Whisper
model_w = whisper.load_model("medium")

### 함수 ###
class EllipseWidget(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.label = QLabel(self.text, self)
        self.label.setFont(QFont('Arial', 14))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setStyleSheet("padding-left: 20px; padding-right: 20px;")
        self.setGeometry(390, 1120, 1780, 130)
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

def gen(x, cur):
    a = PROMPT_DICT['prompt_input'].format(instruction=x, input=cur)
    input_ids = tokenizer.encode(a, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    
    gened = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=75,
        num_return_sequences=1,
        do_sample=False,
        eos_token_id=2,
    )
    response = tokenizer.decode(gened[0])
    return response.split('### Response(응답):')[1].strip()

def slice_move(a, substring, n):
    index = -1
    for _ in range(n):
        index = a.find(substring, index + 1)
        if index == -1:
            break
    if index != -1:
        index += len(substring) + 3
        return a[:index]
    else:
        return None

def rpy(llm_output):
    rpy = []
    matches = re.findall(r"posx\(([^)]+)\)", llm_output)
    for match in matches:
        numbers = match.split(',')
        x, y, z = int(numbers[0]), int(numbers[1]), int(numbers[2])

        roll = round(math.degrees(math.atan2(abs(table_y - y), abs(table_x - x))), 1)
        if table_y > y:
            r = -(roll)
        else:
            r = roll
        pitch = round(math.degrees(math.atan2(abs(math.sqrt(((table_x - x)**2)+((table_y-y)**2))), abs(table_z - z))), 1)
        if table_z > z:
            p = -(pitch)
        else:
            p = -90 -(90-abs(pitch))
        rpy.append((r, p, 0.0))
    return rpy

def output_process(llm_output):
    rpy_values = rpy(llm_output)
    posx_calls = re.findall(r"(posx\(([^)]+)\))", llm_output)
    updated_llm = llm_output
    offset = 0

    for i, (full_match, match_content) in enumerate(posx_calls):
        r, p, y = rpy_values[i]
        new_posx = f"posx([{match_content},{r},{p},{y}])"
        start_index = updated_llm.find(full_match, offset)
        end_index = start_index + len(full_match)
        updated_llm = updated_llm[:start_index] + new_posx + updated_llm[end_index:]
        offset = start_index + len(new_posx)

    updated_llm = re.sub(r"vel=(\d+)", lambda m: f"vel={m.group(1)}, acc={m.group(1)}", updated_llm)

    commands = re.split(r"(?<=vel=\d),", updated_llm)

    movel_sequence = []
    movec_sequence = []

    for command in commands:
        if "movel" in command:
            movel_sequence.append(command.strip())
        elif "movec" in command:
            movec_sequence.append(command.strip())

    dsr_code = []
    while movel_sequence or movec_sequence:
        if movel_sequence:
            dsr_code.append(movel_sequence.pop(0))
        if movec_sequence:
            dsr_code.append(movec_sequence.pop(0))

    return dsr_code

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
                
                if self.recording and self.writer is not None:
                    self.writer.write(color_image)
                sleep(0.03)
                
        finally:
            self.pipeline.stop()
            if self.writer is not None:
                self.writer.release()

    def start_recording(self):
        filename = f"./output/video_{self.recording_index}.avi"
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
        self.setWindowTitle('CoRoC UI')
        self.setGeometry(0, 0, 2560, 1440)

        label = QLabel('Communicative Robot Cinematographer', self)
        label.setFont(QFont('Lucida Sans Typewriter', 40))
        label.move(530, 80)

        label2 = QLabel('<Control Button>', self)
        label2.setFont(QFont('Lucida Sans Typewriter', 20))
        label2.move(605, 275)

        label3 = QLabel('<실시간 영상>', self)
        label3.setFont(QFont('Lucida Sans Typewriter', 20))
        label3.move(1640, 230)

        btn0 = QPushButton('Reset\n\n위치 초기화', self)
        btn0.setFont(QFont('Arial', 20))
        btn0.setStyleSheet("background-color: #FFFFFF; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")
        btn0.clicked.connect(lambda: on_click('0'))
        btn0.move(390, 350)

        btn1 = QPushButton('SAM\n\n비율 맞추기', self)
        btn1.setFont(QFont('Arial', 20))
        btn1.setStyleSheet("background-color: green; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")
        btn1.clicked.connect(lambda: on_click('1'))
        btn1.move(840, 350)

        btn2 = QPushButton('LLM\n\n음성 명령', self)
        btn2.setFont(QFont('Arial', 20))
        btn2.setStyleSheet("background-color: skyblue; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")
        btn2.clicked.connect(lambda: on_click('2'))
        btn2.move(390, 750)

        btn3 = QPushButton('종료', self)
        btn3.setFont(QFont('Arial', 20))
        btn3.setStyleSheet("background-color: orange; color: black; padding: 10px; border-radius: 10px; width: 300px; height: 270px")
        btn3.clicked.connect(lambda: on_click('3'))
        btn3.move(840, 750)

        self.record_btn = QPushButton("녹화 시작", self)
        self.record_btn.setFont(QFont('Arial', 18))
        self.record_btn.setStyleSheet("background-color: lightcoral; color: black; padding: 10px; border-radius: 10px; width: 780px; height: 80px")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.move(1370, 960)
        self.is_recording = False
        
        self.video_viewer_label = QLabel(self)
        self.video_viewer_label.setGeometry(1370, 300, 800, 650)
        self.video_viewer_label.setStyleSheet("background-color: black;")
        
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
        text = voice_result['text'].strip()
        
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
        a = a.split('\n')[0]
        print(a)
        self.update_voice_result_signal.emit(f'음성 명령 : {text} \n추론 좌표 : {a}')
        
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
                    print(f"{n}번째로 나오는 'vel='를 찾을 수 없습니다.")
            else:
                print("잘못된 입력입니다. 1부터 5까지의 정수를 입력하세요.")
        else:
            client_socket.close()
            print("입력이 취소되었습니다.")
        
        self.is_listening = False

def execute_message(message):
    if message == '0':
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_address, server_port))
        client_socket.send(message.encode("utf-8"))
        client_socket.close()        
    
    elif message == '1':
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_address, server_port))
        
        # 프레임 받아오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 컬러 데이터를 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 이미지 저장
        cv2.imwrite("./output/color_image.png", color_image)
        print("컬러 이미지 저장 완료")
        image = cv2.imread("./output/color_image.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 생성
        everything_results = sam_model(image, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image, everything_results, device=device)
        input_point = [[screen_center_x, screen_center_y]]
        input_label = [1]
        ann = prompt_process.point_prompt(points=input_point, pointlabel=input_label)
        masks = ann[0].masks if isinstance(ann, list) else ann.masks
        mask_data = masks[0].data if hasattr(masks[0], 'data') else masks[0]
        mask_array = mask_data.cpu().numpy().astype(np.uint8)
        mask_array = mask_array[0]

        # 바운딩 박스 그리기
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        center_x, center_y = x + w//2, y + h//2
        depth_value = depth_frame.get_distance(center_x, center_y)
        cv2.circle(color_image, (center_x, center_y), 5, (255, 0, 0), -1)
        cv2.circle(color_image, (screen_center_x, screen_center_y), 5, (0, 255, 0), -1)
        center_differ = screen_center_y-center_y
        
        cv2.putText(color_image, f'distance: ({depth_value:.2f})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(color_image, f'Width: {w}, Height: {h}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color_image, f'center: ({center_differ:.2f})', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow("Last Mask with Bounding Box", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ### 요 조절 ###
        cur_joint = [3.775, 33.207, -93.603, 176.218, 59.700, -176.202]
        delta_yaw = 0.03

        if center_differ < 0:
            new_yaw = cur_joint[4] + (delta_yaw * abs(center_differ))
        elif center_differ > 0:
            new_yaw = cur_joint[4] - (delta_yaw * abs(center_differ))
        else:
            new_yaw = cur_joint[4]

        target_joint = [cur_joint[0] ,cur_joint[1] ,cur_joint[2] ,cur_joint[3] , round(new_yaw,3), cur_joint[5]]
        target_joint_str = ', '.join(map(str, target_joint))
        send_yaw = 'movej([' + target_joint_str + '], vel=10, acc=10)'
        
        output = send_yaw

        print(output)
        request = f"{message}&&{output}"
        client_socket.send(request.encode("utf-8"))
        recv_pos = client_socket.recv(1024).decode("utf-8")
        print(recv_pos)

        ### 깊이 조절 ###
        current_pos = recv_pos
        current_pos = ast.literal_eval(current_pos)
        width_ratio = (screen_width * 0.5) / w
        height_ratio = (screen_height * 0.6) / h

        if w > h:
            go_depth = depth_value * (1 - (1 / width_ratio))
        elif w <= h:
            go_depth = depth_value * (1 - (1 / height_ratio))
            
        delta_x = -6 * 100 * go_depth
        delta_z = -3 * 100 * go_depth

        new_x = current_pos[0] + delta_x
        new_z = current_pos[2] + delta_z
        target_pos = [round(new_x,3), current_pos[1], round(new_z,3), current_pos[3], current_pos[4], current_pos[5]]
        target_pos_str = ', '.join(map(str, target_pos))

        half_depth_value = depth_value / 2
        if go_depth >= half_depth_value:
            vel = 25
            acc = 20
        else:
            vel = 10
            acc = 5

        send_xz = 'movejx(posx([' + target_pos_str + ']), vel=' + str(vel) + ', acc=' + str(acc) + ', sol=6)'
        
        client_socket.send(send_xz.encode("utf-8"))
        
        print(send_xz)
        client_socket.close()

    elif message == '3':
        print('프로그램 종료')
        pipeline.stop()
        app.quit()
        sys.exit()

def voice():
    q = queue.Queue()
    listener_thread  = Thread(target=listen, args=(q,))
    listener_thread.start()
    listener_thread.join()

def listen(q):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=1,
        frames_per_buffer=CHUNK
    )

    for _ in range(0, int(RATE / CHUNK)):
        stream.read(CHUNK, exception_on_overflow=False)

    is_started = False
    vol_que = deque(maxlen=SILENCE_SECONDS)

    print('start listening')

    frames = []
    while True:
        try:
            vol_sum = 0

            for _ in range(0, int(RATE / CHUNK)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                vol = max(array('h', data))
                vol_sum += vol

                if not is_started:
                    if vol >= SILENCE_THRESHOLD:
                        print('start of speech detected')
                        ex.update_voice_result_signal.emit('음성 인식 시작')

                        is_started = True

                if is_started:
                    frames.append(data)
                    q.put(data)

            if is_started:
                vol_que.append(vol_sum / (RATE / CHUNK) < SILENCE_THRESHOLD)
                if len(vol_que) == SILENCE_SECONDS and all(vol_que):
                    print('end of speech detected')
                    ex.update_voice_result_signal.emit(f'음성 인식 종료')

                    break
        except queue.Full:
            pass
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    voice_result = model_w.transcribe(WAVE_OUTPUT_FILENAME)
    text = voice_result['text'].strip()
    ex.update_voice_result_signal.emit(f'음성 명령: {text} \n답변 생성 중 ......')

def on_click(message):
    if message == '2':
        ex.toggle_listening()
    else:
        execute_message(message)

### 메인 ###
if __name__ == '__main__':
    pipeline = rs.pipeline()
    app = QApplication(sys.argv)
    ex = App(pipeline)
    ex.show()
    sys.exit(app.exec_())