# Communicative-Robot-Cinematographer

#### 해당 프로젝트는 총 4개월간 캡스톤 디자인 수업(졸업작품)에서 진행된 프로젝트

## 1. 동기 및 목표
#### 동기
- **LLM + Robotics**에 관심이 있었고 적용하기 위한 산업 현장을 조사
- 광고 촬영 현장에서 사용 가능하다고 판단하여 프로젝트 시작

#### 목표
- 사용자의 음성 명령을 입력으로 하여 **Fine-tuning LLM**이 적절한 로봇팔 구동 코드를 생성해내는 것
- 방향에 관한 명령, 촬영 전문 용어로 이루어진 명령을 이해하여 LLM이 코드를 생성해낸다.

&nbsp;&nbsp;&nbsp;&nbsp;![image](https://github.com/user-attachments/assets/d63df860-0138-434e-a97f-638403bcd359)

## 2. 시스템 구성
시스템은 다음과 같이 구성:
- **로봇팔**: 두산 로보틱스 M0605
- **LLM 모델**: KoAlpaca-Polyglot-5.8B을 Colab 환경에서 Fine-tuning
- **턴 테이블**: 직접 제작했으며 상하 이동, 상판 회전 운동이 가능한 테이블

<img src="https://github.com/user-attachments/assets/52f98967-53d5-442c-8a0e-d01e90970bf9"  width="65%" height="65%"/>

## 3. 데이터 설명
프로젝트에서 사용된 데이터:
- **Fine-tuning 데이터**: 방향, 촬영 전문 용어로 이루어진 다양한 지시문과 알맞은 로봇팔 구동 코드 쌍<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ex) {instruction : "오른쪽으로 슬라이드샷"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input : (로봇팔 end-effector 현재 위치)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output : "movejx(posx(-225, 100, 515), vel=50, acc=50)}

데이터는 json 파일 형식으로 저장되어 있으며 속도, 거리에 대한 기준은 일반인 대상 설문조사를 받아 결정

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/user-attachments/assets/284665fa-0be6-4861-952e-0fa0289e980d"  width="65%" height="65%"/>


## 4. 사용한 모델
이 프로젝트에서는 다양한 딥러닝 모델을 사용하여 시스템을 구성, 모델의 세부 사항은 다음과 같다:
- **Segmentation**: Fast SAM(Segmentation Anything Model)
- **STT**: Whisper (medium)
- **LLM** : Fine-tuning LLM (KoAlpaca-Polyglot-5.8B)


## 5. 결과물
결과물로는 맥주, 과자 광고를 제작

1. **맥주**
  - 5가지 명령으로 컷 촬영<br>
  - 턴 테이블은 조이스틱으로 구동<br>
    <img src="https://github.com/user-attachments/assets/44848c14-130f-4ce3-aedf-135740e70707"  width="40%" height="40%"/><br>
    
2. **과자**
  - 4가지 명령으로 컷 촬영 <br>
  - 턴 테이블은 조이스틱으로 구동<br>
    <img src="https://github.com/user-attachments/assets/5220315f-6cab-4bcf-af9e-5d54e0d718ac"  width="40%" height="40%"/><br>


#### 최종 결과물 (컷 촬영본을 편집하여 최종 영상 제작)
<img src="https://github.com/user-attachments/assets/1a2f7d09-662a-45cc-8800-e731408d4696" width="45%" height="45%"/>  <img src="https://github.com/user-attachments/assets/73b7790c-f256-40a7-abe0-46e41fbd341e" width="45%" height="45%"/>


---

## 설치 및 실행 방법
해당 프로젝트는 실제 로봇팔이 있는 환경이어야 하기 때문에 직접 실행하기 힘듬<br>
추가적으로 노트북 GPU 이슈로 인해 추가 데스크탑과 TCP/IP 통신으로 진행하였기 때문에 <br>
환경이 맞지 않으면 직접 실행 불가능

