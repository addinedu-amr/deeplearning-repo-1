#  💪까꿍 PT💪
![image](https://user-images.githubusercontent.com/110883172/213055379-b0ee5d67-a418-4456-8e60-57412390a25a.png)


## 🏃개요🏃 
- 운동을 해야한다고 말만하고, 안하는 사람들이 있다. 그 이유는 운동을 하는 방법을 모르기 때문이라 생각한다. 그러니 딥러닝을 통해 자세를 교정해주고, 서로 양방향 소통이 가능한 프로그램을 제작하고자 한다.


## 🏃시연영상 및 발표자료🏃

시연영상
</br>
https://www.youtube.com/watch?v=VtHZZ8RGzP0&t=18s

</br>

발표자료
</br>
발표자료 폴더 참고




## 🏃팀원 및 역할 분배🏃
#### ◾ 최선민 (Project Management Office)
- 프로젝트 기획 및 설계
- 데이터 수집(운동의 정확한 자세 정의 및 측정) 
- media pipe pose estimation 알고리즘 설계


#### ◾ 김두엽(Project manager)
- 프로젝트 설계 (시간분배, 일정)
- 프로젝트 기록 및 설계도 작성
- media pipe pose estimation 알고리즘 설계

#### ◾ 박민제(system architecture engineer)
- 프로젝트 메니징(설계, 구체화 방향성 제시)
- 팀원 어드바이스
- media pipe
- 데이터 수집(모션 캡처 및 음성녹음)

#### ◾ 박인 (NLT Project Leader)
- 자연어처리 (양방향 소통)
- nltk
- 음성 API (마이크로 소프트, 구글) 를 활용한 인식
- 데이터 수집(모션 캡처 및 음성녹음)


#### ◾ 손훈민 (NLT, DB Project Leader / Data Analyst)
- 자연어처리 (양방향 소통
- nltk
- 음성 API (마이크로 소프트, 구글) 를 활용한 인식
- 데이터 수집(모션 캡처 및 음성녹음)
- __목소리의 주인공__


## 🏃설계🏃
### ◾ 앱 UI 설계
![딥러닝 프로젝트 UI 설계 drawio (2)](https://user-images.githubusercontent.com/110883172/212559353-b19d15be-90f5-467c-a2ae-cbf922123681.png)




### ◾ 엑티브 다이어그램 설계
![딥러닝 엑티브 다이어그램 drawio (1)](https://user-images.githubusercontent.com/110883172/212559636-e462aa40-0f95-4b29-950d-64b6cfb9d7b6.png)




## 🏃구현🏃
### ◾ pose estimation
<center><img src="https://user-images.githubusercontent.com/110883172/213050490-fe6abc68-d3f8-423b-a4b7-73957f0c30f3.png" width="400" height="400"/></center>
- mediapipe를 사용하여, 각 관절의 좌표값을 읽어들임

#### 구현되어 있는 운동

- 스쿼트
- 팔굽혀펴기
- 런지
- 플랭크

#### 데이터 출처
<center><img src="https://user-images.githubusercontent.com/110883172/213050058-c4b89369-5a06-4d10-b916-99b8ea3b83dd.png" width="400" height="700"/></center>
- 정방향 모션 캡처 (준비자세, 운동자세, 틀린자세)


#### 모델 구현
- pose estimation (DL)
- Descision Tree (ML)


### : ◾ 자연어처리
#### 머신러닝을 통한 네 VS 아니요 음성인식 구현

![image](https://user-images.githubusercontent.com/110883172/215765148-77abb25a-a181-49cf-990d-76d3c6ae1860.png)
- 아날로그 신호인 음성신호를 분석하기 위해서 디지털 신호로 변환해주어야 한다.

![image](https://user-images.githubusercontent.com/110883172/215765212-cfddedf6-eea5-4972-a31e-4f2419d04206.png)
- 음성신호는 타임도메인의 연속적인 데이터.  타임도메인에서 주파수 데이터를 얻는 것이 FFT. 음성은 다양한 음성이 합쳐져서 시간 영역 전체를 FFT를 하면 원하는 결과를 얻지 못 할 수 있다.  그래서 시간 영역을 짧게 쪼개어서 각 영역마다 FFT를 수행하는 개념인 STFT하면 타임도메인의 스펙트럼이 나온다,

![image](https://user-images.githubusercontent.com/110883172/215765240-3f171e71-ba05-411e-b7dd-7f013d385435.png)
- 사람은 저주파 대역에 더 민감하다. 그래서 스펙트럼에 mel scale을 기반한 filter bank인 mel filter bank를 적용시켜서 mel spectrum을 얻을 수 있다. 이를 다시  x축을 time domain으로, megnitude 를 dB로 변환해서 그려주면 mel spectrogram이 나온다.

- mel spectrum에 cepstral 분석 방식을 거치면 학습에 사용하기 좋은 mfcc가 나온다 *MFCC는 오디오 신호에서 추출할 수 있는 feature로, 소리의 고유한 특징을 나타내는 수치

#### 결과물
![image](https://user-images.githubusercontent.com/110883172/215765272-6cf185e5-580c-4b3c-b330-fe92705585e5.png)
- Sliding - Window 방식을 통해 네, 아니요 판별

![image](https://user-images.githubusercontent.com/110883172/215765383-fe1ff203-aad5-492a-9e86-45c28507dd35.png)


</br>
</br>
</br>

### ◾ APP 설계
- 메인화면

![image](https://user-images.githubusercontent.com/110883172/213055508-68308d52-acf1-4ed7-b485-2616b79bcb7c.png)

- 회원가입

![image](https://user-images.githubusercontent.com/110883172/213055544-844fd3b2-0cd5-4ebb-af68-91f2b15e9d69.png)

- 운동기록

![image](https://user-images.githubusercontent.com/110883172/213055607-efb77b1e-9908-435b-998f-14781bc1ed89.png)

