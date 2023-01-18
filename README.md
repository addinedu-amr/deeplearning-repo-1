#  까꿍 PT
![image](https://user-images.githubusercontent.com/110883172/213055379-b0ee5d67-a418-4456-8e60-57412390a25a.png)


## 개요 
- 운동을 해야한다고 말만하고, 안하는 사람들이 있다. 그 이유는 운동을 하는 방법을 모르기 때문이라 생각한다. 그러니 딥러닝을 통해 자세를 교정해주고, 서로 양방향 소통이 가능한 프로그램을 제작하고자 한다.



## 팀원 및 역할 분배
#### 최선민 (총괄)
- 프로젝트 기획
- 운동의 정확한 자세 정의 및 측정 
- media pipe pose estimation 알고리즘 설계
- 데이터 수집(모션 캡처 및 음성녹음)

#### 김두엽
- 프로젝트 설계 (시간분배, 일정)
- 프로젝트 기록 및 설계도 작성
- media pipe 
- 데이터 수집(모션 캡처 및 음성녹음)

#### 박민제
- 프로젝트 설계 및 구체화 
- 팀원 어드바이스
- media pipe
- 데이터 수집(모션 캡처 및 음성녹음)

#### 박인
- 자연어처리 (양방향 소통)
- nltk
- 음성 API (마이크로 소프트, 구글) 를 활용한 인식
- 데이터 수집(모션 캡처 및 음성녹음)


#### 손훈민
- 자연어처리 (양방향 소통
- nltk
- 음성 API (마이크로 소프트, 구글) 를 활용한 인식
- 데이터 수집(모션 캡처 및 음성녹음)


## 설계
### 앱 UI 설계
![딥러닝 프로젝트 UI 설계 drawio (2)](https://user-images.githubusercontent.com/110883172/212559353-b19d15be-90f5-467c-a2ae-cbf922123681.png)




### 엑티브 다이어그램 설계
![딥러닝 엑티브 다이어그램 drawio (1)](https://user-images.githubusercontent.com/110883172/212559636-e462aa40-0f95-4b29-950d-64b6cfb9d7b6.png)




## 구현
### pose estimation
<center><img src="https://user-images.githubusercontent.com/110883172/213050490-fe6abc68-d3f8-423b-a4b7-73957f0c30f3.png" width="400" height="400"/></center>
- mediapipe를 사용하여, 각 관절의 좌표값을 읽어들임

#### 구현하고자 하는 운동 종류
- 스쿼트
- 팔굽혀펴기 (예정)

#### 데이터 출처
<center><img src="https://user-images.githubusercontent.com/110883172/213050058-c4b89369-5a06-4d10-b916-99b8ea3b83dd.png" width="400" height="700"/></center>
- 180도 방향 모션 캡처 (준비자세, 스쿼트자세, 틀린자세)


### 자연어처리




### APP (초안)
- 메인화면
![image](https://user-images.githubusercontent.com/110883172/213055508-68308d52-acf1-4ed7-b485-2616b79bcb7c.png)

- 회원가입
![image](https://user-images.githubusercontent.com/110883172/213055544-844fd3b2-0cd5-4ebb-af68-91f2b15e9d69.png)

- 운동기록
![image](https://user-images.githubusercontent.com/110883172/213055607-efb77b1e-9908-435b-998f-14781bc1ed89.png)

