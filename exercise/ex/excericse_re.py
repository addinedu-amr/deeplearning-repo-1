import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime as dt
import pickle
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from playsound import playsound
import threading
import time
from queue import Queue

#운동 클래스 정의
class ex_list():  
  #운동 종류랑 운동 숫자를 카운트
  today_ex_list = []
  ex_name = "운동명"
  user_name = "유저이름"

  this_time_ex_len = 0
  this_time_ex = ""
  this_time_ex_count = ""

  ex_sunseo = 0

  loaded_model = None
  q = Queue()

  def __init__(self, user_name, today_ex_list ):
    self.user_name = user_name
    self.today_ex_list = today_ex_list
    self.this_time_ex_len =  len(self.today_ex_list)

  def ex_this_time(self):
    self.this_time_ex = self.today_ex_list[self.ex_sunseo]
    self.this_time_ex_count = self.today_ex_list[self.ex_sunseo+1]
    self.ex_sunseo += 2

  def return_ex_model(self):
    if self.this_time_ex == "squrt":
        return "./sq_model.sav"
    elif self.this_time_ex == "lunge":
        return "./lunge_model.sav"
         
    
  # 끝과 동시에 데이터베이스 클래스를 호출해야함
  def __del__(self):
    print("운동을 종료했습니다.")

  def calculate_angle(self, a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

  # 소리 재생용 메소드
  def play_sound(self):
    while True:
      data = self.q.get()
      if data is None:
        pass

      else: 
        if data == "ready":
            playsound("./sound/ready.mp3")
        
        elif data == "exit":
            break

        elif data == "./sq_model.sav":
            playsound('./sound/sq.mp3')
            playsound("./sound/start.mp3")

        elif data == "./lunge_model.sav":
            playsound('./sound/lunge.mp3')
            playsound("./sound/start.mp3")

        elif len(data) == 1:
          if data == "g":
            playsound("./sound/pp.mp3")

          elif data[-1] > "0":
            playsound("./sound/" + str(data) + ".mp3" )
      
        elif len(data) == 2:
          if data[-1] == "0":
            if data[0] == "1":
              playsound("./sound/10.mp3")
            
            elif data[0] > "1":
              playsound("./sound/" + data[0] + ".mp3" ) 
              playsound("./sound/10.mp3")
            
          elif data[-1] > "0":
            playsound("./sound/" + data[-1] + ".mp3" )

  def ex_start(self):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    # 자세측정 지연시간?
    time_count = 0
    # 운동 카운트
    ex_count = 0
    # 운동 상태
    ex_status = 0
    status = "start"

    self.ex_this_time()
    loaded_model = pickle.load(open(self.return_ex_model(), 'rb')) 
    self.q.put(self.return_ex_model())
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if str(ex_count) == self.this_time_ex_count:
            ex_count = 0
            self.ex_this_time()
            loaded_model = pickle.load(open(self.return_ex_model(), 'rb'))
            self.q.put(self.return_ex_model())

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.11

        # 렌드마크가0~32 를 가진다.
  
        try:
          landmarks = results.pose_landmarks.landmark
          Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
          Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
          Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
          Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
          
          Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

          Rangle_knee = self.calculate_angle(Rhip, Rknee, Rankle)
          Rknee_angle = 180-Rangle_knee

          Langle_knee = self.calculate_angle(Lhip, Lknee, Lankle)
          Lknee_angle = 180-Langle_knee

          Rangle_hip = self.calculate_angle(Rshoulder, Rhip, Rknee)
          Rhip_angle = 180-Rangle_hip

          Langle_hip = self.calculate_angle(Lshoulder, Lhip, Lknee)
          Lhip_angle = 180-Langle_hip

        except:
          pass

        try:
          if time_count > 10:
            time_count = 0
            pre = np.array([[Rshoulder[0], Rshoulder[1], Lshoulder[0], Lshoulder[1],
                              Rhip[0], Rhip[1], Lhip[0], Lhip[1], 
                              Rknee[0], Rknee[1], Lknee[0], Lknee[1], 
                              Rankle[0], Rankle[1], Lankle[0], Lankle[1], 
                              Rknee_angle, Lknee_angle, Rhip_angle, Lhip_angle
                              ]])
            

            #print(np.argmax(loaded_model.predict_proba(pre).tolist()))
            if 0 == np.argmax(loaded_model.predict_proba(pre).tolist()):
              if ex_status == 0:
                ex_status = 1
                status = "ready"
                self.q.put("ready")

              elif ex_status == 2:
                ex_status = 1
                status = "ready"
                ex_count += 1
                self.q.put(str(ex_count))

            elif 1 == np.argmax(loaded_model.predict_proba(pre).tolist()):
              if ex_status == 1:
                ex_status = 2
                status = "good"
                self.q.put("g")
              
        except:
          print("model")
            
          
        cv2.flip(image, 1)

        cv2.putText(image, self.this_time_ex, org=(30, 90), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)


        cv2.putText(image, str(ex_count), org=(30, 60), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)

        cv2.putText(image, status, org=(30, 30), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)


        cv2.imshow('MediaPipe Pose',image)
        time_count += 1


        if cv2.waitKey(5) & 0xFF == 27:
          playsound("./sound/finish.mp3")
          self.q.put("exit")
          break

    cv2.destroyAllWindows()
    cap.release()
    self.q.put(None)  
 


  def run(self):
    print("1")
    t1 = threading.Thread(target=self.ex_start)
    print("2")
    t2 = threading.Thread(target=self.play_sound)
    print("3")
    t1.start()
    print("4")
    t2.start()
    print("5")
    t1.join()
    t2.join()
            


class DBconnect:
  user_name = "유저이름"
  today_ex_count ="오늘의 할당량"

  # DB에 접속해 유저이름과 그동안의 운동기록을 가져온다.
  def __init__(self):
    pass

  # 넘겨받은 정보를 바탕으로 오늘의 운동량을 생성한다.
  def today_ex(self):
    pass


if __name__ == '__main__':
    ex_infor_list = ["squrt", "5", "lunge", "5"]
    ex = ex_list("kim", ex_infor_list )
    ex.run()
    del ex