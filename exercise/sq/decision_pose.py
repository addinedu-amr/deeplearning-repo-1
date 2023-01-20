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

# 스쿼트 판단용 코드



def play_sound(q):
  while True:
    data = q.get()
    if data is None:
      break
    data = str(data)
    if len(data) == 1:
      if data == "0":
        playsound("./pp.mp3")
        
      elif data[-1] > "0":
        playsound("./sound/" + str(data) + ".mp3" )
  
    if len(data) == 2:
      if data[-1] == "0":
        if data[0] == "1":
          playsound("./sound/10.mp3")
        
        elif data[0] > "1":
          playsound("./sound/" + data[0] + ".mp3" ) 
          playsound("./sound/10.mp3")
         
      elif data[-1] > "0":
        playsound("./sound/" + data[-1] + ".mp3" )
      



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def play_cam(q):

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose

  # 자세측정 지연시간?
  time_count = 0

  # 운동 카운트
  ex_count = 10

  # 운동 상태
  ex_status = 0

  status = "start"

  loaded_model = pickle.load(open('./val_model.sav', 'rb')) 

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

        Rangle_knee = calculate_angle(Rhip, Rknee, Rankle)
        Rknee_angle = 180-Rangle_knee

        Langle_knee = calculate_angle(Lhip, Lknee, Lankle)
        Lknee_angle = 180-Langle_knee

        Rangle_hip = calculate_angle(Rshoulder, Rhip, Rknee)
        Rhip_angle = 180-Rangle_hip

        Langle_hip = calculate_angle(Lshoulder, Lhip, Lknee)
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
              q.put(0)

            elif ex_status == 2:
              ex_status = 1
              status = "ready"
              ex_count += 1
              q.put(ex_count)

          elif 1 == np.argmax(loaded_model.predict_proba(pre).tolist()):
            if ex_status == 1:
              ex_status = 2
              status = "squat"
              q.put(0)
            
      except:
        print("model error")
          
          


      cv2.flip(image, 1)
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
        break

  cap.release()
  q.put(None)



if __name__ == '__main__':
  playsound("./sound/start.mp3")
  q = Queue()
  t1 = threading.Thread(target=play_sound, args=(q, ))
  t2 = threading.Thread(target=play_cam, args=(q, ))
  t2.start()
  t1.start()
  
