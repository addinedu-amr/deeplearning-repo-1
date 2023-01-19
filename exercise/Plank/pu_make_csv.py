import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime as dt

# 스쿼트 측정용 코드
def return_today():
    year = dt.datetime.now().year
    month = dt.datetime.now().month
    day = dt.datetime.now().day
    hour = dt.datetime.now().hour
    minute = dt.datetime.now().minute
    second = dt.datetime.now().second
        
    return str(year) + "_"  + str(month) + "_" +\
           str(day) + "_"  + str(hour) + "_"  +\
           str(minute) + "_"  + str(second)



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

col_list = ["Rshoulder_x", "Rshoulder_y",
            "Lshoulder_x", "Lshoulder_y", 
            "Rhip_x", "Rhip_y",
            "Lhip_x", "Lhip_y", 
            "Rknee_x", "Rknee_y",
            "Lknee_x", "Lknee_y",
            "Rankle_x", "Rankle_y",
            "Lankle_x", "Lankle_y",
           
            "Rknee_angle", "Lknee_angle",
            
            "Rhip_angle", "Lhip_angle",
           
            "Relbow_x", "Relbow_y",
            "Lelbow_x", "Lelbow_y",
            
            "Rwrist_x", "Rwrist_y",
            "Lwrist_x", "Lwrist_y",
            
            "Relbow_angle", "Lelbow_angle",            
            "labels"]

           # labels 추가 해줘야함.

df = pd.DataFrame(columns = col_list)
count = 0


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
    # Flip the image horizontally for a selfie-view display.

    # 렌드마크가0~32 를 가진다.
   
    try:
      landmarks = results.pose_landmarks.landmark
      Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
      Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
      Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
      Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
      Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
      Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


      Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
      Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
      Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
      Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
      Lelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
      Lwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
      
      
      Rangle_knee = calculate_angle(Rhip, Rknee, Rankle)
      Rknee_angle = 180-Rangle_knee

      Langle_knee = calculate_angle(Lhip, Lknee, Lankle)
      Lknee_angle = 180-Langle_knee

      Rangle_hip = calculate_angle(Rshoulder, Rhip, Rknee)
      Rhip_angle = 180-Rangle_hip

      Langle_hip = calculate_angle(Lshoulder, Lhip, Lknee)
      Lhip_angle = 180-Langle_hip

      Langle_elbow = calculate_angle(Lshoulder, Lelbow, Lwrist)
      Lelbow_angle = 180-Langle_elbow

      Rangle_elbow = calculate_angle(Rshoulder, Relbow, Rwrist)
      Relbow_angle = 180-Rangle_elbow

      print("detect")
    
    

      if count > 7:
        df.loc[len(df)] = [Rshoulder[0], Rshoulder[1], Lshoulder[0], Lshoulder[1],
                           Rhip[0], Rhip[1], Lhip[0], Lhip[1], 
                           Rknee[0], Rknee[1], Lknee[0], Lknee[1], 
                           Rankle[0], Rankle[1], Lankle[0], Lankle[1], 
                           Rknee_angle, Lknee_angle, Rhip_angle, Lhip_angle,
                           Relbow[0], Relbow[1], Lelbow[0], Lelbow[1],
                           Rwrist[0], Rwrist[1], Lwrist[0], Lwrist[1],
                           Relbow_angle, Lelbow_angle, 2 ] # 끝에 라벨 추가 0: 준비 // 1: 플랭크자세 // 2: 틀린자세
        count = 0
      
    except:
      print("not find")

    print(count)
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    count += 1


    if cv2.waitKey(5) & 0xFF == 27:
      title = "./" + return_today() + '.csv'
      df.to_csv(title, encoding="UTF-8")
      break

cap.release()