import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime as dt


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


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

col_list = ["shoulder_x", "shoulder_y", 
            "hip_x", "hip_y", 
            "knee_x", "knee_y",
           "ankle_x", "ankle_y"]

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
      shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
      hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
      knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
      ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
      print("detect")

      if count > 10:
        df.loc[len(df)] = [shoulder[0], shoulder[1], hip[0], hip[1], knee[0], knee[1], ankle[0], ankle[1]]
        print("save")
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