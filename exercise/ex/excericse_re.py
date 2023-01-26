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
import mysql.connector
import pyaudio
import wave
import librosa
from xgboost import XGBClassifier


#운동 클래스 정의
class ex_list():  
  #운동 종류랑 운동 숫자를 카운트
  today_ex_list = []
  complete_ex_list = ["squat", "0", "pushup", "0", "lunge", "0", "plank", "0"]
  complete_num = 1


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
    if self.this_time_ex == "squat":
        return "./sq_model.sav"
    elif self.this_time_ex == "lunge":
        return "./lunge_model.sav"
    elif self.this_time_ex == "pushup":
        return "./pushup_model.sav"
    elif self.this_time_ex == "plank":
        return "./plank_model.sav" 
         
    
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
          
        elif data == "stop":
            playsound("./sound/stop.mp3")
        
        elif data == "start":
            playsound("./sound/sijak.mp3")

        elif data == "exit":
            break

        elif data == "./sq_model.sav":
            playsound('./sound/sq.mp3')
            playsound("./sound/start.mp3")

        elif data == "./lunge_model.sav":
            playsound('./sound/lunge.mp3')
            playsound("./sound/start.mp3")

        elif data == "./pushup_model.sav":
            playsound('./sound/pushup.mp3')
            playsound("./sound/start.mp3")

        elif data == "./plank_model.sav":
            playsound('./sound/plank.mp3')
            playsound("./sound/start.mp3")


        # 스쿼트, 런지, 푸쉬업용
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

          # 플랭크용
        else:
            data = data.replace("p","")
            data = data.replace("l","")
            data = data.replace("a","")
            if len(data) == 2 and data[0] >= "1" and data[0] < "6":
              playsound("./sound/" + data + "c.mp3" )

            elif len(data) == 2 and data[0] >= "6" and data[0] <= "9":
              su = int(data)
              cho = su % 60

              if cho == 0:
                playsound("./sound/1m.mp3")
              
              else:
                playsound("./sound/1m.mp3")
                playsound("./sound/" + str(cho) + "c.mp3" )
            
            elif len(data) == 3:
              su = int(data)
              min = su / 60
              cho = su % 60

              playsound("./sound/" +str(int(min))+ "m.mp3")
              if cho != 0:
                playsound("./sound/" + str(cho) + "c.mp3" )

    

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

    #플랭크용
    num = 0
    ex_time = 0

    end = 0
  
    self.ex_this_time()
    if self.ex_sunseo == len(self.today_ex_list):
      end = 1

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


        if str(ex_count) == self.this_time_ex_count or str(num) == self.this_time_ex_count:
            ex_count = 0
            num == 0
            self.complete_ex_list[self.complete_num] = self.this_time_ex_count
            self.complete_num += 2

            self.ex_this_time()
            loaded_model = pickle.load(open(self.return_ex_model(), 'rb'))
            self.q.put(self.return_ex_model())
            ex_status = 0
            status = "start"  

            if self.ex_sunseo == len(self.today_ex_list):
              end = 1


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
          Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
          Rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]


          Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
          Lelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
          Lwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
          
          
          Rangle_knee = self.calculate_angle(Rhip, Rknee, Rankle)
          Rknee_angle = 180-Rangle_knee

          Langle_knee = self.calculate_angle(Lhip, Lknee, Lankle)
          Lknee_angle = 180-Langle_knee

          Rangle_hip = self.calculate_angle(Rshoulder, Rhip, Rknee)
          Rhip_angle = 180-Rangle_hip

          Langle_hip = self.calculate_angle(Lshoulder, Lhip, Lknee)
          Lhip_angle = 180-Langle_hip

          Langle_elbow = self.calculate_angle(Lshoulder, Lelbow, Lwrist)
          Lelbow_angle = 180-Langle_elbow

          Rangle_elbow = self.calculate_angle(Rshoulder, Relbow, Rwrist)
          Relbow_angle = 180-Rangle_elbow

        except:
          pass

        try:
          if time_count > 10:
            time_count = 0
            if self.this_time_ex == "squat" or self.this_time_ex == "lunge":
              pre = np.array([[Rshoulder[0], Rshoulder[1], Lshoulder[0], Lshoulder[1],
                              Rhip[0], Rhip[1], Lhip[0], Lhip[1], 
                              Rknee[0], Rknee[1], Lknee[0], Lknee[1], 
                              Rankle[0], Rankle[1], Lankle[0], Lankle[1], 
                              Rknee_angle, Lknee_angle, Rhip_angle, Lhip_angle]])

            elif self.this_time_ex == "plank" or self.this_time_ex == "pushup":
              pre = np.array([[Rshoulder[0], Rshoulder[1], Lshoulder[0], Lshoulder[1],
                              Rhip[0], Rhip[1], Lhip[0], Lhip[1], 
                              Rknee[0], Rknee[1], Lknee[0], Lknee[1], 
                              Rankle[0], Rankle[1], Lankle[0], Lankle[1], 
                              Rknee_angle, Lknee_angle, Rhip_angle, Lhip_angle,
                              Relbow[0], Relbow[1], Lelbow[0], Lelbow[1],
                              Rwrist[0], Rwrist[1], Lwrist[0], Lwrist[1],
                              Relbow_angle, Lelbow_angle
                              ]])

            #print(np.argmax(loaded_model.predict_proba(pre).tolist()))
            if self.this_time_ex != "plank":
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

            else:
              if 0 == np.argmax(loaded_model.predict_proba(pre).tolist()):
                if ex_status == 0:
                  ex_status = 1
                  status = "ready"
                  self.q.put("ready")

                elif ex_status == 2:
                  ex_status = 0
                  status = "stop"
                  self.q.put("stop")

              elif 1 == np.argmax(loaded_model.predict_proba(pre).tolist()):
                if ex_status == 1:
                  ex_status = 2
                  status = "start"
                  self.q.put("start")
                  num += 1

              
              
        except:
          pass
          #print("model")
        
        if self.this_time_ex == "plank":
          if ex_status == 2:
                  ex_time += 1
                  if ex_time == 25:
                    num += 1
                    if num % 10 == 0 and num != 0:
                      self.q.put("pla" + str(num))

                    ex_time = 0
            
          
        cv2.flip(image, 1)

        cv2.putText(image, self.this_time_ex, org=(30, 90), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)


        if self.this_time_ex != "plank":
          cv2.putText(image, str(ex_count), org=(30, 60), 
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
              color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)
          
        elif self.this_time_ex == "plank":
          cv2.putText(image, str(num), org=(30, 60), 
              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
              color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)

        cv2.putText(image, status, org=(30, 30), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,255),thickness=3, lineType=cv2.LINE_AA)


        cv2.imshow('MediaPipe Pose',image)
        time_count += 1


        if (cv2.waitKey(5) & 0xFF == 27) or (end == 1 and (str(num) == self.this_time_ex_count or str(ex_count) == self.this_time_ex_count)):
          playsound("./sound/finish.mp3")
          self.q.put("exit")
          break

    cv2.destroyAllWindows()
    cap.release()
    self.q.put(None)  

    if self.this_time_ex != "plank":
      self.complete_ex_list[self.complete_num] = ex_count

    else:
      self.complete_ex_list[self.complete_num] = ex_time

 


  def run(self):
    t1 = threading.Thread(target=self.ex_start)
    t2 = threading.Thread(target=self.play_sound)
    t1.start()
    t2.start()


    t2.join()


    t1.join()
    t2.join()
            




class DB_access():
    username = ""
    password = ""
    weight = 0
    destination = 0
    

    login_success = 0
    local = None
    history_df = None
    today_ex_list = []

    mul = 0
    
    def __init__(self):

        try:
            self.local = mysql.connector.connect(
                host = "localhost",
                port = 3306,
                user = "root",
                password = "1234",
                database = "ex"
                )
            
            self.cursor = self.local.cursor(buffered=True)
        except:
            print("데이터 베이스 읽기 오류입니다.")
        
    def login(self):
        username = input("ID >>")
        password = input("PW >>")
        try:
            self.username = username
            self.password = password

            sql = "select * from userinfo where user = '" + self.username + "';"
                    #sql = "select * from userinfo;"
            self.cursor.execute(sql)

            result = self.cursor.fetchall()
            string = str(result).split(",")[1]
            string = string.replace("'", "")
            string = string.replace(" ", "")

            if string == password:
                print("로그인 성공")
                self.login_success = 1

            else:
                print("비밀번호 오류")
      
        except:
            print("그런 사람 없어요.")
        
    def register(self):
        #create table userinfo( user varchar(50), password varchar(50), weight int, destination int );
        self.username = input("ID >>")
        self.password = input("PW >>")
        self.weight = input("weight >>")
        self.destination = input("destiny >>")
        
        sql = "select * from userinfo where user = '" + self.username + "';"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        
        try:
          if str(result) == "[]":
            time.sleep(1)
            sql = "insert into userinfo values('" + self.username + "', '"  + self.password + "', " + self.weight + ", " + self.destination + ");"
            self.cursor.execute(sql)
            self.local.commit()
            print("등록완료")
            sql = "insert into history values('" + self.username + "', 0, '" + self.return_today() + "', 0, '" + self.weight + "', '" + self.destination + "', 0, 0, 0, 0, 0);"
            time.sleep(1)
            self.cursor.execute(sql)
            self.local.commit()
            print("데이터 등록 완료")
          
          else:
              print("이미 존재하는 유저입니다.")
    
        except:
          print("알수없는 이유로 실패했습니다.")
            
    def see_ex_history(self):
        if self.login_success == 0:
            print("로그인을 먼저 진행하세요")
            return
        else:
            sql = "select * from history where user = '" + self.username + "';"
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.history_df = pd.DataFrame(result)
            self.history_df.columns = ['user', 'reward', 'day', 'cal', 'weight', 'destination', 'squrt', 'lunge', 'pushup', 'plank', "exday"]

    def insert_data(self, complete_ex_list):

      if complete_ex_list == self.today_ex_list:
        self.mul += 1

      cal = float(complete_ex_list[1]) * 0.5 + float(complete_ex_list[3]) * 0.5  + float(complete_ex_list[5]) * 0.5 + float(complete_ex_list[7]) * 0.33
      reward = int(cal)

      sql = "insert into history values('" + self.username + "', " + str(reward) + " , '" + self.return_today() + "', " + str(cal) + " , " + str(self.weight) + ", " + str(self.destination) + ", " + str(complete_ex_list[1]) + " , "  + str(complete_ex_list[3]) + " , " +  str(complete_ex_list[5]) + " , " + str(complete_ex_list[7]) + " , " + str(self.mul) + ");"
      self.cursor.execute(sql)
      self.local.commit()
      print("데이터 등록 완료")
        
    def return_today(self):
      year = dt.datetime.now().year
      month = dt.datetime.now().month
      day = dt.datetime.now().day

      return str(year) + "-"  + str(month) + "-" + str(day)
      
    def return_today_ex(self):
      self.weight = self.history_df["weight"][len(self.history_df)-1]
      self.destination = self.history_df["destination"][len(self.history_df)-1]


      if self.history_df["exday"][len(self.history_df)-1] % 2 == 0:
        self.mul = int(self.history_df["exday"][len(self.history_df)-1])
        if self.mul > 6:
          self.mul = 6

        # 30 20 30 60 // 시연용으로
        self.today_ex_list.append("squat")
        self.today_ex_list.append(str(5 + self.mul*5))
        
        self.today_ex_list.append("pushup")
        self.today_ex_list.append(str(5 + self.mul*5))

        self.today_ex_list.append("lunge")
        self.today_ex_list.append(str(5 + self.mul*5))
        
        self.today_ex_list.append("plank")
        self.today_ex_list.append(str(20 + self.mul*5))
        

        
        
    def run(self):
        while(True):
            sel = input("Login > 1 / regist > 2 / exit > 3 \n >>>>>")
            if sel == "1":
                self.login()
            elif sel == "2":
                self.register()
            elif sel == "3":
                break
                
            if self.login_success == 1:
                time.sleep(1)
                print("로그인 성공 메인메뉴로 이동")
                break
        self.see_ex_history()
        print("운동 정보를 가져오고 있습니다.")
        time.sleep(1)

        self.return_today_ex()
        print("오늘의 운동량을 가져옵니다.")
        time.sleep(1)


class AudioRecognition():
    def __init__(self):
        self.wav_dict = {'아니요' : 0,
                         '네' : 1,
                         '소음' : 2}
        
        self.reverse_dict = dict(map(reversed, self.wav_dict.items()))

    def load_model(self, filename='./xgb_model_with_noise.model'):
        filename = filename
        self.xgb_model = pickle.load(open(filename, 'rb'))

    def recording(self, seconds=3):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050 #SAMPLE_RATE

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("start recording...")

        frames = []
        self.seconds = seconds

        for i in range(0, int(RATE / CHUNK*seconds)):
            data = stream.read(CHUNK)
            frames.append(data)

        print('recording stopped')

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open("./output.wav", "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def get_mfccs(self):
        # Load the audio file
        audio, sr = librosa.load("./output.wav")

        # Define the window size and step size (in seconds)
        window_size = 1.0
        step_size = 0.3

        # Convert window size and step size to samples
        window_size_samples = int(window_size * sr)
        step_size_samples = int(step_size * sr)

        # Initialize an empty list to store the MFCCs
        self.mfccs = []

        # Iterate over the audio using the sliding window
        for i in range(0, len(audio) - window_size_samples, step_size_samples):
            # Get the current window of audio
            window = audio[i:i+window_size_samples]

            # Compute the MFCCs for the window
            mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=40)

            # Append the MFCCs to the list
            self.mfccs.append(mfcc)

        # Convert the list of MFCCs to a numpy array
        self.mfccs = np.array(self.mfccs)
    
    def check_answer(self):
        count_no = 0
        count_yes = 0
        count_noise = 0
        
        for mfcc in self.mfccs:
            listen = []
            mfcc =  pd.Series(np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))))
            listen.append([mfcc])
            listen_df = pd.DataFrame(listen, columns=['feature'])

            X_listen = np.array(listen_df.feature.tolist())

            # Use MFCCs as input to model
            # prediction = rfc.predict_proba(X_listen)
            prediction = self.xgb_model.predict_proba(X_listen)
            yes_or_no = self.reverse_dict[np.argmax(prediction)]
            # print(prediction)
            # print(yes_or_no)
            if yes_or_no == '아니요':
                count_no += 1
            elif yes_or_no == '네':
                count_yes += 1
            else:
                count_noise += 1
        print('네 :', count_yes, '아니요 :', count_no, '소음 :', count_noise)
        
        self.answer = '네'

        if count_no > count_yes:
            self.answer = '아니요'
            print('결론 :', self.answer)
        else:
            print('결론 :', self.answer)
        return self.answer


if __name__ == '__main__':
    db = DB_access()
    db.run()
    time.sleep(1)
    print("오늘의 운동량 " + str(db.today_ex_list))

    

    playsound("./sound/go.mp3")
    audio_recognition = AudioRecognition()
    audio_recognition.load_model()
    audio_recognition.recording(seconds=3)
    audio_recognition.get_mfccs()
    answer = audio_recognition.check_answer()

    if answer == "네":
      playsound('./sound/good.mp3')

    elif answer == "아니요":
      playsound('./sound/sick.mp3')
      audio_recognition = AudioRecognition()
      audio_recognition.load_model()
      audio_recognition.recording(seconds=3)
      audio_recognition.get_mfccs()
      answer2 = audio_recognition.check_answer()
      if answer2 == "네":
        playsound('./sound/down.mp3')
        db.today_ex_list[1]  = int(int(db.today_ex_list[1]) * 0.7)
        db.today_ex_list[3]  = int(int(db.today_ex_list[3]) * 0.7)
        db.today_ex_list[5]  = int(int(db.today_ex_list[5]) * 0.7)
        db.today_ex_list[7]  = int(int(db.today_ex_list[7]) * 0.7)
        
      elif answer2 == "아니요":
        playsound("./sound/good.mp3")

      print("오늘의 운동량 " + str(db.today_ex_list))
      time.sleep(1)
      print("곧 운동이 시작됩니다.")


    ex = ex_list(db.username, db.today_ex_list )
    ex.run()
    db.insert_data(ex.complete_ex_list)

    del ex
    del db

  #용민 : 스쿼트 스펠링틀림

  