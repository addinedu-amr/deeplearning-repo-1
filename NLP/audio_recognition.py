import pyaudio
import wave
import librosa
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

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
    audio_recognition = AudioRecognition()
    audio_recognition.load_model()
    audio_recognition.recording(seconds=3)
    audio_recognition.get_mfccs()
    answer = audio_recognition.check_answer()
