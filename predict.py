import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import librosa
import joblib 
import sys
import sounddevice
import speech_recognition as sr
import wavio

# Load scaler
scaler = joblib.load('scaler.save')

# Load the previously saved model
json_model_file = open('./model.json', 'r')
json_model = json_model_file.read()
json_model_file.close()
model = model_from_json(json_model)
model.load_weights('./model.h5')
print("Loaded model!")

# Load the labels
labels = np.load('./labels.npy', allow_pickle=True)
labels = list(labels)

# MFCC feature extraction
def extract_features(y, sample_rate):
  sample_rate = np.array(sample_rate)
  mfccs = librosa.feature.mfcc(y=y, sr = sample_rate, n_mfcc = 40)
  mfccs_mean = np.mean(mfccs, axis = 1)
  return mfccs_mean

def process(features):
  features_df = pd.concat([pd.DataFrame(features)], axis=1) # change to dataframe
  features_df = features_df.T
  features_df = scaler.transform(features_df)
  features_dim = np.expand_dims(features_df, axis=2)      # expand dimensions
  return features_dim

# convert audio from file to text
def speech_to_text(file):
  r = sr.Recognizer()
  print("Converting your audio to text...")
  fn = sr.AudioFile(file)
  with fn as source:
      audio = r.record(source)
  transcript = r.recognize_google(audio, key=None)
  print(transcript)

# predict the emotion of the audio with the previously loaded model
def predict(path):
  # Calculate the audio time series and its sampling rate
  print("Predicting the emotion for a sample audio...")
  x, sample_rate = librosa.load(path, res_type='kaiser_fast') 
  features = extract_features(x, sample_rate)
  features = process(features)
  predict = model.predict(features)               
  prediction = np.argmax(predict)
  print("The model has predicted class/label", prediction, "which is the emotion", labels[prediction])

# get live audio from user if they choose this option
def get_audio():
    print("Recording...")
    rate = 44100  # samples per second
    myrecording = sounddevice.rec(int(3 * 44100), samplerate=rate, channels=1)
    sounddevice.wait()
    wavio.write('./output.wav', myrecording, rate, sampwidth=2)
    speech_to_text('./output.wav')
    predict('./output.wav')

# command line args
opt = str(sys.argv[1])

# user wants to record
if opt == 'record':
  print("You have chosen to record, the recording will be for a duration of 3 seconds.")
  get_audio()
# user passes filename 
else:
  speech_to_text(opt)
  predict(opt)



