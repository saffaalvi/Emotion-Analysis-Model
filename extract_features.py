# Get the original dataset from dataset.csv
# Extract the features (with data augmentation) --> data_features.csv

import numpy as np
import pandas as pd
import librosa
from sklearn.utils import shuffle

# Generate Data Frames for each augmentation method, use to extract features of each Data Frame
df = pd.DataFrame(columns=['feature'])
speedpitchDF = pd.DataFrame(columns=['feature'])
whitenoiseDF = pd.DataFrame(columns=['feature'])
stretchDF = pd.DataFrame(columns=['feature'])

# keep track of index
counter = 0     

# MFCC feature extraction
def extract_features(y, sample_rate):
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=y, sr = sample_rate, n_mfcc = 40)
    mfccs_mean = np.mean(mfccs, axis = 1)
    return mfccs_mean

# ---- DATA AUGMENTATION ----
# Stretch out the audio
def audioStretch(x):
    data = librosa.effects.time_stretch(x, rate=0.8)
    return data

# Add white noise in the background
def addWhiteNoise(x): 
    noise_amp = 0.05*np.random.uniform()*np.amax(x)   
    x = x.astype('float64') + noise_amp * np.random.normal(size = x.shape[0])
    return x

# Change the speed and pitch 
def speedPitch(x):
    timeChange = np.random.uniform(low=0.8, high = 1) # default values
    newSpeed = 1.4  / timeChange # generate a new speed used to change the speed and pitch
    interpolantTemp = np.interp(np.arange(0,len(x),newSpeed),np.arange(0,len(x)),x) # generate a temporary 1D linear interpolant
    minlen = min(x.shape[0], interpolantTemp.shape[0])
    x = x*0
    x[0:minlen] = interpolantTemp[0:minlen]
    return x

# Read dataset (from saved csv)
dataset = pd.read_csv('dataset.csv')

# Iterate through RAVDESS files
for file in dataset.filepath:
    y, sample_rate = librosa.load(file, res_type = 'kaiser_fast')
    # Extract feature of original audio
    feature = extract_features(y, sample_rate)
    df.loc[counter] = [feature]
    # Add white noise
    noise_X = addWhiteNoise(y)
    noise_feature = extract_features(noise_X, sample_rate)
    whitenoiseDF.loc[counter] = [noise_feature]
    # Add stretch
    stretch_X = audioStretch(y)
    stretch_feature = extract_features(stretch_X, sample_rate)
    stretchDF.loc[counter] = [stretch_feature]
    # Change speed and pitch
    sPtich_X = audioStretch(y)
    sPtich_feature = extract_features(sPtich_X, sample_rate)
    speedpitchDF.loc[counter] = [sPtich_feature]

    counter = counter+1

# Update Datasets with New Features
data_features = pd.concat([dataset, pd.DataFrame(df['feature'].values.tolist())], axis=1)            # original features
noise_dataset = pd.concat([dataset, pd.DataFrame(whitenoiseDF['feature'].values.tolist())], axis=1)  # white noise features
stretch_dataset = pd.concat([dataset, pd.DataFrame(stretchDF['feature'].values.tolist())], axis=1)   # stretch features
sPitch_dataset = pd.concat([dataset, pd.DataFrame(speedpitchDF['feature'].values.tolist())], axis=1) # speed and pitch features

print("Extracted features: ")
print(data_features.shape, noise_dataset.shape, stretch_dataset.shape, sPitch_dataset.shape)

# Combine all the different audio features into one dataset
data_features = pd.concat([data_features, noise_dataset, stretch_dataset, sPitch_dataset],axis=0,sort=False)

print("Size of new dataset with features:", data_features.shape)
print("Any null values:", data_features.isnull().values.any())

# Remove Filenames from dataset (don't need them anymore)
data_features.drop(columns='filepath',inplace=True)

# Shuffle the dataset so an equal distribution is present in the training and testing sets later
data_features = shuffle(data_features)

# Save features
data_features.to_csv('data_features.csv', index=False)
print("Saved to csv")
