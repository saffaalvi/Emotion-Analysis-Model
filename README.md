# Emotion-Analysis-Model

Deployed at [emotion-analysis-cnn.herokuapp.com](https://emotion-analysis-cnn.herokuapp.com/) with limited functionality (takes a few seconds to load). Can run this locally with Flask to get full functionality.

## Highlights and Features

- Speech-to-Text Converter
- Emotion Analysis Prediction
- Convolutional Neural Network Model has high accuracy
- Can process both live recordings and loaded audio files
- GitHub page provides three deployments:
  - Local through command-line
  - Flask Application
  - Deployed on Heroku for public access (limited functionality)

## Implementation

This repository contains the source codes for this program. The files it contains are as follows:

Local:
- **requirements.txt, Pipfile, Piplock**: All the required packages and versions needed to run the program files.
- **prep_dataset.py**: Takes the RAVDESS audio files and combines the actors, emotions, and file path of each audio file into a single dataset, which is saved as _dataset.csv_. This dataset is used later for feature extraction and to train and test the model.
- **extract_features.py**: Uses the dataset (_dataset.csv_) saved from prep_dataset.py and iterates through each file to extract the features of each audio sample. The features of the audio samples are also extracted after several data augmentation techniques have been applied such as adding white noise, stretching the audio, and changing the speed and pitch. We apply data augmentation methods to make our model invariant to small variations such as speed, white noise, stretch, pitch, etc. and enhance its ability to generalize. It acts as a regularizer and helps reduce overfitting when training a machine learning model. This program outputs all the extracted features of all the RAVDESS audio samples into a single dataset, which is saved as _data_features.csv_ to be used to train and test the model.
- **model.py**: Creates the Convolutional Neural Network (CNN) model that is used to predict the emotion of audio samples. The model is compiled with the Adam optimizer and it trains and tests based off the _data_features.csv_ previously extracted. Saves the label encodings (_labels.npy_), scaler (_scaler.save_), and the model in .json format (_model.json_) and as _model.h5_ to load the weights. 
- **predict.py**: Contains a speech-to-text converter and loads the CNN to use for audio predictions. Have the option to either process and predict from an audio file (.wav only) or from a live audio from your device microphone. Will then output the transcribed audio text and the predicted emotion from the CNN model.
  - To predict from an audio file: `python predict.py <filename>`. Please enter the full filename and relative path.
  - To predict from a live audio: `python predict.py record`. This will record from your device microphone for the duration of 3 seconds, then display the transcribed text and predicted emotion. *May have to change sounddevice.rec parameter ‘channels’ to either 1 or 2 based on device.
- **ravdess.wav**: sample audio file from RAVDESS dataset.

Flask:
- **predict2.py**: Same as predict.py but just returns some outputs for Flask development.
- **app.py**: Code to run the application on Flask.
- **templates/**: Contain the html files to render the application.
- **style/css**: Defines styles for HTML pages.

The dataset used was the [RAVDESS dataset](https://smartlaboratory.org/ravdess/) (which is too large to upload to this repository). If trying to run _prep_dataset.py_ again, will have to download the [Audio_Speech_Actors_01-24 file](https://zenodo.org/record/1188976#.Yl54V5OZP0o) from RAVDESS to get that data. However, this dataset has already been extracted into the .csv files so there is no need to access the RAVDESS dataset again to run this.

## Installing, Setting Up, and Executing the System

To test the program, the following steps were taken:

To set up the environement for local development and Flask:
1. Install python and pip
2. `git clone https://github.com/saffaalvi/Emotion-Analysis-Model.git`
3. `pip install pipenv`
4. `cd Emotion-Analysis-Model`
5. `pipenv install`
6. `pipenv shell`

If running on Ubuntu, or getting 'UnknownValueError', add in language parameter in [predict.py](https://github.com/saffaalvi/Emotion-Analysis-Model/blob/main/predict.py#L48):

`transcript = r.recognize_google(audio, key=None, language="en-US")`

## Test with Flask
7. `flask run`
8. navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) on browser

This provides a user interface to load and test our model as well as other information about out our project.
To test the model, you can either load a .wav audio file (make sure it is in the same directory as app.py) and then click "Load Audio File" or can select 
"Record Live Audio" to record audio from the device microphone (duration of 3 seconds). The page will then redirect to the speech-to-text conversion and model prediction.

## Test through command line
To test the model and speech-to-text converter from a live audio:

7. `python predict.py record`
8. When prompted, record a 3 second audio from device microphone
9. Program will display transcription and model prediction.

To test the model from an audio file (.wav only):

7. `python predict.py <file-name-with-full-relative-path>`
8. Program will display transcription and model prediction.

## Run all Python files:
Run in this order to ensure all files are properly generated.

7. Download the RAVDESS Dataset as Audio_Speech_Actors_01-24/ in the home directory.
8. `python prep_dataset.py`
9. `python extract_features.py`
10. `python model.py`
11. `python predict.py record` or `python predict.py <file-name-with-full-relative-path>`

## Limitations
On Heroku, we are unable to upload files and get the model to predict the emotion due to size limitations and not being able to access the uploaded file object to load from librosa. The Record Live Audio functionality also does not work as we could not access a user's device microphone through the cloud. Model predictions and live audio processing therefore can only be made by deploying locally.

Deploying on Flask, the only limitation is that the selected file has to be in the home directory (same as `app.py`) so user will have to move the desired audio file into that directory. This was due to not being able to get the filepath of the loaded object and we were unable to convert it from a BytesIO object to a .wav file.

We were unable to upload the RAVDESS dataset onto this repository so if a user wants to run `prep_dataset.py` or access RAVDESS audio files, they will have to download it themselves. Running `extract_features.py` also takes a few minutes as it iterates through all of the RAVDESS files. However, saving the data to a .csv ensured this would only have to be run once.


