# Imports
import pandas as pd
import os

def prep_dataset():
    ravdess = os.listdir('./Audio_Speech_Actors_01-24')
    ravdess.sort()
    ravdess_path = './Audio_Speech_Actors_01-24/'
    emotionList = [] # create a list for all the emotions
    actorList = [] # create a list for all the actors
    filePathList = [] # create a list to save each file path
    for actor in ravdess:
        actor_files = os.listdir(ravdess_path + actor)
        for f in actor_files:
            identifiers = f.split('.')[0].split('-')
            emotion = int(identifiers[2])
            if emotion == 1: emotionList.append("neutral")
            if emotion == 2: emotionList.append("calm")
            if emotion == 3: emotionList.append("happy")
            if emotion == 4: emotionList.append("sad")
            if emotion == 5: emotionList.append("angry")
            if emotion == 6: emotionList.append("fearful")
            if emotion == 7: emotionList.append("disgust")
            if emotion == 8: emotionList.append("surprised")
            actorList.append(int(identifiers[6]))
            filePathList.append(ravdess_path + actor + '/' + f)
    dataset = pd.concat([pd.DataFrame(actorList),pd.DataFrame(emotionList),pd.DataFrame(filePathList)],axis=1)
    dataset.columns = ["actor", "emotion", "filepath"]
    return dataset

original_dataset = prep_dataset()
# save the dataset to a csv so it can be used later
original_dataset.to_csv('dataset.csv', index=False)
print("Dataset has been created and is of size", original_dataset.shape)
