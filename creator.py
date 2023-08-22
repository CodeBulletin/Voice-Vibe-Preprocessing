import configparser
import pandas as pd
import os

config = configparser.ConfigParser()
config.read('settings.ini')

max_count = int(config['Main']['count'])

dataset_names = config['Datasets']['datasets_names'].split(',')
download_loc = config['Main']['dataset_download_folder']

output_loc = config['Main']['output_folder']
output_file = config['Main']['output_file']

if not os.path.exists(output_loc):
    os.mkdir(output_loc)

print(dataset_names)

df = pd.DataFrame([], columns=['FileName', 'FileDir', 'Emotion', 'Intensity'])


# CREMA-D
def getEmotionFromCREMAFile(file):
    emo = file.split("_")[2]
    if emo == "SAD":
        return "sad"
    elif emo == "ANG":
        return "angry"
    elif emo == "DIS":
        return "disgust"
    elif emo == "FEA":
        return "fear"
    elif emo == "HAP":
        return "happy"
    elif emo == "NEU":
        return "neutral"
    
def getIntensityFromCREMAFile(file):
    intensity = file.split("_")[3][0:2]
    if intensity == "LO":
        return "low"
    elif intensity == "MD":
        return "medium"
    elif intensity == "HI":
        return "high"
    elif intensity == "XX":
        return "unknown"

audios = os.listdir(download_loc + "/" + dataset_names[0] + "/AudioWAV")

for audio in audios[:max_count]:
    df.loc[len(df)] = [audio, download_loc + "/" + dataset_names[0] + "/AudioWAV", getEmotionFromCREMAFile(audio), getIntensityFromCREMAFile(audio)]


# RAVDESS
def getEmotionFromRAVDESSFile(file):
    emo = file.split("-")[2]
    if emo == "01":
        return "neutral"
    elif emo == "02":
        return "calm"
    elif emo == "03":
        return "happy"
    elif emo == "04":
        return "sad"
    elif emo == "05":
        return "angry"
    elif emo == "06":
        return "fearful"
    elif emo == "07":
        return "disgust"
    elif emo == "08":
        return "surprised"
    
def getIntensityFromRAVDESSFile(file):
    intensity = file.split("-")[3]
    if intensity == "01":
        return "medium"
    elif intensity == "02":
        return "high"
    
folders = os.listdir(download_loc + "/" + dataset_names[1] + "/audio_speech_actors_01-24")

audios = []
af = []
for folder in folders:
    ls = os.listdir(download_loc + "/" + dataset_names[1] + "/audio_speech_actors_01-24/" + folder)
    audios += ls
    af += [folder] * len(ls)


for audio, folder in list(zip(audios, af))[:max_count]:
    df.loc[len(df)] = [audio, download_loc + "/" + dataset_names[1] + "/audio_speech_actors_01-24/" + folder, getEmotionFromRAVDESSFile(audio), getIntensityFromRAVDESSFile(audio)]


# TESS
def getEmotionFromTESSFile(file):
    emo = file.split("_")[2][:-4]
    if emo == "angry":
        return "angry"
    elif emo == "disgust":
        return "disgust"
    elif emo == "fear":
        return "fear"
    elif emo == "happy":
        return "happy"
    elif emo == "ps":
        return "surprised"
    elif emo == "sad":
        return "sad"
    elif emo == "neutral":
        return "neutral"
    
def getIntensityFromTESSFile(file):
    return "medium"

folders = os.listdir(download_loc + "/" + dataset_names[2] + "/TESS Toronto emotional speech set data")

audios = []
af = []
for folder in folders:
    ls = os.listdir(download_loc + "/" + dataset_names[2] + "/TESS Toronto emotional speech set data/" + folder)
    audios += ls
    af += [folder] * len(ls)

for audio, folder in list(zip(audios, af))[:max_count]:
    df.loc[len(df)] = [audio, download_loc + "/" + dataset_names[2] + "/TESS Toronto emotional speech set data/" + folder, getEmotionFromTESSFile(audio), getIntensityFromTESSFile(audio)]


# SAVEE
def getEmotionFromSAVEEFile(file):
    emo = file.split("_")[1][:-6]
    if emo == "a":
        return "angry"
    elif emo == "d":
        return "disgust"
    elif emo == "f":
        return "fear"
    elif emo == "h":
        return "happy"
    elif emo == "n":
        return "neutral"
    elif emo == "sa":
        return "sad"
    elif emo == "su":
        return "surprised"
    
def getIntensityFromSAVEEFile(file):
    return "medium"

audios = os.listdir(download_loc + "/" + dataset_names[3] + "/ALL")

for audio in audios[:max_count]:
    df.loc[len(df)] = [audio, download_loc + "/" + dataset_names[3] + "/ALL", getEmotionFromSAVEEFile(audio), getIntensityFromSAVEEFile(audio)]


# Save to CSV
df.to_csv(output_loc + "/" + output_file, index=False)