import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys

df = pd.read_csv('output/datasets_v3/dataset_s4.csv')
df['Emotion'] = df['Emotion'].replace('fearful', 'fear')
df['Emotion'] = df['Emotion'].replace('calm', 'neutral')

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

arr = []
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    mel = np.load(row['MelSpectrogram'])

    # convert to (128, 274) by padding
    mel = np.pad(mel, ((0, 0), (0, 274 - mel.shape[1])), 'constant', constant_values=0)

    arr.append(mel)

if not os.path.exists('output/datasets_v3/chunks'):
    os.makedirs('output/datasets_v3/chunks')
else:
    print('output/datasets_v3/chunks already exists')
    sys.exit()

chunck_size = 1000
for i in tqdm(range(0, len(arr), chunck_size)):
    np.save(f'output/datasets_v3/chunks/mel_spectrograms_{i}.npy', np.array(arr[i:min(i + chunck_size, len(arr))]))

# convert Emotion to one-hot
One_hot_emotions = pd.get_dummies(df['Emotion'])
One_hot_intensities = pd.get_dummies(df['Intensity'])

print(One_hot_emotions.head())
print(One_hot_intensities.head())

print(One_hot_emotions.shape)
print(One_hot_intensities.shape)

Emotions = One_hot_emotions.columns.values
Intensities = One_hot_intensities.columns.values

print(Emotions)
print(Intensities)

# save emotion and intensity to txt
with open('output/datasets_v3/emotion.txt', 'w') as f:
    for emotion in Emotions:
        f.write(emotion + '\n')

with open('output/datasets_v3/intensity.txt', 'w') as f:
    for intensity in Intensities:
        f.write(str(intensity) + '\n')

# save one-hot to npy
np.save('output/datasets_v3/one_hot_emotions.npy', One_hot_emotions.to_numpy())
np.save('output/datasets_v3/one_hot_intensities.npy', One_hot_intensities.to_numpy())