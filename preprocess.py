import pandas as pd
import swifter
import numpy as np
import librosa
import soundfile as sf
import configparser
from tqdm import tqdm
import os
from functions import reduce_sample_rate, pad_to_length, convert_to_spectrogram, convert_to_mel_spectrogram, convert_to_mfcc, convert_to_mfcc_extra, noise, stretch, shift, pitch, cutoffs, normalize

config = configparser.ConfigParser()
config.read('settings.ini')

max_count = int(config['Main']['count'])
stage = int(config['Main']['stage'])

"""
    Stage 1
        - Calculate AudioLength column
        - Calculate SampleRate column
"""

if stage == 1 or stage == 0:
    dataset_path = config['Main']['output_folder'] + '/' + config['Main']['output_file']
    output_path = config['Main']['output_folder'] + '/' + config['Main']['output_file_s1']

    df = pd.read_csv(dataset_path)

    # Add AudioLength column

    def get_audio_length(row):
        length = librosa.get_duration(path=(row['FileDir'] + '/' + row['FileName']))
        return length

    def get_sample_rate(row):
        sample_rate = librosa.get_samplerate(path=(row['FileDir'] + '/' + row['FileName']))
        return sample_rate

    tqdm.pandas()
    df['AudioLength'] = df.swifter.progress_bar(True, "Calculating Length").apply(get_audio_length, axis=1)
    df['SampleRate'] = df.swifter.progress_bar(True, "Calculating Samplerate").apply(get_sample_rate, axis=1)

    # Save to csv
    df.to_csv(output_path, index=False)


"""
    Stage 2
        - Reduce sample rate to 16kHz
        - Pad to 3.5 seconds
"""

if stage == 2 or stage == 0:

    output_path = config['Main']['output_dataset_folder']
    dataset_path = config['Main']['output_folder'] + '/' + config['Main']['output_file_s1']
    audio_length = float(config['Main']['audio_length'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(dataset_path)

    def Preprocess_s2(row):
        data, sample_rate = librosa.load(row['FileDir'] + '/' + row['FileName'], sr=None)
        data = reduce_sample_rate(data, sample_rate)
        data = pad_to_length(data, int(16000*audio_length))
        sf.write(output_path + '/' + row['FileName'], data, 16000, subtype='PCM_24')
        return output_path + '/' + row['FileName']
    
    df['file'] = df.swifter.progress_bar(True, "Resizeing").apply(Preprocess_s2, axis=1)

    df['AudioLength'] = df['AudioLength'].apply(lambda x: 3.5 if x > 3.5 else x)
    df.drop(['FileDir', 'FileName', 'SampleRate'], axis=1, inplace=True)

    df.to_csv(config['Main']['output_folder'] + '/' + config['Main']['output_file_s2'], index=False)

"""
    Stage 3
        - Add Noise, Pitch, Speed, Shift and Stretch
"""

if stage == 3 or stage == 0:
    dataset_path = config['Main']['output_folder'] + '/' + config['Main']['output_file_s2']
    output_path = config['Main']['output_dataset_folder_v2']

    df = pd.read_csv(dataset_path)
    print(df.head())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save(row, end, *funcs):
        data, sample_rate = librosa.load(row['file'], sr=16000)
        for func in funcs:
            data = func(data, sample_rate)
        sf.write(output_path + '/' + row['file'].split('/')[-1][:-4] + '_' + end + '.wav', data, sample_rate, subtype='PCM_24')
        row['file'] = output_path + '/' + row['file'].split('/')[-1][:-4] + '_' + end + '.wav'
        return row
    
    tqdm.pandas()
    dfa = df.swifter.progress_bar(True, "Noise").apply(lambda row: save(row, "noise", noise), axis=1)
    dfb = df.swifter.progress_bar(True, "Pitch").apply(lambda row: save(row, "pitch", pitch), axis=1)
    dfc = df.swifter.progress_bar(True, "Stretch&Shift").apply(lambda row: save(row, "stretch_shift", stretch, shift), axis=1)
    df = pd.concat([df, dfa, dfb, dfc], axis=0, ignore_index=True)

    df.to_csv(config['Main']['output_folder'] + '/' + config['Main']['output_file_s3'], index=False)

    print(len(df))

"""
    Stage 4
        - Convert to Spectrogram 
        - Convert to Mel Spectrogram 
        - Convert to MFCC
        - Convert to MFCC + Extra 
"""

if stage == 4 or stage == 0:
    dataset_path = config['Main']['output_folder'] + '/' + config['Main']['output_file_s3']
    save_path = config['Main']['output_dataset_folder_v3']

    df = pd.read_csv(dataset_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path + '/MFCCExtra'):
        os.makedirs(save_path + '/MFCCExtra')
    
    def Preprocess_s3(row):
        data, sample_rate = librosa.load(row['file'], sr=16000)

        file = row['file'].split('/')[-1][:-4] + '.npy'

        # Spectrogram
        # spectrogram = convert_to_spectrogram(data)
        # np.save(save_path + '/Spectrogram/' + file, spectrogram)

        # Mel Spectrogram
        # mel_spectrogram = convert_to_mel_spectrogram(data)
        # np.save(save_path + '/MelSpectrogram/' + file, mel_spectrogram)

        # MFCC
        # mfcc = convert_to_mfcc(data)
        # np.save(save_path + '/MFCC/' + file, mfcc)

        # MFCC Extra
        mfcc_extra = convert_to_mfcc_extra(data)
        np.save(save_path + '/MFCCExtra/' + file, mfcc_extra)

        # row['Spectrogram'] = save_path + '/Spectrogram/' + file
        # row['MelSpectrogram'] = save_path + '/MelSpectrogram/' + file
        # row['MFCC'] = save_path + '/MFCC/' + file
        row['MFCCExtra'] = save_path + '/MFCCExtra/' + file

        return row
    
    tqdm.pandas()
    df = df.swifter.progress_bar(True, "MelSpectrogram").apply(Preprocess_s3, axis=1)

    df.drop(['file'], axis=1, inplace=True)

    df.to_csv(config['Main']['output_folder'] + '/' + config['Main']['output_file_s4'], index=False)

"""
    Stage 5
        - Normalize
"""

# if stage == 5 or stage == 0:
#     dataset_path = config['Main']['output_folder'] + '/' + config['Main']['output_file_s4']
#     save_path = config['Main']['output_dataset_folder_v4']

#     df = pd.read_csv(dataset_path)

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     if not os.path.exists(save_path + '/Spectrogram'):
#         os.makedirs(save_path + '/Spectrogram')
    
#     if not os.path.exists(save_path + '/MelSpectrogram'):
#         os.makedirs(save_path + '/MelSpectrogram')

#     if not os.path.exists(save_path + '/MFCC'):
#         os.makedirs(save_path + '/MFCC')
    
#     if not os.path.exists(save_path + '/MFCCExtra'):
#         os.makedirs(save_path + '/MFCCExtra')

#     def Preprocess_s4(row):
#         spectrogram = np.load(row['Spectrogram'])
#         spectrogram = normalize(spectrogram)
#         np.save(save_path + '/Spectrogram/' + row['Spectrogram'].split('/')[-1], spectrogram)

#         mel_spectrogram = np.load(row['MelSpectrogram'])
#         mel_spectrogram = normalize(mel_spectrogram)
#         np.save(save_path + '/MelSpectrogram/' + row['MelSpectrogram'].split('/')[-1], mel_spectrogram)

#         mfcc = np.load(row['MFCC'])
#         mfcc = normalize(mfcc)
#         np.save(save_path + '/MFCC/' + row['MFCC'].split('/')[-1], mfcc)

#         mfcc_extra = np.load(row['MFCCExtra'])
#         mfcc_extra = normalize(mfcc_extra)
#         np.save(save_path + '/MFCCExtra/' + row['MFCCExtra'].split('/')[-1], mfcc_extra)

#         row['Spectrogram'] = save_path + '/Spectrogram/' + row['Spectrogram'].split('/')[-1]
#         row['MelSpectrogram'] = save_path + '/MelSpectrogram/' + row['MelSpectrogram'].split('/')[-1]
#         row['MFCC'] = save_path + '/MFCC/' + row['MFCC'].split('/')[-1]
#         row['MFCCExtra'] = save_path + '/MFCCExtra/' + row['MFCCExtra'].split('/')[-1]

#         return row
    
#     tqdm.pandas()
#     df = df.progress_apply(Preprocess_s4, axis=1)

#     df.to_csv(config['Main']['output_folder'] + '/' + config['Main']['output_file_s5'], index=False)