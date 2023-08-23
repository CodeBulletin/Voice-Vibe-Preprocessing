import librosa
import numpy as np

def reduce_sample_rate(data, sample_rate):
    if sample_rate == 16000:
        return data
    else:
        return librosa.resample(data, orig_sr=sample_rate, target_sr=16000)
    
def pad_to_length(data, length):
    if len(data) == length:
        return data
    elif len(data) > length:
        return data[:length]
    else:
        return np.pad(data, (0, length-len(data)), 'constant')
    
def convert_to_spectrogram(data):
    return librosa.stft(data, n_fft=512, hop_length=256)

def convert_to_mel_spectrogram(data):
    return librosa.feature.melspectrogram(y=data, sr=16000, n_fft=512, hop_length=256, n_mels=128)

def convert_to_mfcc(data):
    return librosa.feature.mfcc(y=data, sr=16000, n_fft=512, hop_length=256, n_mfcc=40)

def convert_to_mfcc_extra(data):
    # ZCR - Zero Crossing Rate
    result = np.array([])
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=512, hop_length=256)
    result = np.hstack((result, np.mean(zcr.T, axis=0)))

    # RMS - Root Mean Square
    rms = librosa.feature.rms(y=data, frame_length=512, hop_length=256)
    result = np.hstack((result, np.mean(rms.T, axis=0)))

    # Chroma STFT - Chromagram of a short-time Fourier transform
    chroma_stft = librosa.feature.chroma_stft(y=data, sr=16000, n_fft=512, hop_length=256)
    result = np.hstack((result, np.mean(chroma_stft.T, axis=0)))

    # MFCC - Mel-frequency cepstral coefficients
    mfcc = convert_to_mfcc(data)
    result = np.hstack((result, np.mean(mfcc.T, axis=0)))

    # Mel Spectrogram
    mel_spectrogram = convert_to_mel_spectrogram(data)
    result = np.hstack((result, np.mean(mel_spectrogram.T, axis=0)))

    return result

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def cutoffs(data):
    n = np.random.randint(1, 3)
    for i in range(n):
        f = int(np.random.uniform(0.0, 0.2) * len(data))
        # Select Random parts of length f and make it 0
        x = np.random.randint(0, len(data)-f)
        data[x: x+f] = 0
    return data

def normalize(data):
    return (data - np.mean(data)) / np.std(data)