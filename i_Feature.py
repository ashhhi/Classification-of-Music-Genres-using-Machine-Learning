import librosa
import essentia.standard as es

def chroma_stft(audio, sr):
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return chroma.mean(), chroma.var()

def rms(audio):
    rms = librosa.feature.rms(y=audio)
    return rms.mean(), rms.var()

def spectral_centroid(audio, sr):
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return sc.mean(), sc.var()

def spectral_bandwidth(audio, sr):
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return sb.mean(), sb.var()


def rolloff(audio, sr):
    sr = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    return sr.mean(), sr.var()

def zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    return zcr.mean(), zcr.var()

def tempo(audio, sr):
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo

def mfcc(audio, sr):
    mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr)
    return list(mfcc_feat.mean(1)), list(mfcc_feat.var(1))

