import pickle
import pandas as pd
from i_Feature import *
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRAIN_SAMPLE = 66149
ROOT_PATH = '/Users/shijunshen/Documents/dataset/GTZAN/genres_original/'

data = pd.read_csv('/Users/shijunshen/Documents/dataset/GTZAN/features_30_sec.csv')
y_true = []
y_pred = []
errors = {'pred': [], 'gt': []}

for index, row in data.iterrows():
    filename = row['filename']
    path = ROOT_PATH + filename.split('.')[0].upper() + '/' + filename
    print(path)
    try:
        audio, sr = librosa.load(path)
    except Exception as e:
        continue
    original_length = len(audio)   # 661794

    split_part = []

    if original_length % TRAIN_SAMPLE == 0:         # 可以刚好分开
        split_number = original_length // TRAIN_SAMPLE
        for i in range(split_number):
            split_part.append(audio[i * TRAIN_SAMPLE: (i + 1) * TRAIN_SAMPLE])
    else:
        split_number = original_length // TRAIN_SAMPLE
        for i in range(split_number):
            split_part.append(audio[i * TRAIN_SAMPLE: (i + 1) * TRAIN_SAMPLE])
        split_part.append(audio[-TRAIN_SAMPLE:])

    # calculate features
    features = []
    for item in split_part:
        chroma_mean, chroma_var = chroma_stft(item, sr)
        rms_mean, rms_var = rms(item)
        sc_mean, sc_var = spectral_centroid(item, sr)
        sr_mean, sr_var = rolloff(item, sr)
        sb_mean, sb_var = spectral_bandwidth(item, sr)
        zcr_mean, zcr_var = zero_crossing_rate(item)
        t = tempo(item, sr)
        m_mean, m_var = mfcc(item, sr)

        feature = [chroma_mean, chroma_var, rms_mean, rms_var, sc_mean, sc_var, sb_mean, sb_var, sr_mean, sr_var, zcr_mean, zcr_var, t]
        for i in range(len(m_mean)):
            feature.append(m_mean[i])
            feature.append(m_var[i])
        features.append(feature)

    # print(features)
    # print(np.array(features).shape)
    column = 'chroma_stft_mean	chroma_stft_var	rms_mean	rms_var	spectral_centroid_mean	spectral_centroid_var	spectral_bandwidth_mean	spectral_bandwidth_var	rolloff_mean	rolloff_var	zero_crossing_rate_mean	zero_crossing_rate_var	tempo	mfcc1_mean	mfcc1_var	mfcc2_mean	mfcc2_var	mfcc3_mean	mfcc3_var	mfcc4_mean	mfcc4_var	mfcc5_mean	mfcc5_var	mfcc6_mean	mfcc6_var	mfcc7_mean	mfcc7_var	mfcc8_mean	mfcc8_var	mfcc9_mean	mfcc9_var	mfcc10_mean	mfcc10_var	mfcc11_mean	mfcc11_var	mfcc12_mean	mfcc12_var	mfcc13_mean	mfcc13_var	mfcc14_mean	mfcc14_var	mfcc15_mean	mfcc15_var	mfcc16_mean	mfcc16_var	mfcc17_mean	mfcc17_var	mfcc18_mean	mfcc18_var	mfcc19_mean	mfcc19_var	mfcc20_mean	mfcc20_var'
    column = column.split()
    features = pd.DataFrame(features, columns=column)

    # Load Model
    with open('./rf_3_model.pkl', 'rb') as f:
        rf_classifier = pickle.load(f)

    pred_list = rf_classifier.predict(features)
    counter = Counter(pred_list)
    max_count = max(counter.values())
    most_frequent = [item for item, count in counter.items() if count == max_count]

    # Voting System
    if len(most_frequent) >= 2:
        tmp = ''
        for i in range(len(most_frequent)):
            tmp += ' ' + most_frequent[i]
        y_pred.append(tmp)
        errors['pred'].append(y_pred)
        errors['gt'].append(row['label'])
    else:
        y_pred.append(most_frequent[0])
        if most_frequent[0] != row['label']:
            errors['pred'].append(most_frequent[0])
            errors['gt'].append(row['label'])
    y_true.append(row['label'])

print(pd.DataFrame(errors))

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy：", accuracy)

# 计算精确率
precision = precision_score(y_true, y_pred, average='macro')
print("Precision：", precision)

# 计算召回率
recall = recall_score(y_true, y_pred, average='macro')
print("Recall：", recall)

# 计算F1分数
f1 = f1_score(y_true, y_pred, average='macro')
print("F1 Score：", f1)
