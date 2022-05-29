import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random
from skimage.transform import resize
import pandas as pd
from random import sample

# sample_rateは 1秒あたり16,000
SR=16000

# SpeechDataset クラスを定義します。 torch.utils.dataの Dataset の属性を継承します。
class SpeechDataset(Dataset):

    def __init__(self, mode, label_words_dict, wav_list, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=None, is_1d=False):
        # Dataset 定義するための設定値を受け取ります。
        self.mode = mode
        self.label_words_dict = label_words_dict
        self.wav_list = wav_list[0]
        self.label_list = wav_list[1]
        self.add_noise = add_noise
        self.sr = sr
        self.n_silence = int(len(self.wav_list) * 0.09)
        self.preprocess_fun = preprocess_fun
        self.preprocess_param = preprocess_param

        # ノイズ追加のため_background_noise_は手動で読み込みます。必要な場合、パスを適合するように修正しましょう。
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../input/train/audio/_background_noise_/*.wav")]
        self.resize_shape = resize_shape
        self.is_1d = is_1d

    def get_one_noise(self):
        # ノイズを追加したサウンドファイルを返す
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def timeshift(self, wav, ms=100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    # データの大きさを変換します。test modeの場合は指定された音声データのリストの大きさ、train modeの場合は9% 追加した“沈黙”の件数を追加します。
    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    #  1つの音声データを読み込む関数です。
    def __getitem__(self, idx):
        if idx < len(self.wav_list):
            # test modeには音声データをそのまま読み込み、train modeには.get_noisy_wav() 関数を通してノイズが追加された音声データを読み込みます。
            wav_numpy = self.preprocess_fun(self.get_one_word_wav(idx) if self.mode != 'train' else self.get_noisy_wav(idx), **self.preprocess_param)

            # 読み込まれた音声の波形データをリサイジングします。
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)

            # test modeの場合、{spec, id}情報を変換し、train modeの場合は、{spec, id, label}情報を変換します。
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}

            label = self.label_words_dict.get(self.label_list[idx], len(self.label_words_dict))

            return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}

        # "沈黙" 音声データを任意に生成します。
        else:
            wav_numpy = self.preprocess_fun(self.get_silent_wav(num_noise=random.choice([0, 1, 2, 3]), max_ratio=random.choice([x / 10. for x in range(20)])), **self.preprocess_param)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            
            wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_words_dict) + 1}

    def get_noisy_wav(self, idx):
        # 音声の波形の高さを調整する scale
        scale = random.uniform(0.75, 1.25)
        # 追加するノイズの個数
        num_noise = random.choice([1, 2])
        # ノイズ音声の波形の高さを調整する max_ratio
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        # ノイズを追加する確率 mix_noise_proba
        mix_noise_proba = random.choice([0.1, 0.3])
        # 音声データを左右に平行移動する大きさ shift_range
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            # Data Augmentationを遂行します。
            return scale * (self.timeshift(one_word_wav, shift_range) + self.get_mix_noises(num_noise, max_ratio))
        else:
            # 原本の音声データをそのまま返します。
            return one_word_wav 


# 1次元音声の波形を2次元 mfccに変換する前処理関数です。
def preprocess_mfcc(wave):
    # librosa ライブラリを通して入力されたwaveデータを変換します。
    spectrogram = librosa.feature.melspectrogram(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    # 0より大きい値はlog関数をとります。
    idx = [spectrogram > 0]
    spectrogram[idx] = np.log(spectrogram[idx])

    # フィルターを使用してスペクトログラムデータに最後の前処理を遂行します。
    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)
    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]
    mfcc = np.hstack(mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc

# 1次元音声の波形を2次元melデータに変換する前処理関数です。
def preprocess_mel(data, n_mels=40, normalization=False):
    # librosaライブラリを通して入力されたwave データを変換します。
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    #  mel データを正規化します。
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram

# テストデータを sub_pathから呼び出す関数です。
def get_sub_list(num, sub_path):
    lst = []
    df = pd.read_csv(sub_path)
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
    each_num = int(num * 0.085)
    labels = []
    for w in words:
        # 12個の分類(words)に対してそれぞれ 1/12 の分量ずつ(each_num) ランダムにデータパスを指定します。
        tmp = df['fname'][df['label'] == w].sample(each_num).tolist()
        lst += ["../input/test/audio/" + x for x in tmp]
        for _ in range(len(tmp)):
            labels.append(w)
    return lst, labels

def get_semi_list(words, sub_path, unknown_ratio=0.2, test_ratio=0.2):
    # 訓練データのパスを呼び込みます。
    train_list, train_labels, _ = get_wav_list(words=words, unknown_ratio=unknown_ratio)
    # 訓練データの 20%~35%に該当する量だけテストデータのパスを呼び込みます。
    test_list, test_labels = get_sub_list(num=int(len(train_list) * test_ratio), sub_path=sub_path)
    file_list = train_list + test_list
    label_list = train_labels + test_labels
    assert(len(file_list) == len(label_list))

    # データのパスが指定された listの順序をランダムに混ぜ合わせます。
    random.seed(2018)
    file_list = sample(file_list, len(file_list))
    random.seed(2018)
    label_list = sample(label_list, len(label_list))

    return file_list, label_list


def get_label_dict():
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    label_to_int = dict(zip(words, range(len(words))))
    int_to_label = dict(zip(range(len(words)), words))
    int_to_label.update({len(words): 'unknown', len(words) + 1: 'silence'})
    return label_to_int, int_to_label


def get_wav_list(words, unknown_ratio=0.2):
    full_train_list = glob("../input/train/audio/*/*.wav")
    full_test_list = glob("../input/test/audio/*.wav")

    # sample full train list
    sampled_train_list = []
    sampled_train_labels = []
    for w in full_train_list:
        l = w.split("/")[-2]
        if l not in words:
            if random.random() < unknown_ratio:
                sampled_train_list.append(w)
                sample_train_labels.append('unknown')
        else:
            sampled_train_list.append(w)
            sampled_train_labels.append(l)

    return sampled_train_list, sampled_train_labels, full_test_list

def preprocess_wav(wav, normalization=True):
    data = wav.reshape(1, -1)
    if normalization:
        mean = data.mean()
        data -= mean
    return data
