# -*- encoding : utf-8 -*-

# 10個の labelとデータのパスを指定します。
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
data_path = '~/.kaggle/competitions/tensorflow-speech-recognition-challenge' 

from glob import glob
import random
import os
import numpy as np

SEED = 2018

# リストをランダムにシャッフルする関数です。
def random_shuffle(lst):
    random.seed(SEED)
    random.shuffle(lst)
    return lst

# テキストファイルを保存するフォルダーを生成します。
if not os.path.exists('input'):
    os.mkdir('input')

# 訓練データ全体をまず trn_all.txtに保存します。
trn_all = []
trn_all_file = open('input/trn_all.txt', 'w')
# 提供された訓練データのパスをすべて読み込みます。
files = glob(data_path + '/train/audio/*/*.wav')
for f in files:
    # 背景騒音は skipします。
    if '_background_noise_' in f:
        continue

    # 正答値(label)と話者(speaker)の情報をファイル名から抽出します。
    label = f.split('/')[-2]
    speaker = f.split('/')[-1].split('_')[0]
    if label not in labels:
        # 10個のlabelとデータを20%の確率でunknownに分類し、追加します。
        label = 'unknown'
        if random.random() < 0.2:
            trn_all.append((label, speaker, f))
            trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
    else:
        trn_all.append((label, speaker, f))
        trn_all_file.write('{},{},{}\n'.format(label, speaker, f))
trn_all_file.close()


# 訓練データを話者を基準として 9:1 の比率で分離します。
uniq_speakers = list(set([speaker for (label, speaker, path) in trn_all]))
random_shuffle(uniq_speakers)
cutoff = int(len(uniq_speakers) * 0.9)
speaker_val = uniq_speakers[cutoff:]

# 交差検証用のファイルを生成します。
trn_file = open('input/trn.txt', 'w')
val_file = open('input/val.txt', 'w')
for (label, speaker, path) in trn_all:
    if speaker not in speaker_val:
        trn_file.write('{},{},{}\n'.format(label, speaker, path))
    else:
        val_file.write('{},{},{}\n'.format(label, speaker, path))
trn_file.close()
val_file.close()

# テストデータに対してもテキストファイルを生成します。
tst_all_file = open('input/tst.txt', 'w')
files = glob(data_path + '/test/audio/*.wav')
for f in files:
    tst_all_file.write(',,{}\n'.format(f))
tst_all_file.close()
