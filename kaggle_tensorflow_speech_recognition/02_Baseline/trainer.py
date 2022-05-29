"""
model trainer
"""
from torch.autograd import Variable
from data import SpeechDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from resnet import ResModel
from tqdm import tqdm

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_time(now, start):
    time_in_min = int((now - start) / 60)
    return time_in_min

# 学習のための基本設定値を指定します。
BATCH_SIZE = 32  # データの束に該当する batch_sizeは GPU メモリーにあわせて指定します。
mGPU = False  # multi-GPUを使用する場合は Trueに指定します。
epochs = 20  # モデルが訓練データを学習する回数を指定します。
mode = 'cv' # 交差検証モード(cv) or テストモード(test)
model_name = 'model/model_resnet.pth'  # モデルの結果を指定するときにモデル名を指定します。

# ResNet モデルを活性化します。
loss_fn = torch.nn.CrossEntropyLoss()
model = ResModel
speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
speechmodel = speechmodel.cuda()

# SpeechDatasetを活性化します。
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_to_int = dict(zip(labels, range(len(labels))))
int_to_label = dict(zip(range(len(labels)), labels))
int_to_label.update({len(labels): 'unknown', len(labels) + 1: 'silence'})

# モードによって学習および検証に使うファイルを選択します。
trn = 'input/trn.txt' if mode == 'cv' else 'input/trn_all.txt'
tst = 'input/val.txt' if mode == 'cv' else 'input/tst.txt'

trn = [line.strip() for line in open(trn, 'r').readlines()]
wav_list = [line.split(',')[-1] for line in trn]
label_list = [line.split(',')[0] for line in trn]
# 学習用 SpeechDatasetを呼び出します。
traindataset = SpeechDataset(mode='train', label_to_int=label_to_int, wav_list=wav_list, label_list=label_list)

start_time = time()
for e in range(epochs):
    print("training epoch ", e)
    # learning_rateを epochごとに異なって指定します。
    learning_rate = 0.01 if e < 10 else 0.001
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)
    # モデルを学習するため .train() 関数を実行します。
    speechmodel.train()

    total_correct = 0
    num_labels = 0
    trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)
    # 学習を実行します。
    for batch_idx, batch_data in enumerate(tqdm(trainloader)):
        # batch_size だけの音声データ(spec)と正答値(label)を受け取ります。
        spec = batch_data['spec']
        label = batch_data['label']
        spec, label = Variable(spec.cuda()), Variable(label.cuda())
        # 現在のモデルの予測値(y_pred)を計算します。
        y_pred = speechmodel(spec)
        _, pred_labels = torch.max(y_pred.data, 1)
        correct = (pred_labels == label.data).sum()
        # 正答と予測値の差異(loss)を計算します。
        loss = loss_fn(y_pred, label)

        total_correct += correct
        num_labels += len(label)
    
        optimizer.zero_grad()
        # lossをもとにバックプロパゲーションを遂行します。
        loss.backward()
        # モデルのパラメータをアップデートします。 (実質的学習)
        optimizer.step()
    
    # 訓練データにおける正確率を記録します。
    print("training accuracy:", 100. * total_correct / num_labels, get_time(time(), start_time))

    # 交差検証モードの場合、検証データに対する正確率を記録します。
    if mode == 'cv':
        # 現在学習中のモデルを臨時に保存します。
        torch.save(speechmodel.state_dict(), '{}_cv'.format(model_name))
        
        # 検証データを呼び出します。
        softmax = Softmax()
        tst_list = [line.strip() for line in open(tst, 'r').readlines()]
        wav_list = [line.split(',')[-1] for line in tst_list]
        label_list = [line.split(',')[0] for line in tst_list]
        cvdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
        cvloader = DataLoader(cvdataset, BATCH_SIZE, shuffle=False)

        # モデルを呼び出し、.eval() 関数で検証を準備します。
        speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
        speechmodel.load_state_dict(torch.load('{}_cv'.format(model_name)))
        speechmodel = speechmodel.cuda()
        speechmodel.eval()

        # 検証データをbatch_sizeだけ受け取り、予測値を保存します。
        fnames, preds = [], []
        for batch_idx, batch_data in enumerate(tqdm(cvloader)):
            spec = Variable(batch_data['spec'].cuda())
            fname = batch_data['id']
            y_pred = softmax(speechmodel(spec))
            preds.append(y_pred.data.cpu().numpy())
            fnames += fname

        preds = np.vstack(preds)
        preds = [int_to_label[x] for x in np.argmax(preds, 1)]
        fnames = [fname.split('/')[-2] for fname in fnames]
        num_correct = 0
        for true, pred in zip(fnames, preds):
            if true == pred:
                num_correct += 1

        # 検証データの正確率を記録します。
        print("cv accuracy:", 100. * num_correct / len(preds), get_time(time(), start_time))

# 学習が完了したモデルを保存します。
create_directory("model")
torch.save(speechmodel.state_dict(), model_name)

# テストデータに対する予測値をファイルに保存します。
print("doing prediction...")
softmax = Softmax()

# テストデータを呼び出します。
tst = [line.strip() for line in open(tst, 'r').readlines()]
wav_list = [line.split(',')[-1] for line in tst]
testdataset = SpeechDataset(mode='test', label_to_int=label_to_int, wav_list=wav_list)
testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

# モデルを呼び出します。
speechmodel = torch.nn.DataParallel(model()) if mGPU else model()
speechmodel.load_state_dict(torch.load(model_name))
speechmodel = speechmodel.cuda()
speechmodel.eval()
    
test_fnames, test_labels = [], []
pred_scores = []

# テストデータに対する予測値を計算します。
for batch_idx, batch_data in enumerate(tqdm(testloader)):
    spec = Variable(batch_data['spec'].cuda())
    fname = batch_data['id']
    y_pred = softmax(speechmodel(spec))
    pred_scores.append(y_pred.data.cpu().numpy())
    test_fnames += fname

# もっとも高い確率値を持つ予測値を label 形態で保存します。
final_pred = np.vstack(pred_scores)
final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]
test_fnames = [x.split("/")[-1] for x in test_fnames]

# テストファイル名と予測値をsubフォルダーの下に保存します。
# Kaggleに直接アップロードできるファイルフォーマットです。
create_directory("sub")
pd.DataFrame({'fname': test_fnames, 'label': final_labels}).to_csv("sub/{}.csv".format(model_name.split('/')[-1]), index=False)
