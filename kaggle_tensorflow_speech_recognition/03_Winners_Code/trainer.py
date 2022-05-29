"""
model trainer
"""
from torch.autograd import Variable
from data import get_label_dict, get_wav_list, SpeechDataset, get_semi_list
from pretrain_data import PreDataset
from torch.utils.data import DataLoader
import torch
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice

# train_modelは総計13個の変数を入力値として受け取ります。
def train_model(model_class, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, preprocess_param={}, bagging_num=1, semi_train_path=None, pretrained=None, pretraining=False, MGPU=False):
    """
    :param model_class: model class. e.g. vgg, resnet, senet
    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave
    :param is_1d: boolean. True for conv1d models and false for conv2d
    :param reshape_size: int. only for conv2d, reshape the image size
    :param BATCH_SIZE: batch size.
    :param epochs: number of epochs
    :param CODER: string for saving and loading model/files
    :param preprocess_param: parameters for preprocessing function
    :param bagging_num: number of training per model, aka bagging models
    :param semi_train_path: path to semi supervised learning file.
    :param pretrained: path to pretrained model
    :param pretraining: boolean. if this is pretraining
    :param MGPU: whether using multiple gpus
    """
    # 学習に使用するモデルを定義する get_model() 関数です。
    def get_model(model=model_class, m=MGPU, pretrained=pretrained):
        # multi-GPUの場合、 Data Parallelism
        mdl = torch.nn.DataParallel(model()) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 事前学習した torch.load()でモデルを呼び出します。
            mdl.load_state_dict(torch.load(pretrained))
            if 'vgg' in pretrained:
                # VGG モデルの場合、最上位層のパラメータ以外のすべてのパラメータを学習しないようにrequires_grad=Falseに指定します。
                fixed_layers = list(mdl.features)
                for l in fixed_layers:
                    for p in l.parameters():
                        p.requires_grad = False
            return mdl

    label_to_int, int_to_label = get_label_dict()
    # bagging_num だけモデル学習を反復して遂行します。
    for b in range(bagging_num):
        print("training model # ", b)

        # 学習に使うloss functionを定義します。
        loss_fn = torch.nn.CrossEntropyLoss()

        # モデルを定義し、.cuda()で GPU、CUDAと連動します。
        speechmodel = get_model()
        speechmodel = speechmodel.cuda()

        # 学習の中間で性能を表示するための値を準備します。
        total_correct = 0
        num_labels = 0
        start_time = time()

        # 指定された epoch 分の学習を遂行します。
        for e in range(epochs):
            print("training epoch ", e)
            # 10 epoch 以後は learning_rateを 1/10に減らします。
            learning_rate = 0.01 if e < 10 else 0.001
            # 学習に使用するSGD optimizer + momentumを定義します。
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)

            # モデル内部のモジュールを学習の直前に活性化します。
            speechmodel.train()

            if semi_train_path:
                # 半教師あり学習の場合は訓練データを呼び出す基準が異なります。 [半教師あり学習のモデル学習]で詳しく扱います。
                # 学習に使用するファイルのリスト train_listにテストデータを追加します。
                train_list, label_list = get_semi_list(words=label_to_int.keys(), sub_path=semi_train_path,
                                           test_ratio=choice([0.2, 0.25, 0.3, 0.35]))
                print("semi training list length: ", len(train_list))
            else:
                # 教師あり学習の場合、訓練データリストを受け取ります。
                train_list, label_list, _ = get_wav_list(words=label_to_int.keys())

            if pretraining:
                traindataset = PreDataset(label_words_dict=label_to_int,
                                          add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                          resize_shape=reshape_size, is_1d=is_1d)
            else:
                traindataset = SpeechDataset(mode='train', label_words_dict=label_to_int, wav_list=(train_list, label_list),
                                             add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                             resize_shape=reshape_size, is_1d=is_1d)

            # Dataloaderを通してデータキューを生成します。Shuffle=True 設定を通してepochごとに読み込むデータをランダムに選定します。
            trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)

            # trainloaderを通してbatch_size 分の訓練データを読み込みます。
            for batch_idx, batch_data in enumerate(trainloader):

                #  specはスペクトログラムの略で音声データを意味し、labelは正答値を意味します。
                spec = batch_data['spec']
                label = batch_data['label']
                spec, label = Variable(spec.cuda()), Variable(label.cuda())

                # 現在のモデル(speechmodel)にデータ(spec)を入力し、予測結果(y_pred)を得ます。
                y_pred = speechmodel(spec)

                # 予測結果と正答値から現在のモデルの Loss値を求めます。
                loss = loss_fn(y_pred, label)
                optimizer.zero_grad()
                # backpropagationを遂行し、Loss値を改善するためにモデルのパラメータを修正すべき方法を獲得します。
                loss.backward()
                # optimizer.step() 関数によってモデルのパラメータをアップデートします。
                # これまでよりloss値が減少する方向でモデルのパラメータがアップデートされました。
                optimizer.step()

                # 確率値であるy_predでmax値を求め、現在のモデルの正確率(correct)を求めます。
                _, pred_labels = torch.max(y_pred.data, 1)
                correct = (pred_labels == label.data).sum()
                total_correct += correct
                num_labels += len(label)

            # 訓練データに対する正確率をその中間ごとに出力します。
            print("training loss:", 100. * total_correct / num_labels, time()-start_time)

        # 学習が完了したモデルのパラメータを保存します。
        create_directory("model")
        torch.save(speechmodel.state_dict(), "model/model_%s_%s.pth" % (CODER, b))

    if not pretraining:
        print("doing prediction...")
        softmax = Softmax()

        # 保存した学習モデルのパスを指定します。Bagging_num の個数分のモデルを読み込みます。
        trained_models = ["model/model_%s_%s.pth" % (CODER, b) for b in range(bagging_num)]

        # テストデータに対するDatasetを生成し、DataLoaderを通してData Queueを生成します。
        _, _, test_list = get_wav_list(words=label_to_int.keys())
        testdataset = SpeechDataset(mode='test', label_words_dict=label_to_int, wav_list=(test_list, []),
                                    add_noise=False, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,
                                    resize_shape=reshape_size, is_1d=is_1d)
        testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)

        for e, m in enumerate(trained_models):
            print("predicting ", m)
            speechmodel = get_model(m=MGPU)
            # torch.load()関数によって学習が完了したモデルを読み込みます。
            speechmodel.load_state_dict(torch.load(m))
            # モデルをcudaに連動し、evaluation モードに指定します。
            speechmodel = speechmodel.cuda()
            speechmodel.eval()

            test_fnames, test_labels = [], []
            pred_scores = []
            # テストデータをbatch_size 分受け取り、予測結果を生成します。
            for batch_idx, batch_data in enumerate(testloader):
                spec = Variable(batch_data['spec'].cuda())
                fname = batch_data['id']
                # y_predはデータに対するモデルの予測値です。
                y_pred = softmax(speechmodel(spec))
                pred_scores.append(y_pred.data.cpu().numpy())
                test_fnames += fname

            # bagging_num個のモデルが出力した確率値y_predを加え、アンサンブル予測値を求めます。
            if e == 0:
                final_pred = np.vstack(pred_scores)
                final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores)
                assert final_test_fnames == test_fnames

        # bagging_num 個数で割り、最終予測確率値(final_pred)をもとに最終予測値(final_labels)を生成します。
        final_pred /= len(trained_models)
        final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]

        # Kaggle提出用ファイル生成のためのファイル名(test_fnames)を定義します。
        test_fnames = [x.split("/")[-1] for x in final_test_fnames]
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']
        # Kaggle提出用ファイルを保存します。(ファイル名と最終予測値が記録されます)
        create_directory("sub")
        pd.DataFrame({'fname': test_fnames,
                      'label': final_labels}).to_csv("sub/%s.csv" % CODER, index=False)

        # 互いに異なるモデルのアンサンブル、学習性能向上を目的としてbaggingアンサンブルモデルの予測確率値を別のファイルに保存します。
        pred_scores = pd.DataFrame(np.vstack(final_pred), columns=labels)
        pred_scores['fname'] = test_fnames
        create_directory("pred_scores")
        pred_scores.to_csv("pred_scores/%s.csv" % CODER, index=False)


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
