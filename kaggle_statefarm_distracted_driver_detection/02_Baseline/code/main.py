# -*- coding: utf-8 -*-

# ディープラーニング関連のKerasライブラリ
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator

# File I/O
import subprocess
import shutil
import os
from glob import glob
from datetime import datetime
import argparse

# Data processing
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold

# Image processing
import cv2
from scipy.ndimage import rotate
import scipy.misc

# 学習パラメータのセット
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='None')
parser.add_argument('--learning-rate', required=False, type=float, default=1e-4)
parser.add_argument('--semi-train', required=False, default=None)
parser.add_argument('--batch-size', required=False, type=int, default=8)
parser.add_argument('--random-split', required=False, type=int, default=1)
parser.add_argument('--data-augment', required=False, type=int, default=0)
args = parser.parse_args()

fc_size = 2048
n_class = 10
seed = 10
nfolds = 5
test_nfolds = 3
img_row_size, img_col_size = 224, 224
train_path = '../input/train'
if args.semi_train is not None:
    train_path = args.semi_train
    args.semi_train = True
test_path = '../input/test'
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

suffix = 'm{}.w{}.lr{}.s{}.nf{}.semi{}.b{}.row{}col{}.rsplit{}.augment{}.d{}'.format(args.model, args.weights, args.learning_rate, seed, nfolds, args.semi_train, args.batch_size, img_row_size, img_col_size, args.random_split, args.data_augment, datetime.now().strftime("%Y-%m-%d-%H-%M"))
temp_train_fold = '../input/train_{}'.format(suffix)
temp_valid_fold = '../input/valid_{}'.format(suffix)
cache = '../cache/{}'.format(suffix)
subm = '../subm/{}'.format(suffix)

def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
for path in [temp_train_fold, temp_valid_fold, cache, subm]:
    _clear_dir(path)

def get_model():
    # 最上位の以外のモデルをロード
    if args.weights == 'None':
        args.weights = None
    if args.model in ['vgg16']:
        base_model = keras.applications.vgg16.VGG16(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['vgg19']:
        base_model = keras.applications.vgg19.VGG19(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['resnet50']:
        base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    else:
        print('# {} is not a valid value for "--model"'.format(args.model))
        exit()

    # 最上位全結合層を定義します。
    out = Flatten()(base_model.output)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    output = Dense(n_class, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=output)

    # SGD Optimizerを使用して、モデルをcompileします。
    sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

print('# Data Load')
drivers = pd.read_csv('../input/driver_imgs_list.csv')
img_to_driver = {}
uniq_drivers = []

for i, row in drivers.iterrows():
    label_n_driver = {}
    label_n_driver['label'] = row['classname']
    label_n_driver['driver'] = row['subject']
    img_to_driver[row['img']] = label_n_driver

    if row['subject'] not in uniq_drivers:
        uniq_drivers.append(row['subject'])

def generate_driver_based_split(img_to_driver, train_drivers):
    # イメージジェネレーターのために一時的に訓練／検証フォルダーを生成します。
    def _generate_temp_folder(root_path):
        _clear_dir(root_path)
        for i in range(n_class):
            os.mkdir('{}/c{}'.format(root_path, i))
    _generate_temp_folder(temp_train_fold)
    _generate_temp_folder(temp_valid_fold)

    # データを一時的に訓練／検証フォルダーへランダムにコピーします。
    train_samples = 0
    valid_samples = 0
    if not args.random_split:
        for img_path in img_to_driver.keys():
            cmd = 'cp {}/{}/{} {}/{}/{}'
            label = img_to_driver[img_path]['label']
            if not os.path.exists('{}/{}/{}'.format(train_path, label, img_path)):
                continue
            if img_to_driver[img_path]['driver'] in train_drivers:
                cmd = cmd.format(train_path, label, img_path, temp_train_fold, label, img_path)
                train_samples += 1
            else:
                cmd = cmd.format(train_path, label, img_path, temp_valid_fold, label, img_path)
                valid_samples += 1
            # イメージの複製
            subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)
    else:
        for label in labels:
            files = glob('{}/{}/*jpg'.format(train_path, label))
            for fl in files:
                cmd = 'cp {} {}/{}/{}'
                if np.random.randint(nfolds) != 1:
                    # 訓練データに4/5のデータを追加する
                    cmd = cmd.format(fl, temp_train_fold, label, os.path.basename(fl))
                    train_samples += 1
                else:
                    # 検証データに1/5のデータを追加する
                    cmd = cmd.format(fl, temp_valid_fold, label, os.path.basename(fl))
                    valid_samples += 1
                # 原本の訓練データを一時的に訓練／検証データにコピーします。
                subprocess.call(cmd, stderr=subprocess.STDOUT, shell=True)

    # 訓練／検証データの個数を出力します。
    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

def crop_center(img, cropx, cropy):
    # イメージの中間をCropする関数を定義します。
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocess(image):
    # イメージデータを0~1の値で調整します。
    image /= 255.

    # イメージを最大-20~20度回転させます。
    rotate_angle = np.random.randint(40) - 20
    image = rotate(image, rotate_angle)

    # イメージを最大 -30~30pixel 移動します。
    rows, cols, _ = image.shape
    width_translate = np.random.randint(60) - 30
    height_translate = np.random.randint(60) - 30
    M = np.float32([[1,0,width_translate],[0,1,height_translate]])
    image = cv2.warpAffine(image,M,(cols,rows))    

    # イメージを最大 0.8~1.0 ズームインします。
    width_zoom = int(img_row_size * (0.8 + 0.2 * (1 - np.random.random())))
    height_zoom = int(img_col_size * (0.8 + 0.2 * (1 - np.random.random())))
    final_image = np.zeros((height_zoom, width_zoom, 3))
    final_image[:,:,0] = crop_center(image[:,:,0], width_zoom, height_zoom)
    final_image[:,:,1] = crop_center(image[:,:,1], width_zoom, height_zoom)
    final_image[:,:,2] = crop_center(image[:,:,2], width_zoom, height_zoom)

    # (224, 224)の大きさでイメージを再調整します。
    image = cv2.resize(final_image, (img_row_size, img_col_size))
    return image

print('# Train Model')
# (224, 224)の大きさでイメージを再調整します。
# リアルタイム前処理を追加する場合、前処理関数を設定値に入れます。
if args.data_augment:
    datagen = ImageDataGenerator(preprocessing_function=preprocess)
else:
    datagen = ImageDataGenerator()

# テストデータを呼び出すImageGeneratorを生成します。
test_generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_row_size, img_col_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)
test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(test_path))]

# 5分割交差検証を進めます。
kf = KFold(len(uniq_drivers), n_folds=nfolds, shuffle=True, random_state=20)
for fold, (train_drivers, valid_drivers) in enumerate(kf):
    # 新しいモデルを定義します。
    model = get_model()

    # 訓練／検証データを生成します。
    train_drivers = [uniq_drivers[j] for j in train_drivers]
    train_samples, valid_samples = generate_driver_based_split(img_to_driver, train_drivers)

    # 訓練／検証データジェネレーターを定義します。
    train_generator = datagen.flow_from_directory(
            directory=temp_train_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            seed=seed)
    valid_generator = datagen.flow_from_directory(
            directory=temp_valid_fold,
            target_size=(img_row_size, img_col_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            seed=seed)

    weight_path = '../cache/{}/weight.fold_{}.h5'.format(suffix, fold)
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, verbose=0)]
    # モデルを学習します。val_loss値が3 epoch連続悪化した場合、学習を停止し最適weightを保存します。
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_samples/args.batch_size,
            epochs=500,
            validation_data=valid_generator,
            validation_steps=valid_samples/args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1)

    # データに対してリアルタイムの前処理を行い、予測結果のn回分の平均値を最終的な予測値とします。
    for j in range(test_nfolds):
        preds = model.predict_generator(
                test_generator,
                steps=len(test_id),
                verbose=1)

        if j == 0:
            result = pd.DataFrame(preds, columns=labels)
        else:
            result += pd.DataFrame(preds, columns=labels)
    result /= test_nfolds
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    sub_file = '../subm/{}/f{}.csv'.format(suffix, fold)
    result.to_csv(sub_file, index=False)

    # Kaggleに提出します。
    submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}.fold{}'.format(sub_file, suffix, fold)
    subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)

    # 5分割交差検証の処理で作成した訓練/検証データを削除します。
    shutil.rmtree(temp_train_fold)
    shutil.rmtree(temp_valid_fold)

print('# Ensemble')
# 5分割交差検証の結果を単純アンサンブルします。
ensemble = 0
for fold in range(nfolds):
    ensemble += pd.read_csv('../subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
ensemble = pd.DataFrame(ensemble, columns=labels)
ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
sub_file = '../subm/{}/ens.csv'.format(suffix)
ensemble.to_csv(sub_file, index=False)

# Kaggleに提出します。
submit_cmd = 'kaggle competitions submit -c state-farm-distracted-driver-detection -f {} -m {}'.format(sub_file, suffix)
subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)
