# データの探索的分析

```
# (オプション) 仮想環境のインストール
pip install virtualenv
# 仮想環境の構築
virtualenv venv
# 仮想環境をアクティベート
. venv/bin/activate

# 必要なライブラリをインストール
pip install -r requirements.txt

# jupyter notebook をバックグラウンドで動かす
nohup jupyter notebook &

# Webブラウザで http://localhost:8000 に接続した後, 
# 01_EDA/EDA.ipynb ファイルを実行
```

# Baseline モデル

```
cd 02_Baseline

# input /フォルダにコンテストのデータを入れる
cd code

## 01. Baseline モデル
python main.py

# 改善実験の再現
## 02. ドライバー別交差検証
python main.py --weights None --random-split 0 --data-augment 0 --learning-rate 1e-4

## 03. ImageNet事前学習モデル
python main.py --weights imagenet --random-split 0 --data-augment 0 --learning-rate 1e-5

## 04. リアルタイム データ拡張
python main.py --weights imagenet --random-split 0 --data-augment 1 --learning-rate 1e-4

## 05. ランダム交差検証
python main.py --weights imagenet --random-split 1 --data-augment 1 --learning-rate 1e-4

## 06. 様々なCNNモデル学習 (ResNet50)
python main.py --weights imagenet --random-split 0 --data-augment 1 --learning-rate 1e-4 --model resnet50

## 07. アンサンブル
# アンサンブルを実行するファイルをrsc / ensembleフォルダに移動する
cp ../subm/<ResNet50モデル結果ファイルのパス>/ens.csv ../rsc/ensemble/resnet50.csv
cp ../subm/<VGG19モデル結果ファイルのパス>/ens.csv ../rsc/ensemble/vgg19.csv
cp ../subm/<VGG16モデル結果ファイルのパス>/ens.csv ../rsc/ensemble/vgg16.csv
python ../tools/ensemble.py

## 08. Semi-Supervised Learning
# Semi-Supervised Learning用の訓練データを構築
python ../tools/prepare_data_for_semi_supervised.py
python main.py --weights imagenet --random-split 1 --data-augment 1 --learning-rate 1e-4 --semi-train ../input/<semi-supervised 学習データパス> --model resnet50
```
