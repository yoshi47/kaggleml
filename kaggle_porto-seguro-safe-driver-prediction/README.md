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

# Baselineモデルの実行
python gbm_model.py
```

# 勝者のコード

```
cd ../03_Winners_Code

# input /フォルダにコンテストのデータを入れる
cd code

# 勝者のコードを再現
python fea_eng0.py
python gbm_model291.py
python nn_model290.py
python simple_average.py
```
