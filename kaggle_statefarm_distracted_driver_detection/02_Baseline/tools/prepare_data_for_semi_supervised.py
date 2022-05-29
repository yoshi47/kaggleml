import pandas as pd
import numpy as np
import os

# 表5-10のアンサンブルの結果を得たcsvファイルを保存します。(ファイル名は読者によって異なるでしょう)
test_pred_fname = 'FIX_ME'
test_pred = pd.read_csv(test_pred_fname)
test_pred_probs = test_pred.iloc[:, :-1]
test_pred_probs_max = np.max(test_pred_probs.values, axis=1)

# 確率値の区間別にいくつのファイルが存在するのかを出力します。
for thr in range(1,10):
  thr = thr / 10.
  count = sum(test_pred_probs_max > thr)
  print('# Thre : {} | count : {} ({}%)'.format(thr, count, 1. * count / len(test_pred_probs_max)))

# 確率値の基準値を0.90に指定します。
print('=' * 50)
threshold = 0.90
count = {}
print('# Extracting data with threshold : {}'.format(threshold))

# 既存の訓練データをsemi_train_{}ディレクトリにコピーします。
cmd = 'cp -r input/train input/semi_train_{}'.format(os.path.basename(test_pred_fname))
os.system(cmd)

# 確率値0.9以上のテストデータをsemi_train_{}ディレクトリにコピーします。
for i, row in test_pred.iterrows():
  img = row['img']
  row = row.iloc[:-1]
  if np.max(row) > threshold:
    label = row.values.argmax()
    cmd = 'cp input/test/imgs/{} input/semi_train_{}/c{}/{}'.format(img, os.path.basename(test_pred_fname), label, img)
    os.system(cmd)
    count[label] = count.get(label, 0) + 1

# クラス別に追加されたテストデータの統計を出力します。
print('# Added semi-supservised labels: \n{}'.format(count))
