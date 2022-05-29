import pandas as pd
# 各モデルの結果を読み込みます。
keras5_test = pd.read_csv("../model/keras5_pred.csv")
lgbm3_test = pd.read_csv("../model/lgbm3_pred_avg.csv")

def get_rank(x):
    return pd.Series(x).rank(pct=True).values

# 2つの予測値の単純平均を最終アンサンブルの結果として保存します。
pd.DataFrame({'id': keras5_test['id'], 'target': get_rank(keras5_test['target']) * 0.5 + get_rank(keras5_test['target']) * 0.5}).to_csv("../model/simple_average.csv", index = False)
