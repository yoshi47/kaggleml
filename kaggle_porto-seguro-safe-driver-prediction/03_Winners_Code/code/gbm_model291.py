# モデル学習に必要なライブラリ
import lightgbm as lgbm
from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def Gini(y_true, y_pred):
    # 正答と予測値の個数が同一であるかどうかを確認します。
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # 予測値(y_pred)を昇順に整理します。
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # Lorenz curvesを計算します。
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # Gini 係数を計算します。
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    #  Gini 係数を正規化します。
    return G_pred * 1. / G_true

# LightGBM モデル学習の過程で評価関数として使用します。
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

#################
### READ DATA ###
#################

# 訓練データ、テストデータを読み込みます。
path = "../input/"
train = pd.read_csv(path+'train.csv')
train_label = train['target']
train_id = train['id']
test = pd.read_csv(path+'test.csv')
test_id = test['id']

# target 変数を別途に分離し、‘id, target’ 変数を除去します。訓練データとテストデータの変数を同じように持っていくためです。
y = train['target'].values
drop_feature = [
    'id',
    'target'
]
X = train.drop(drop_feature,axis=1)

###########################
### FEATURE ENGINEERING ###
###########################

# カテゴリ変数と数値型変数を分離します。
feature_names = X.columns.tolist()
cat_features = [c for c in feature_names if ('cat' in c and 'count' not in c)]
num_features = [c for c in feature_names if ('cat' not in c and 'calc' not in c)]

# 派生変数 01 : 欠損値を意味する “-1”の個数を数えます。
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)
num_features.append('missing')

# 派生変数 02 : カテゴリ変数を LabelEncoder()を通して数値型に変換した後、OneHotEncoder()を通して固有値別に 0/1の2進変数をデータとして使用します。
for c in cat_features:
    le = LabelEncoder()
    le.fit(train[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])
    
enc = OneHotEncoder()
enc.fit(train[cat_features])
X_cat = enc.transform(train[cat_features])
X_t_cat = enc.transform(test[cat_features])

# 派生変数 03 : ‘ind’ 変数の固有値を組み合わせた ‘new_ind’ 変数を生成します。
# 例: ps_ind_01 = 1, ps_ind_02 = 0の値を持っている場合, new_indは ‘1_2_’という文字列変数になります。ind 変数の組み合わせを基盤として派生変数を生成するわけです。
ind_features = [c for c in feature_names if 'ind' in c]
count=0
for c in ind_features:
    if count==0:
        train['new_ind'] = train[c].astype(str)+'_'
        test['new_ind'] = test[c].astype(str)+'_'
        count+=1
    else:
        train['new_ind'] += train[c].astype(str)+'_'
        test['new_ind'] += test[c].astype(str)+'_'

# 派生変数 03 continue : カテゴリ変数と ‘new_ind’ 固有値の頻度を派生変数として生成します。
cat_count_features = []
for c in cat_features+['new_ind']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)
    
# 数値型変数、カテゴリ変数/new_ind 頻度およびカテゴリ変数のモデル学習に使用します。それ以外の変数は学習に使用しません。
train_list = [train[num_features+cat_count_features].values,X_cat,]
test_list = [test[num_features+cat_count_features].values,X_t_cat,]

# モデル学習速度およびメモリ最適化のためデータを Sparse Matrix 形態に変換します。
X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()

######################
### MODEL TRAINING ###
######################

# LightGBM モデルの設定値です。
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": 0.1,
          "num_leaves": 15,
           "max_bin": 256,
          "feature_fraction": 0.6,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
          }

# 層化5分割内部交差検証
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

x_score = []
final_cv_train = np.zeros(len(train_label))
final_cv_pred = np.zeros(len(test_id))
# 総計16の異なるランダムシードで学習を回し、平均値を最終予測結果として使用します。シード値が多いほどランダム要素による分断を減らすことができます。
for s in xrange(16):
    cv_train = np.zeros(len(train_label))
    cv_pred = np.zeros(len(test_id))

    params['seed'] = s
    
    kf = kfold.split(X, train_label)

    best_trees = []
    fold_scores = []

    for i, (train_fold, validate) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = X[train_fold, :], X[validate, :], train_label[train_fold], train_label[validate]
        dtrain = lgbm.Dataset(X_train, label_train)
        dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
        # 訓練データを学習し、evalerror() 関数を通して検証データに対する正規化Gini係数の点数を基準として最適のツリー個数を探します。
        bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
        best_trees.append(bst.best_iteration)
        # テストデータに対する予測値をcv_predに加えます。
        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
        cv_train[validate] += bst.predict(X_validate)

        # 検証データに対する評価点数を出力します。
        score = Gini(label_validate, cv_train[validate])
        print(score)
        fold_scores.append(score)

    cv_pred /= NFOLDS
    final_cv_train += cv_train
    final_cv_pred += cv_pred

    # シード値別に交差検証の点数を出力します。
    print("cv score:")
    print(Gini(train_label, cv_train))
    print("current score:", Gini(train_label, final_cv_train / (s + 1.)), s+1)
    print(fold_scores)
    print(best_trees, np.mean(best_trees))

    x_score.append(Gini(train_label, cv_train))

print(x_score)
# テストデータに対する結果をシード値の個数で割り、0~1の値に修正して、結果を保存します。
pd.DataFrame({'id': test_id, 'target': final_cv_pred / 16.}).to_csv('../model/lgbm3_pred_avg.csv', index=False)
