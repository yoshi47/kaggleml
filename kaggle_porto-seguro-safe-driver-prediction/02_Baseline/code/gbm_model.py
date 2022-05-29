import pandas as pd

#  訓練／テストデータを読み込みます。
train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

# 派生変数 01 : 欠損値を意味する “-1”の個数を数えます
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 派生変数 02 : 2進変数の合計。
bin_features = [c for c in train.columns if 'bin' in c]
train['bin_sum'] = train[bin_features].sum(axis=1)
test['bin_sum'] = test[bin_features].sum(axis=1)

# 派生変数 03 : 単一変数ターゲット比率分析によって選ばれた変数をもとに Target Encodingを遂行します。Target Encodingは交差検証の過程で進めます。
features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']

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
          "subsample": 0.9,
          "seed": 2018
}

# Stratified 5分割内部交差検証を準備します。
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
kf = kfold.split(train, train_label)

cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))    
best_trees = []
fold_scores = []

for i, (train_fold, validate) in enumerate(kf):
    # 訓練／検証データを分離します。
    X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate, :], train_label[train_fold], train_label[validate]
    
    #  target encoding 特徴量エンジニアリングを遂行します。
    for feature in features:
        # 訓練データで feature 固有値別ターゲット変数の平均を求めます。
        map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')
        map_dic = map_dic.to_dict()['target']
        # 訓練／検証／テストデータに平均値をマッピングします。
        X_train[feature + ‘_target_enc’] = X_train[feature].apply(lambda x: map_dic.get(x, 0))
        X_validate[feature + ‘_target_enc’] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))
        test[feature + ‘_target_enc’] = test[feature].apply(lambda x: map_dic.get(x, 0))

    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
    # 訓練データを学習し、evalerror()関数によって検証データに対する正規化ジニ係数の点数を基準として最適なツリーの個数を求めます。
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
    best_trees.append(bst.best_iteration)
    # テストデータに対する予測値を cv_predに加えます。
    cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
    cv_train[validate] += bst.predict(X_validate)

    # 検証データに対する評価点数を出力します。
    score = Gini(label_validate, cv_train[validate])
    print(score)
    fold_scores.append(score)

cv_pred /= NFOLDS

# シード値別に交差検証の点数を出力します。
print("cv score:")
print(Gini(train_label, cv_train))
print(fold_scores)
print(best_trees, np.mean(best_trees))

# テストデータに対する結果を保存します。
pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('../model/lgbm_baseline.csv', index=False)
