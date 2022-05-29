import os
import pickle

import pandas as pd
import numpy as np

# xgboost, lightgbm ライブラリ
import xgboost as xgb
import lightgbm as lgbm

from utils import *

# XGBoost モデルを学習する関数です。
def xgboost(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 最適の parameterを指定します。
    param = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        # 'nthread': 16,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': len(products),
    }

    if not restore:
        # 訓練データからX, Y, weightを抽出します。as_matrixを通してメモリを効率的にarrayだけ保存します。
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=["y"])
        W_train = XY_train.as_matrix(columns=["weight"])
        # xgboost 専用データ形式に変換します。
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        # 検証データに対して同じ作業を進めます。
        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=["y"])
        W_validate = XY_validate.as_matrix(columns=["weight"])
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        # XGBoost モデルを学習します。early_stop の条件は 20番であり、最大1000個のツリーを学習します。
        evallist  = [(train,'train'), (validate,'eval')]
        model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)
        # 学習したモデルを保存します。
        pickle.dump(model, open("next_multi.pickle", "wb"))
    else:
        # “2016-06-28” テストデータを使用するときは、事前に学習したモデルを呼び出します。
        model = pickle.load(open("next_multi.pickle", "rb"))
    # 交差検証によって最適のツリーの個数を定めます。
    best_ntree_limit = model.best_ntree_limit

    if XY_all is not None:
        # 全体の訓練データに対してX, Y, weightを抽出し、XGBoost専用データ形式に変換します。
        X_all = XY_all.as_matrix(columns=features)
        Y_all = XY_all.as_matrix(columns=["y"])
        W_all = XY_all.as_matrix(columns=["weight"])
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)

        evallist  = [(all_data,'all_data')]
        # 学習するツリーの個数を全体の訓練データが増加した分だけ調整します。
        best_ntree_limit = int(best_ntree_limit * (len(XY_train) + len(XY_validate)) / len(XY_train))
        # モデル学習！
        model = xgb.train(param, all_data, best_ntree_limit, evals=evallist)

    # 変数の重要度を出力します。学習したXGBoostモデルで.get_fscore()を通して変数の重要度を確認できます。
    print("Feature importance:")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
        print(kv)

    # 予測に使うテストデータをXGBoost専用データに変換します。このとき、 weightはすべて1であるため、別途作業を行いません。
    X_test = test_df.as_matrix(columns=features)
    test = xgb.DMatrix(X_test, feature_names=features)

    # 学習したモデルをもとにして、best_ntree_limit個のツリーを基盤に予測します。
    return model.predict(test, ntree_limit=best_ntree_limit)


def lightgbm(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    # 訓練データ、検証データ、X, Y, weight 抽出の後、LightGBM専用データに変換します。
    train = lgbm.Dataset(XY_train[list(features)], label=XY_train["y"], weight=XY_train["weight"], feature_name=features)
    validate = lgbm.Dataset(XY_validate[list(features)], label=XY_validate["y"], weight=XY_validate["weight"], feature_name=features, reference=train)

    # 多様な実験を通して得られた最適の学習パラメータ
    params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'multiclass',
        'num_class': 24,
        'metric' : {'multi_logloss'},
        'is_training_metric': True,
        'max_bin': 255,
        'num_leaves' : 64,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 5,
        # 'num_threads': 16,
    }

    if not restore:
        # XGBoostと同じようにして訓練／検証データをもとに最適のツリーの個数を計算します。
        model = lgbm.train(params, train, num_boost_round=1000, valid_sets=validate, early_stopping_rounds=20)
        best_iteration = model.best_iteration
        # 学習したモデルと最適のツリーの個数を保存します。
        model.save_model("tmp/lgbm.model.txt")
        pickle.dump(best_iteration, open("tmp/lgbm.model.meta", "wb"))
    else:
        model = lgbm.Booster(model_file="tmp/lgbm.model.txt")
        best_iteration = pickle.load(open("tmp/lgbm.model.meta", "rb"))

    if XY_all is not None:
        # 全体の訓練データには増えた分だけツリーの個数を増やします。
        best_iteration = int(best_iteration * len(XY_all) / len(XY_train))
        # 全体の訓練データに対するLightGBM専用データを生成します。
        all_train = lgbm.Dataset(XY_all[list(features)], label=XY_all["y"], weight=XY_all["weight"], feature_name=features)
        # LightGBM モデル学習！
        model = lgbm.train(params, all_train, num_boost_round=best_iteration)
        model.save_model("tmp/lgbm.all.model.txt")

    # LightGBMモデルが提出する変数重要度機能を通して変数重要度を出力します。
    print("Feature importance by split:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("split"))], key=lambda kv: kv[1], reverse=True):
        print(kv)
    print("Feature importance by gain:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("gain"))], key=lambda kv: kv[1], reverse=True):
        print(kv)

    # テストデータに対する予測結果をリターンします。
    return model.predict(test_df[list(features)], num_iteration=best_iteration)
