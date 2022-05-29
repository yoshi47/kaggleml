# ニューラルネットワークモデル keras ライブラリを読み込みます。
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model

# 時間の測定および圧縮ファイルを読み込むためのライブラリ
from time import time
import datetime
from itertools import combinations
import pickle

# 特徴量エンジニアリングのためのライブラリ
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

#################
### READ DATA ###
#################

# 訓練データとテストデータを読み込みます。
train = pd.read_csv("../input/train.csv")
train_label = train['target']
train_id = train['id']
del train['target'], train['id']

test = pd.read_csv("../input/test.csv")
test_id = test['id']
del test['id']

######################
### UTIL FUNCTIONS ###
######################

def proj_num_on_cat(train_df, test_df, target_column, group_column):
    # train_df : 訓練データ
    # test_df : テストデータ
    # target_column : 統計を基盤とした派生変数を生成するターゲット変数
    # group_column : ピボット(pivot)変数
    train_df['row_id'] = range(train_df.shape[0])
    test_df['row_id'] = range(test_df.shape[0])
    train_df['train'] = 1
    test_df['train'] = 0

    # 訓練データとテストデータを統合します。
    all_df = train_df[['row_id', 'train', target_column, group_column]].append(test_df[['row_id','train', target_column, group_column]])
    
    # group_column を基盤としてピボットした target_columnの値を求めます。
    grouped = all_df[[target_column, group_column]].groupby(group_column)

    # 頻度(size), 平均(mean), 標準偏差(std), 中央値(median), 最大値(max), 最小値(min)を求めます。
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size' % target_column]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean' % target_column]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std' % target_column]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median' % target_column]
    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max' % target_column]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min' % target_column]

    # 統計を基盤とした派生変数を集めます。
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)
    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)
    all_df = pd.merge(all_df, the_stats, how='left')

    # 訓練データとテストデータに分離して返します。
    selected_train = all_df[all_df['train'] == 1]
    selected_test = all_df[all_df['train'] == 0]
    selected_train.sort_values('row_id', inplace=True)
    selected_test.sort_values('row_id', inplace=True)
    selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
    selected_train, selected_test = np.array(selected_train), np.array(selected_test)
    return selected_train, selected_test


def interaction_features(train, test, fea1, fea2, prefix):
    # train : 訓練データ
    # test : テストデータ
    # fea1, fea2 : 相互作用を遂行する変数名
    # prefix : 派生変数の変数名
    
    # 2つの変数間の掛け算／割り算の相互作用に対する派生変数を生成します。
    train['inter_{}*'.format(prefix)] = train[fea1] * train[fea2]
    train['inter_{}/'.format(prefix)] = train[fea1] / train[fea2]

    test['inter_{}*'.format(prefix)] = test[fea1] * test[fea2]
    test['inter_{}/'.format(prefix)] = test[fea1] / test[fea2]

    return train, test

###########################
### FEATURE ENGINEERING ###
###########################

# カテゴリ変数と2進変数の名前を抽出します。
cat_fea = [x for x in list(train) if 'cat' in x]
bin_fea = [x for x in list(train) if 'bin' in x]

# 欠損値 (-1)の個数で missing 派生変数を生成します。
train['missing'] = (train==-1).sum(axis=1).astype(float)
test['missing'] = (test==-1).sum(axis=1).astype(float)

# 6個の変数に対して相互作用変数を生成します。
for e, (x, y) in enumerate(combinations(['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01'], 2)):
    train, test = interaction_features(train, test, x, y, e)

# 数値型変数、相互作用派生変数、ind 変数の名前を抽出します。
num_features = [c for c in list(train) if ('cat' not in c and 'calc' not in c)]
num_features.append('missing')
inter_fea = [x for x in list(train) if 'inter' in x]
feature_names = list(train)
ind_features = [c for c in feature_names if 'ind' in c]

#  ind 変数の組み合わせを1つの文字列変数として表現します。
count = 0
for c in ind_features:
    if count == 0:
        train['new_ind'] = train[c].astype(str)
        count += 1
    else:
        train['new_ind'] += '_' + train[c].astype(str)
ind_features = [c for c in feature_names if 'ind' in c]
count = 0
for c in ind_features:
    if count == 0:
        test['new_ind'] = test[c].astype(str)
        count += 1
    else:
        test['new_ind'] += '_' + test[c].astype(str)

# reg 変数グループの組み合わせを1つの文字列変数として表現します。
reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        train['new_reg'] = train[c].astype(str)
        count += 1
    else:
        train['new_reg'] += '_' + train[c].astype(str)
reg_features = [c for c in feature_names if 'reg' in c]
count = 0
for c in reg_features:
    if count == 0:
        test['new_reg'] = test[c].astype(str)
        count += 1
    else:
        test['new_reg'] += '_' + test[c].astype(str)

# car 変数グループの組み合わせを1つの文字列変数として表現します。
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        train['new_car'] = train[c].astype(str)
        count += 1
    else:
        train['new_car'] += '_' + train[c].astype(str)
car_features = [c for c in feature_names if 'car' in c]
count = 0
for c in car_features:
    if count == 0:
        test['new_car'] = test[c].astype(str)
        count += 1
    else:
        test['new_car'] += '_' + test[c].astype(str)

# カテゴリ型データと数値型データを別に管理します。
train_cat = train[cat_fea]
train_num = train[[x for x in list(train) if x in num_features]]
test_cat = test[cat_fea]
test_num = test[[x for x in list(train) if x in num_features]]

# カテゴリ型データに LabelEncode()を遂行します。
max_cat_values = []
for c in cat_fea:
    le = LabelEncoder()
    x = le.fit_transform(pd.concat([train_cat, test_cat])[c])
    train_cat[c] = le.transform(train_cat[c])
    test_cat[c] = le.transform(test_cat[c])
    max_cat_values.append(np.max(x))

# カテゴリ変数の頻度値によって新しい派生変数を生成します。
cat_count_features = []
for c in cat_fea + ['new_ind','new_reg','new_car']:
    d = pd.concat([train[c],test[c]]).value_counts().to_dict()
    train['%s_count'%c] = train[c].apply(lambda x:d.get(x,0))
    test['%s_count'%c] = test[c].apply(lambda x:d.get(x,0))
    cat_count_features.append('%s_count'%c)

#  XGBoost 基盤変数を読み込みます。
train_fea0, test_fea0 = pickle.load(open("../input/fea0.pk", "rb"))

# 数値型変数の欠損値／異常値を0で代替し、カテゴリ変数とXGBoost基盤変数を統合します。
train_list = [train_num.replace([np.inf, -np.inf, np.nan], 0), train[cat_count_features], train_fea0]
test_list = [test_num.replace([np.inf, -np.inf, np.nan], 0), test[cat_count_features], test_fea0]

# ピボットベースの基本統計の派生変数を作成
for t in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01']:
    for g in ['ps_car_13', 'ps_ind_03', 'ps_reg_03', 'ps_ind_15', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat']:
        if t != g:
            #  group_column変数を基盤としてtarget_column値をピボットした後、基礎統計値を派生変数に追加します。
s_train, s_test = proj_num_on_cat(train, test, target_column=t, group_column=g)
            train_list.append(s_train)
            test_list.append(s_test)
            
# メモリ効率性のためにデータ全体を疎行列に変換します。
X = sparse.hstack(train_list).tocsr()
X_test = sparse.hstack(test_list).tocsr()
all_data = np.vstack([X.toarray(), X_test.toarray()])

# ニューラルネットワーク学習のためにすべての変数値を -1~1に Scalingします。
scaler = StandardScaler()
scaler.fit(all_data)
X = scaler.transform(X.toarray())
X_test = scaler.transform(X_test.toarray())

######################
### MODEL TRAINING ###
######################

# 2階層ニューラルネットワークモデルを定義します。
def nn_model():
    inputs = []
    flatten_layers = []

    # カテゴリ型データに対するEmbedding階層を定義します。すべてのカテゴリ変数は該当変数の最大値(num_c)の大きさのベクトルembeddingを学習します。
    for e, c in enumerate(cat_fea):
        input_c = Input(shape=(1, ), dtype='int32')
        num_c = max_cat_values[e]
        embed_c = Embedding(
            num_c,
            6,
            input_length=1
        )(input_c)
        embed_c = Dropout(0.25)(embed_c)
        flatten_c = Flatten()(embed_c)

        inputs.append(input_c)
        flatten_layers.append(flatten_c)

    # 数値型変数に対する入力階層を定義します。
    input_num = Input(shape=(X.shape[1],), dtype='float32')
    flatten_layers.append(input_num)
    inputs.append(input_num)

    # カテゴリ変数と数値型変数を統合して2階層Fully Connected Layerを定義します。
    flatten = merge(flatten_layers, mode='concat')

    # 1階層は512次元を持ち、PReLU Activation関数とBatchNormalization, Dropout関数を通過します。
    fc1 = Dense(512, init='he_normal')(flatten)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.75)(fc1)

    # 2階層は64次元を持ちます。
    fc1 = Dense(64, init='he_normal')(fc1)
    fc1 = PReLU()(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    outputs = Dense(1, init='he_normal', activation='sigmoid')(fc1)

    # モデル学習を遂行するoptimizerと学習基準となるloss関数を定義します。
    model = Model(input = inputs, output = outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return (model)

# 5分割交差検証を遂行します。
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

# モデル学習を5回のランダムシードで遂行した後、平均値を最終結果として得ます。
num_seeds = 5
begintime = time()

# 内部交差検証およびテストデータに対する予測値を保存するための準備をします。
cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))

X_cat = train_cat.as_matrix()
X_test_cat = test_cat.as_matrix()

x_test_cat = []
for i in range(X_test_cat.shape[1]):
    x_test_cat.append(X_test_cat[:, i].reshape(-1, 1))
x_test_cat.append(X_test)

# ランダムシードの個数だけモデル学習を遂行します。
for s in range(num_seeds):
    np.random.seed(s)
    for (inTr, inTe) in kfold.split(X, train_label):
        xtr = X[inTr]
        ytr = train_label[inTr]
        xte = X[inTe]
        yte = train_label[inTe]

        xtr_cat = X_cat[inTr]
        xte_cat = X_cat[inTe]

        # カテゴリ型データを抽出し、数値型データと統合します。
        xtr_cat_list, xte_cat_list = [], []
        for i in range(xtr_cat.shape[1]):
            xtr_cat_list.append(xtr_cat[:, i].reshape(-1, 1))
            xte_cat_list.append(xte_cat[:, i].reshape(-1, 1))
        xtr_cat_list.append(xtr)
        xte_cat_list.append(xte)

        # ニューラルネットワークモデルを定義します。
        model = nn_model()
        # モデルを学習します。
        model.fit(xtr_cat_list, ytr, epochs=20, batch_size=512, verbose=2, validation_data=[xte_cat_list, yte])
        
        # 予測値の順位を求める関数get_rank()を定義します。Gini評価関数は予測値間の順位を基準として評価するため、最終評価点数に影響を及ぼしません。
        def get_rank(x):
            return pd.Series(x).rank(pct=True).values
        
        # 内部交差検証データに対する予測値を保存します。
        cv_train[inTe] += get_rank(model.predict(x=xte_cat_list, batch_size=512, verbose=0)[:, 0])
        print(Gini(train_label[inTe], cv_train[inTe]))
        
        # テストデータに対する予測値を保存します。
        cv_pred += get_rank(model.predict(x=x_test_cat, batch_size=512, verbose=0)[:, 0])

    print(Gini(train_label, cv_train / (1. * (s + 1))))
    print(str(datetime.timedelta(seconds=time() - begintime)))
    
# テストデータに対する最終予測値をファイルに保存します。
pd.DataFrame({'id': test_id, 'target': get_rank(cv_pred * 1./ (NFOLDS * num_seeds))}).to_csv('../model/keras5_pred.csv', index=False)
