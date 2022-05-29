import pandas as pd
import numpy as np
import xgboost as xgb

np.random.seed(2018)

# データを呼び出します。
trn = pd.read_csv('../input/train_ver2.csv')
tst = pd.read_csv('../input/test_ver2.csv')

## データの前処理 ##

# 製品の変数を別途に保存しておきます。
prods = trn.columns[24:].tolist()

# 製品変数の欠損値をあらかじめ0に代替しておきます。
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 24個の製品を1つも保有していない顧客のデータを除去します。
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

# 訓練データとテストデータを統合します。テストデータにない製品変数は0で埋めます。
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)

# 学習に使用する変数を入れるlistです。
features = []

# カテゴリ変数を .factorize() 関数に通して label encodingします。
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)
features += categorical_cols

# 数値型変数の特異値と欠損値を -99に代替し、整数型に変換します。
df['age'].replace(' NA', -99, inplace=True)
df['age'] = df['age'].astype(np.int8)

df['antiguedad'].replace('     NA', -99, inplace=True)
df['antiguedad'] = df['antiguedad'].astype(np.int8)

df['renta'].replace('         NA', -99, inplace=True)
df['renta'].fillna(-99, inplace=True)
df['renta'] = df['renta'].astype(float).astype(np.int8)

df['indrel_1mes'].replace('P', 5, inplace=True)
df['indrel_1mes'].fillna(-99, inplace=True)
df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

# 学習に使用する数値型変数を featuresに追加します。
features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']

# (特徴量エンジニアリング) 2つの日付変数から年度と月の情報を抽出します。
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month', 'fecha_alta_year']

df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']

# それ以外の変数の欠損値をすべて -99に代替します。
df.fillna(-99, inplace=True)

# (特徴量エンジニアリング) lag-1 データを生成します。
# コード 2-12と類似したコードの流れです

# 日付を数字に変換する関数です。 2015-01-28は 1, 2016-06-28は 18に変換します。
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 日付を数字に変換し int_dateに保存します。
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# データをコピーし, int_date 日付に1を加え lagを生成します。変数名に _prevを追加します。
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
df_lag['int_date'] += 1

# 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

# メモリの効率化のために、不必要な変数をメモリから除去します。
del df, df_lag

# 前の月の製品情報が存在しない場合に備えて、0に代替します。
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

# lag-1 変数を追加します。
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

###
### Baseline モデル以後、多様な特徴量エンジニアリングをここに追加します。
###


## モデル学習
# 学習のため、データを訓練、検証用に分離します。
# 学習には 2016-01-28 ~ 2016-04-28 のデータだけを使用し、検証には 2016-05-28 のデータを使用します。
use_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']
del df_trn

# 訓練データから新規購買件数だけを抽出します。
X = []
Y = []
for i, prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY['y'] = Y

# 訓練、検証データに分離します。
vld_date = '2016-05-28'
XY_trn = XY[XY['fecha_dato'] != vld_date]
XY_vld = XY[XY['fecha_dato'] == vld_date]


# XGBoost モデルの parameterを設定します。
param = {
    'booster': 'gbtree',
    'max_depth': 8,
    'nthread': 4,
    'num_class': len(prods),
    'objective': 'multi:softprob',
    'silent': 1,
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'min_child_weight': 10,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.9,
    'seed': 2018,
    }

# 訓練、検証データを XGBoost 形態に変換します。
X_trn = XY_trn.as_matrix(columns=features)
Y_trn = XY_trn.as_matrix(columns=['y'])
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

X_vld = XY_vld.as_matrix(columns=features)
Y_vld = XY_vld.as_matrix(columns=['y'])
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

# XGBoost モデルを訓練データで学習させます！
watch_list = [(dtrn, 'train'), (dvld, 'eval')]
model = xgb.train(param, dtrn, num_boost_round=1000, evals=watch_list, early_stopping_rounds=20)

# 学習したモデルを保存します。
import pickle
pickle.dump(model, open("../model/xgb.baseline.pkl", "wb"))
best_ntree_limit = model.best_ntree_limit

# MAP@7 評価基準のための準備作業です。
# 顧客識別番号を抽出します。
vld = trn[trn['fecha_dato'] == vld_date]
ncodpers_vld = vld.as_matrix(columns=['ncodpers'])
# 検証データから新規購買を求めます。
for prod in prods:
    prev = prod + '_prev'
    padd = prod + '_add'
    vld[padd] = vld[prod] - vld[prev]    
add_vld = vld.as_matrix(columns=[prod + '_add' for prod in prods])
add_vld_list = [list() for i in range(len(ncodpers_vld))]

# 顧客別新規購買正答値を add_vld_listに保存し、総 countを count_vldに保存します。
count_vld = 0
for ncodper in range(len(ncodpers_vld)):
    for prod in range(len(prods)):
        if add_vld[ncodper, prod] > 0:
            add_vld_list[ncodper].append(prod)
            count_vld += 1

# 検証データから得ることのできる MAP@7 の最高点をあらかじめ求めておきます。(0.042663)
print(mapk(add_vld_list, add_vld_list, 7, 0.0))

# 検証データに対する予測値を求めます。
X_vld = vld.as_matrix(columns=features)
Y_vld = vld.as_matrix(columns=['y'])
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
preds_vld = model.predict(dvld, ntree_limit=best_ntree_limit)

# 前の月に保有していた商品は新規購買が不可能なので、確率値からあらかじめ1を引いておきます。
preds_vld = preds_vld - vld.as_matrix(columns=[prod + '_prev' for prod in prods])

# 検証データの予測上位7個を抽出します。
result_vld = []
for ncodper, pred in zip(ncodpers_vld, preds_vld):
    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
    result_vld.append([ip for y,p,ip in y_prods])
    
# 検証データの MAP@7の点数を求めます。(0.036466)
print(mapk(add_vld_list, result_vld, 7, 0.0))

# XGBoost モデルを全体の訓練データで学習します。
X_all = XY.as_matrix(columns=features)
Y_all = XY.as_matrix(columns=['y'])
dall = xgb.DMatrix(X_all, label=Y_all, feature_names=features)
watch_list = [(dall, 'train')]
# ツリーの個数を増加したデータの量に比例して増やします。
best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(XY_vld)) / len(XY_trn))
# XGBoost モデル再学習！
model = xgb.train(param, dall, num_boost_round=best_ntree_limit, evals=watch_list)

# 変数の重要度を出力してみます。予想していた変数が上位に来ていますか？
print("Feature importance:")
for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
    print(kv)

# Kaggleに提出するため、テストデータに対する予測値を求めます。
X_tst = tst.as_matrix(columns=features)
dtst = xgb.DMatrix(X_tst, feature_names=features)
preds_tst = model.predict(dtst, ntree_limit=best_ntree_limit)
ncodpers_tst = tst.as_matrix(columns=['ncodpers'])
preds_tst = preds_tst - tst.as_matrix(columns=[prod + '_prev' for prod in prods])

# 提出ファイルを生成します。
submit_file = open('../model/xgb.baseline.2015-06-28', 'w')
submit_file.write('ncodpers,added_products\n')
for ncodper, pred in zip(ncodpers_tst, preds_tst):
    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
    y_prods = [p for y,p,ip in y_prods]
    submit_file.write('{},{}\n'.format(int(ncodper), ' '.join(y_prods)))

