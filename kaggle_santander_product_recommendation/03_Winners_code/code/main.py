import math    
import io    

# ファイル圧縮用途
import gzip    
import pickle    
import zlib    

# データ、配列を扱うための基本ライブラリ
import pandas as pd 
import numpy as np

# カテゴリ型データを数値型データに変換する前処理ツール
from sklearn.preprocessing import LabelEncoder

import engines
from utils import *

np.random.seed(2016)
transformers = {}

def assert_uniq(series, name):
    uniq = np.unique(series, return_counts=True)
    print("assert_uniq", name, uniq)

def custom_one_hot(df, features, name, names, dtype=np.int8, check=False):
    for n, val in names.items():
        # 新規変数を “変数名_数字”として指定します。
        new_name = "%s_%s" % (name, n)
        # 既存の変数で該当固有値を持っていれば 1, それ以外は 0の2進変数を生成します。
        df[new_name] = df[name].map(lambda x: 1 if x == val else 0).astype(dtype)
        features.append(new_name)

def label_encode(df, features, name):
    # データフレームdfの変数 nameの値をすべて stringに変換します。
    df[name] = df[name].astype('str')
    # 既に,label_encode した変数である場合, transformer[name]にある LabelEncoder()を再活用します。
    if name in transformers:
        df[name] = transformers[name].transform(df[name])
    # 初めて見る変数である場合,transformerにLabelEncoder()を保存し, .fit_transform() 関数で label encodingを遂行します。
    else: # train
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
    # label encodingした変数は features リストに追加します。
    features.append(name)

def encode_top(s, count=100, dtype=np.int8):
    # すべての固有値に対する頻度を計算します。
    uniqs, freqs = np.unique(s, return_counts=True)
    # 頻度 Top 100を抽出します。
    top = sorted(zip(uniqs,freqs), key=lambda vk: vk[1], reverse = True)[:count]
    # { 既存データ：順位 }をあらわす dict()を生成します。
    top_map = {uf[0]: l+1 for uf, l in zip(top, range(len(top)))}
    # 高頻度 100個のデータは順位に代替し、それ以外は 0 に代替します。
    return s.map(lambda x: top_map.get(x, 0)).astype(dtype)

def apply_transforms(train_df):
    # 学習に使用する変数を保存する features リストを生成します。
    features = []

    # 2つの変数を label_encode() します。
    label_encode(train_df, features, "canal_entrada")
    label_encode(train_df, features, "pais_residencia")

    # ageの欠損値を 0.0に代替し、すべての値を整数に変換します。
    train_df["age"] = train_df["age"].fillna(0.0).astype(np.int16)
    features.append("age")

    # rentaの欠損値を 1.0に代替し、 logをかけて分布を変形します。
    train_df["renta"].fillna(1.0, inplace=True)
    train_df["renta"] = train_df["renta"].map(math.log)
    features.append("renta")

    # 高頻度の 100個の順位を抽出します。
    train_df["renta_top"] = encode_top(train_df["renta"])
    features.append("renta_top")

    # 欠損値あるいは負の数である場合は 0に代替し、残りの値は +1.0 をした後、整数に変換します。
    train_df["antiguedad"] = train_df["antiguedad"].map(lambda x: 0.0 if x < 0 or math.isnan(x) else x+1.0).astype(np.int16)
    features.append("antiguedad")

    # 欠損値を 0.0に代替し、整数に変換します。
    train_df["tipodom"] = train_df["tipodom"].fillna(0.0).astype(np.int8)
    features.append("tipodom")
    train_df["cod_prov"] = train_df["cod_prov"].fillna(0.0).astype(np.int8)
    features.append("cod_prov")

    # fecha_datoから月／年度を抽出し、整数値に変換します。
    train_df["fecha_dato_month"] = train_df["fecha_dato"].map(lambda x: int(x.split("-")[1])).astype(np.int8)
    features.append("fecha_dato_month")
    train_df["fecha_dato_year"] = train_df["fecha_dato"].map(lambda x: float(x.split("-")[0])).astype(np.int16)
    features.append("fecha_dato_year")

    # 欠損値を 0.0に代替し、fecha_altaから月／年度を抽出して整数値に変換します。
    # x.__class__が欠損値の場合 floatを変換するため、欠損値探知用に使用します。
    train_df["fecha_alta_month"] = train_df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[1])).astype(np.int8)
    features.append("fecha_alta_month")
    train_df["fecha_alta_year"] = train_df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[0])).astype(np.int16)
    features.append("fecha_alta_year")

    # 日付データを月を基準とした数値型変数に変換します。
    train_df["fecha_dato_float"] = train_df["fecha_dato"].map(date_to_float)
    train_df["fecha_alta_float"] = train_df["fecha_alta"].map(date_to_float)

    # fecha_dato と fecha_altoの月を基準とした数値型変数の差異値を派生変数として生成します。
    train_df["dato_minus_alta"] = train_df["fecha_dato_float"] - train_df["fecha_alta_float"]
    features.append("dato_minus_alta")

    # 日付データを月を基準とした数値型変数に変換します (1 ~ 18 間の値に制限)。
    train_df["int_date"] = train_df["fecha_dato"].map(date_to_int).astype(np.int8)

    # 独自に開発した one-hot-encodingを遂行します。
    custom_one_hot(train_df, features, "indresi", {"n":"N"})
    custom_one_hot(train_df, features, "indext", {"s":"S"})
    custom_one_hot(train_df, features, "conyuemp", {"n":"N"})
    custom_one_hot(train_df, features, "sexo", {"h":"H", "v":"V"})
    custom_one_hot(train_df, features, "ind_empleado", {"a":"A", "b":"B", "f":"F", "n":"N"})
    custom_one_hot(train_df, features, "ind_nuevo", {"new":1})
    custom_one_hot(train_df, features, "segmento", {"top":"01 - TOP", "particulares":"02 - PARTICULARES", "universitario":"03 - UNIVERSITARIO"})
    custom_one_hot(train_df, features, "indfall", {"s":"S"})
    custom_one_hot(train_df, features, "tiprel_1mes", {"a":"A", "i":"I", "p":"P", "r":"R"}, check=True)
    custom_one_hot(train_df, features, "indrel", {"1":1, "99":99})

    # 欠損値を0.0に代替し、その他は +1.0を加え、整数に変換します。
    train_df["ind_actividad_cliente"] = train_df["ind_actividad_cliente"].map(lambda x: 0.0 if math.isnan(x) else x+1.0).astype(np.int8)
    features.append("ind_actividad_cliente")

    # 欠損値を 0.0に代替し、 “P”を 5に代替し、整数に変換します。
    train_df["indrel_1mes"] = train_df["indrel_1mes"].map(lambda x: 5.0 if x == "P" else x).astype(float).fillna(0.0).astype(np.int8)
    features.append("indrel_1mes")
    
    # データ前処理／特徴量エンジニアリングが1次的に完了したデータフレームtrain_dfと、学習に使用する変数リスト featuresを tuple 形式に変換します。
    return train_df, tuple(features)


def make_prev_df(train_df, step):
    # 新しいデータフレームに ncodpersを追加し、int_dateを stepだけ移動した値を入れます。
    prev_df = pd.DataFrame()
    prev_df["ncodpers"] = train_df["ncodpers"]
    prev_df["int_date"] = train_df["int_date"].map(lambda x: x+step).astype(np.int8)

    # “変数名_prev1” 形式の lag 変数を生成します。
    prod_features = ["%s_prev%s" % (prod, step) for prod in products]
    for prod, prev in zip(products, prod_features):
        prev_df[prev] = train_df[prod]

    return prev_df, tuple(prod_features)


def load_data():
    # “データ準備”で統合したデータを読み込みます。
    fname = "../input/8th.clean.all.csv"
    train_df = pd.read_csv(fname, dtype=dtypes)

    # productsは util.pyで定義した 24個の金融商品の名前です。
    # 欠損値を 0.0で代替し、整数型に変換します。
    for prod in products:
        train_df[prod] = train_df[prod].fillna(0.0).astype(np.int8)

    # 48個の変数ごとに前処理／特徴量エンジニアリングを適用します。
    train_df, features = apply_transforms(train_df)

    prev_dfs = []
    prod_features = None

    use_features = frozenset([1,2])
    # 1 ~ 5までの stepに対して make_prev_df()を通して lag-n データを生成します。
    for step in range(1,6):
        prev1_train_df, prod1_features = make_prev_df(train_df, step)
        # 生成した lag データは prev_dfs リストに保存します。
        prev_dfs.append(prev1_train_df)
        # featuresには lag-1,2だけ追加します。
        if step in use_features:
            features += prod1_features
        # prod_featuresには lag-1の変数名だけを保存します。
        if step == 1:
            prod_features = prod1_features

    return train_df, prev_dfs, features, prod_features


def join_with_prev(df, prev_df, how):
    # pandas merge 関数を通して join
    df = df.merge(prev_df, on=["ncodpers", "int_date"], how=how)
    # 24個の金融変数を小数型に変換します。
    for f in set(prev_df.columns.values.tolist()) - set(["ncodpers", "int_date"]):
        df[f] = df[f].astype(np.float16)
    return df

def make_data():
    train_df, prev_dfs, features, prod_features = load_data()

    for i, prev_df in enumerate(prev_dfs):
        with Timer("join train with prev%s" % (i+1)):
            how = "inner" if i == 0 else "left"
            train_df = join_with_prev(train_df, prev_df, how=how)

    # 24個の金融変数に対して for loopをまわします。
    for prod in products:
        # [1~3], [1~5], [2~5] の3つの区間に対して標準偏差を求めます。
        for begin, end in [(1,3),(1,5),(2,5)]:
            prods = ["%s_prev%s" % (prod, i) for i in range(begin,end+1)]
            mp_df = train_df.as_matrix(columns=prods)
            stdf = "%s_std_%s_%s" % (prod,begin,end)

            # np.nanstdで標準偏差を求め、featuresに新規派生変数の名前を追加します。
            train_df[stdf] = np.nanstd(mp_df, axis=1)
            features += (stdf,)

        # [2~3], [2~5] の2つの区間に対して最小値／最大値を求めます。
        for begin, end in [(2,3),(2,5)]:
            prods = ["%s_prev%s" % (prod, i) for i in range(begin,end+1)]
            mp_df = train_df.as_matrix(columns=prods)

            minf = "%s_min_%s_%s"%(prod,begin,end)
            train_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)

            maxf = "%s_max_%s_%s"%(prod,begin,end)
            train_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            features += (minf,maxf,)

    # 顧客の固有識別番号(ncodpers), 整数で表現された日付(int_date), 実際の日付(fecha_dato), 24個の金融変数(products)と学習に使用するために前処理／特徴量エンジニアリングをした変数(features)が重要な変数です。
    leave_columns = ["ncodpers", "int_date", "fecha_dato"] + list(products) + list(features)
    # 重複値がないかを確認します。
    assert len(leave_columns) == len(set(leave_columns))
    # train_dfで主要な変数だけを抽出します。
    train_df = train_df[leave_columns]

    return train_df, features, prod_features


def make_submission(f, Y_test, C):
    Y_ret = []
    # ファイルの最初の行にheaderを書き込みます。
    f.write("ncodpers,added_products\n".encode('utf-8'))
    # 顧客識別番号(C)と、予測結果(Y_test)の for loop
    for c, y_test in zip(C, Y_test):
        # (確率値、金融変数名、金融変数id)の tupleを求めます。
        y_prods = [(y,p,ip) for y,p,ip in zip(y_test, products, range(len(products)))]
        # 確率値をもとに、上位7個の結果だけを抽出します。
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        # 金融変数idを Y_retに保存します。
        Y_ret.append([ip for y,p,ip in y_prods])
        y_prods = [p for y,p,ip in y_prods]
        # ファイルに “顧客識別番号、7個の金融変数”を書き込みます。
        f.write(("%s,%s\n" % (int(c), " ".join(y_prods))).encode('utf-8'))
    # 上位7個の予測値を返します。
    return Y_ret


def train_predict(all_df, features, prod_features, str_date, cv):
    # all_df : 統合データ
    # features : 学習に使用する変数
    # prod_features : 24個の金融変数
    # str_date : 予測結果を算出する日付。 2016-05-28の場合は、訓練データの一部であり正答がわかっているので交差検証を意味し、2016-06-28の場合はKaggleにアップロードするデータを生成します。
    # cv : 交差検証を実行するかどうか

    # str_dateで予測結果を算出する日付を指定します。
    test_date = date_to_int(str_date)
    # 訓練データは test_date 以前のすべてのデータを使用します。
    train_df = all_df[all_df.int_date < test_date]
    # テストデータを統合データから分離します。
    test_df = pd.DataFrame(all_df[all_df.int_date == test_date])

    # 新規購買顧客だけを訓練データへ抽出します。
    X = []
    Y = []
    for i, prod in enumerate(products):
        prev = prod + "_prev1"
        # 新規購買顧客を prX に保存します。
        prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
        # prY には新規購買に対する label 値を保存します。
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)

    XY = pd.concat(X)
    Y = np.hstack(Y)
    # XY は新規購買データだけを含みます。
    XY["y"] = Y

    # メモリから変数を削除
    del train_df
    del all_df

    # データ別の加重値を計算するため、新しい変数 (ncodpers + fecha_dato)を生成します。
    XY["ncodepers_fecha_dato"] = XY["ncodpers"].astype(str) + XY["fecha_dato"]
    uniqs, counts = np.unique(XY["ncodepers_fecha_dato"], return_counts=True)
    # ネイピア数(e)を用いて, countが高いデータに低い加重値を与えます。
    weights = np.exp(1/counts - 1)

    # 加重値を XY データに追加します。
    wdf = pd.DataFrame()
    wdf["ncodepers_fecha_dato"] = uniqs
    wdf["counts"] = counts
    wdf["weight"] = weights
    XY = XY.merge(wdf, on="ncodepers_fecha_dato")

    # 交差検証のため、XYを訓練：検証(8:2)に分離します。
    mask = np.random.rand(len(XY)) < 0.8
    XY_train = XY[mask]
    XY_validate = XY[~mask]

    # テストデータで加重値はすべて1です。
    test_df["weight"] = np.ones(len(test_df), dtype=np.int8)

    # テストデータから“新規購買”の正答値を抽出します。
    test_df["y"] = test_df["ncodpers"]
    Y_prev = test_df.as_matrix(columns=prod_features)
    for prod in products:
        prev = prod + "_prev1"
        padd = prod + "_add"
        # 新規購買であるかどうかを求めます。
        test_df[padd] = test_df[prod] - test_df[prev]

    test_add_mat = test_df.as_matrix(columns=[prod + "_add" for prod in products])
    C = test_df.as_matrix(columns=["ncodpers"])
    test_add_list = [list() for i in range(len(C))]
    # 評価基準 MAP@7 の計算のため、顧客別新規購買正答値をtest_add_listに記録します。
    count = 0
    for c in range(len(C)):
        for p in range(len(products)):
            if test_add_mat[c,p] > 0:
                test_add_list[c].append(p)
                count += 1
    
    # 交差検証で、テストデータから分離されたデータが得ることのできる
    if cv:
        max_map7 = mapk(test_add_list, test_add_list, 7, 0.0)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        print("Max MAP@7", str_date, max_map7, max_map7*map7coef)

    # LightGBM モデル学習の後、予測結果を保存します。
    Y_test_lgbm = engines.lightgbm(XY_train, XY_validate, test_df, features, XY_all = XY, restore = (str_date == "2016-06-28"))
    test_add_list_lightgbm = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.lightgbm.csv.gz" % str_date, "wb"), Y_test_lgbm - Y_prev, C)

    # 交差検証の場合, LightGBM モデルのテストデータ MAP@7 評価基準を出力します。
    if cv:
        map7lightgbm = mapk(test_add_list, test_add_list_lightgbm, 7, 0.0)
        print("LightGBMlib MAP@7", str_date, map7lightgbm, map7lightgbm*map7coef)

    # XGBoost モデル学習の後、予測結果を保存します。
    Y_test_xgb = engines.xgboost(XY_train, XY_validate, test_df, features, XY_all = XY, restore = (str_date == "2016-06-28"))
    test_add_list_xgboost = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.xgboost.csv.gz" % str_date, "wb"), Y_test_xgb - Y_prev, C)

    # 交差検証の場合, XGBoost モデルのテストデータ MAP@7 評価基準を出力します。
    if cv:
        map7xgboost = mapk(test_add_list, test_add_list_xgboost, 7, 0.0)
        print("XGBoost MAP@7", str_date, map7xgboost, map7xgboost*map7coef)

    # 平方した後、平行根を求めるやり方でアンサンブルを遂行します。
    Y_test = np.sqrt(np.multiply(Y_test_xgb, Y_test_lgbm))
    # アンサンブルの結果を保存し、テストデータに対する MAP@7 を出力します。
    test_add_list_xl = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.xgboost-lightgbm.csv.gz" % str_date, "wb"), Y_test - Y_prev, C)

    # 正答値の test_add_listとアンサンブルモデルの予測値を mapk 関数に入れて、評価基準の点数を確認します。
    if cv:
        map7xl = mapk(test_add_list, test_add_list_xl, 7, 0.0)
        print("XGBoost+LightGBM MAP@7", str_date, map7xl, map7xl*map7coef)




if __name__ == "__main__":
    all_df, features, prod_features = make_data()
    
    # 特徴量エンジニアリングが完了したデータを保存します。
    train_df.to_pickle("../input/8th.feature_engineer.all.pkl")
    pickle.dump((features, prod_features), open("../input/8th.feature_engineer.cv_meta.pkl", "wb"))

    train_predict(all_df, features, prod_features, "2016-05-28", cv=True)
    train_predict(all_df, features, prod_features, "2016-06-28", cv=False)
