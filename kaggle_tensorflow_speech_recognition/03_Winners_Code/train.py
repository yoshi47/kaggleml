# ResNet モデルが定義したモデル関数を読み込みます。
from resnet import ResModel
# モデル学習用関数 trainerを読み込みます。
from trainer import train_model
# データ前処理用関数を読み込みます。
from data import preprocess_mel, preprocess_mfcc

# ResNet モデルは mel, mfccで前処理された入力値を受け取るモデルをそれぞれ学習します。 
list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]
# 同じモデルを4個学習し、4個のモデルの結果の平均値を最終結果として使用します。(baggingアンサンブル)
BAGGING_NUM=4

# モデルを学習し、最終モデルをもとにテストデータに対する予測結果を保存するツール関数です。
def train_and_predict(cfg_dict, preprocess_list):
    # 前処理の方式によってそれぞれ別のモデルを学習します。
    for p, preprocess_fun in preprocess_list:
        # モデル学習の設定値(config)を定義します。
        cfg = cfg_dict.copy()
        cfg['preprocess_fun'] = preprocess_fun
        cfg['CODER'] += '_%s' %p
        cfg['bagging_num'] = BAGGING_NUM
        print("training ", cfg['CODER'])
        # モデルを学習します！
        train_model(**cfg)

# ResNet モデル学習の設定値です。
res_config = {
    'model_class': ResModel,
    'is_1d': False,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'resnet'
}

print("train resnet.........")
train_and_predict(res_config, list_2d)

se_config = {
    'model_class': SeModel,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'senet'
}

print("train senet..........")
train_and_predict(se_config, list_2d)

dense_config = {
    'model_class': densenet121,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 16,
    'epochs': 100,
    'CODER': 'dense'
}

print("train densenet.........")
train_and_predict(dense_config, list_2d)

vgg2d_config = {
    'model_class': vgg2d,
    'is_1d': False,
    'reshape_size': 128,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg2d'
}

print("train vgg2d...........")
train_and_predict(vgg2d_config, list_2d)

vgg1d_config = {
    'model_class': vgg1d,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 32,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on raw features..........")
train_and_predict(vgg1d_config, [('raw', preprocess_wav)])

vggmel_config = {
    'model_class': vggmel,
    'is_1d': True,
    'reshape_size': None,
    'BATCH_SIZE': 64,
    'epochs': 100,
    'CODER': 'vgg1d'
}

print("train vgg1d on mel features..........")
train_and_predict(vggmel_config, [('mel', preprocess_mel)])


