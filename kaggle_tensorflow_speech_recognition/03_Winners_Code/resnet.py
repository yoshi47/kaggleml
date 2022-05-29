import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d

# ResNet モデルが定義されたPythonクラス
class ResModel(torch.nn.Module):

    # モデルの構造を定義するための準備作業をします。
    def __init__(self):
        super(ResModel, self).__init__()

        # 12-class 分類問題であり、モデルが使用するチャンネルの数を128に指定します。
        n_labels = 12
        n_maps = 128

        # 1チャンネルの入力値を n_maps(128)チャンネルに出力する3x3 Conv カーネルを事前に定義します。
        self.conv0 = torch.nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)

        # 入力と出力のチャンネルがn_maps(128)である 3x3 Conv カーネルを 9個、事前に定義します。
        self.n_layers = n_layers = 9
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False) for _ in range(n_layers)])

        # max-pooling 階層を事前に定義します。
        self.pool = MaxPool2d(2, return_indices=True)

        # batch_normalizationと conv モジュールを事前に定義します。
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), torch.nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        # n_maps(128)を入力として受け取り、n_labels(12)を出力する最終線形階層を事前に定義します。
        self.output = torch.nn.Linear(n_maps, n_labels)

    # モデルの欠損値を出力するforward関数です。
    def forward(self, x):
        # 9階層の Conv モジュールと最終線形階層の総計10階層モジュールです。
        for i in range(self.n_layers + 1):
            # 入力値xをconvモジュールに適用した後、relu activationを通過させます。
            y = F.relu(getattr(self, "conv{}".format(i))(x))

            # residualモジュール生成のためのコードです。
            # iが偶数のとき、xはy + old_xの和でresidual演算が遂行されます。
            if i == 0:
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y

            # 2番目の階層からはbatch_normalizationを適用します。
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)

            # max_poolingは使用しないよう設定します。
            pooling = False
            if pooling:
                x_pool, pool_indices = self.pool(x)
                x = self.unpool(x_pool, pool_indices, output_size=x.size())

        # view 関数によって xの大きさを調整します。
        x = x.view(x.size(0), x.size(1), -1)
        # 2番目の dimensionに対して平均値を求めます。
        x = torch.mean(x, 2)
        # 最後の線形階層を通過した結果値を返します。
        return self.output(x)
