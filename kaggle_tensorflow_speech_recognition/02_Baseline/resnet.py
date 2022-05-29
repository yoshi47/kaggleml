import torch
import torch.nn.functional as F
from torch.nn import MaxPool2d

# ResNet モデルを実装します。
class ResModel(torch.nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        # 今回のコンペティションで使われる label の個数は 12です。
        n_labels = 12
        n_maps = 128
        # 合計9階層のモデルを積み重ねます。
        self.n_layers = n_layers = 9
        # 最初の階層に使用するconvolutional モジュールを定義します。
        self.conv0 = torch.nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        # MaxPooling モジュールを定義します。
        self.pool = MaxPool2d(2, return_indices=True)
        # 2階層以後に使用する convolutional モジュールを定義します。
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False) for _ in range(n_layers)])
        # BatchNormalization モジュールと conv モジュールを組み合わせます。
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), torch.nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        # 最後の階層には線形モジュールを追加します。
        self.output = torch.nn.Linear(n_maps, n_labels)

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                old_x = y
            # 以前のlayerの結果値(old_x)と今回のlayerの結果値(y)を加えたのがresidualモジュールです。
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            # BatchNormalizationでパラメータ値を正規化します。
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
            # poolingを使用するかどうかをTrue/Falseで指定します。
            pooling = False
            if pooling:
                x_pool, pool_indices = self.pool(x)
                x = self.unpool(x_pool, pool_indices, output_size=x.size())
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        # 最終の線形階層を通過した結果値を返します。
        return self.output(x)
