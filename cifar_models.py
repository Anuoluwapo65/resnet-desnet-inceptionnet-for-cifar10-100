from types import SimpleNamespace

import torch
import torch.nn as nn


act_fn_nm = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "swish": nn.SiLU(),
    "gelu": nn.GELU(),
    "elu": nn.ELU(),
}


model_dict = {}


def _init_conv_weight(module, act_fn_name):
    if act_fn_name == "relu":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif act_fn_name == "leakyrelu":
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
    else:
        gain_by_activation = {
            "tanh": nn.init.calculate_gain("tanh"),
            "sigmoid": nn.init.calculate_gain("sigmoid"),
            "swish": nn.init.calculate_gain("relu"),
            "gelu": nn.init.calculate_gain("relu"),
            "elu": nn.init.calculate_gain("relu"),
        }
        nn.init.xavier_normal_(module.weight, gain=gain_by_activation[act_fn_name])


def _normalize_model_request(model_name, model_hparams):
    normalized_name = model_name
    normalized_hparams = dict(model_hparams)

    if normalized_name == "ResNet":
        normalized_name = "Resnet"

    if "act_fn_nm" in normalized_hparams and "act_fn_name" not in normalized_hparams:
        normalized_hparams["act_fn_name"] = normalized_hparams.pop("act_fn_nm")

    block_aliases = {
        "ResNetBlock": "resnet",
        "preact_ResNetBlock": "preactresnet",
    }
    block_name = normalized_hparams.get("block_name")
    if block_name in block_aliases:
        normalized_hparams["block_name"] = block_aliases[block_name]

    return normalized_name, normalized_hparams


def create_model(model_name, model_hparams):
    model_name, model_hparams = _normalize_model_request(model_name, model_hparams)
    model_cls = model_dict.get(model_name)
    assert model_cls is not None, (
        f'unknown model_name "{model_name}". '
        f"Available: {sorted(model_dict.keys())}"
    )
    return model_cls(**model_hparams)


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out, c_red, act_fn):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn,
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn,
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn,
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn,
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn,
        )
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn,
        )

    def forward(self, x):
        x_1 = self.conv_1x1(x)
        x_2 = self.conv_3x3(x)
        x_3 = self.conv_5x5(x)
        x_4 = self.max_pool(x)
        return torch.cat([x_1, x_2, x_3, x_4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=act_fn_nm[act_fn_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn,
        )
        self.inception_blocks = nn.Sequential(
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
        )
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes),
        )

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv_weight(module, self.hparams.act_fn_name)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        super().__init__()
        if not subsample:
            c_out = c_in
        self.net = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn,
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2)
            if subsample
            else None
        )
        self.act_fn = act_fn

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.act_fn(x + z)


class PreactResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        super().__init__()
        if not subsample:
            c_out = c_in
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                stride=1 if not subsample else 2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn,
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )
        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(c_in),
                act_fn,
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False),
            )
            if subsample
            else None
        )
        self.act_fn = act_fn

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.act_fn(z + x)


resnet_block_name = {
    "resnet": ResNetBlock,
    "preactresnet": PreactResNetBlock,
}


class Resnet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_blocks=None,
        c_hidden=None,
        act_fn_name="relu",
        block_name="resnet",
        **kwargs,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]
        if c_hidden is None:
            c_hidden = [16, 32, 64]
        self.hparams = SimpleNamespace(
            num_blocks=num_blocks,
            c_hidden=c_hidden,
            act_fn_name=act_fn_name,
            act_fn=act_fn_nm[act_fn_name],
            block_name=block_name,
            block_class=resnet_block_name[block_name],
            num_classes=num_classes,
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        if self.hparams.block_class == PreactResNetBlock:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn,
            )

        blocks = []
        current_in_channels = c_hidden[0]
        for block_idx, num_blocks_in_stage in enumerate(self.hparams.num_blocks):
            output_channels_for_stage = c_hidden[block_idx]
            for i in range(num_blocks_in_stage):
                subsample = i == 0 and block_idx > 0
                blocks.append(
                    self.hparams.block_class(
                        c_in=current_in_channels,
                        act_fn=self.hparams.act_fn,
                        subsample=subsample,
                        c_out=output_channels_for_stage,
                    )
                )
                current_in_channels = output_channels_for_stage
        self.blocks = nn.Sequential(*blocks)

        if self.hparams.block_class == PreactResNetBlock:
            self.output_net = nn.Sequential(
                nn.BatchNorm2d(current_in_channels),
                self.hparams.act_fn,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(current_in_channels, self.hparams.num_classes),
            )
        else:
            self.output_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(current_in_channels, self.hparams.num_classes),
            )

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv_weight(module, self.hparams.act_fn_name)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


ResNet = Resnet


class DenseLayer(nn.Module):
    def __init__(self, act_fn, b_size, growth_rate, c_in):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, b_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(b_size * growth_rate),
            act_fn,
            nn.Conv2d(
                b_size * growth_rate,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.input_net(x)


class DenseBlock(nn.Module):
    def __init__(self, act_fn, num_layers, b_size, growth_rate, c_in):
        super().__init__()
        layers = []
        current_input_channels = c_in
        for _ in range(num_layers):
            layers.append(
                DenseLayer(
                    c_in=current_input_channels,
                    act_fn=act_fn,
                    b_size=b_size,
                    growth_rate=growth_rate,
                )
            )
            current_input_channels += growth_rate
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.model:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x


class Transitional(nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn,
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(
        self,
        c_in,
        b_size=2,
        num_layers=None,
        num_classes=10,
        growth_rate=16,
        act_fn_name="relu",
        **kwargs,
    ):
        super().__init__()
        if num_layers is None:
            num_layers = [6, 6, 6, 6]
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            growth_rate=growth_rate,
            num_layers=num_layers,
            act_fn_name=act_fn_name,
            act_fn=act_fn_nm[act_fn_name],
            b_size=b_size,
            c_in=c_in,
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.growth_rate * self.hparams.b_size
        self.input_net = nn.Sequential(
            nn.Conv2d(self.hparams.c_in, c_hidden, kernel_size=3, padding=1)
        )

        blocks = []
        for block_index, num_layers_in_block in enumerate(self.hparams.num_layers):
            blocks.append(
                DenseBlock(
                    act_fn=self.hparams.act_fn,
                    c_in=c_hidden,
                    num_layers=num_layers_in_block,
                    b_size=self.hparams.b_size,
                    growth_rate=self.hparams.growth_rate,
                )
            )
            c_hidden = c_hidden + num_layers_in_block * self.hparams.growth_rate
            if block_index < len(self.hparams.num_layers) - 1:
                blocks.append(
                    Transitional(
                        c_in=c_hidden,
                        c_out=c_hidden // 2,
                        act_fn=self.hparams.act_fn,
                    )
                )
                c_hidden = c_hidden // 2
        self.model = nn.Sequential(*blocks)
        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            self.hparams.act_fn,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, self.hparams.num_classes),
        )

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv_weight(module, self.hparams.act_fn_name)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.model(x)
        x = self.output_net(x)
        return x


model_dict.update(
    {
        "GoogleNet": GoogleNet,
        "Resnet": Resnet,
        "DenseNet": DenseNet,
    }
)
