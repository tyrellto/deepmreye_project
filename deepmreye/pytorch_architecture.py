import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    """3D convolution followed by optional average pooling and activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.pool = nn.AvgPool3d(2, stride=2) if stride > 1 else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.activation(x)


class ResBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, channels, filters, groups, activation):
        super().__init__()
        self.shortcut = Conv3dBlock(channels, filters, 1, 1, activation)

        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = Conv3dBlock(channels, filters, 3, 1, activation)

        self.norm2 = nn.GroupNorm(groups, filters)
        self.conv2 = Conv3dBlock(filters, filters, 3, 1, activation)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.conv2(out)
        return out + residual


class DownsampleBlock(nn.Module):
    """Stack of residual blocks followed by downsampling."""

    def __init__(self, in_channels, filters, depth, multiplier, groups, activation):
        super().__init__()
        blocks = []
        channels = in_channels
        skip_channels = []
        for level in range(depth):
            n_filters = int(multiplier**level) * filters
            for _ in range(level):
                blocks.append(ResBlock(channels, n_filters, groups, activation))
                channels = n_filters
            skip_channels.append(channels)
            if level < depth - 1:
                blocks.append(Conv3dBlock(channels, n_filters, 3, 2, activation))
                channels = n_filters
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.GroupNorm(groups, channels)
        self.activation = activation
        self.skip_channels = skip_channels
        self.out_channels = channels

    def forward(self, x):
        skip_layers = []
        idx = 0
        for level in range(len(self.skip_channels)):
            for _ in range(level):
                x = self.blocks[idx](x)
                idx += 1
            skip_layers.append(x)
            if level < len(self.skip_channels) - 1:
                x = self.blocks[idx](x)
                idx += 1
        x = self.out_norm(x)
        x = self.activation(x)
        return x, skip_layers


class RegressionBlock(nn.Module):
    """Fully connected regression head repeated for each inner timestep."""

    def __init__(self, in_features, num_dense, num_fc, activation, dropout_rate, inner_timesteps, mc_dropout):
        super().__init__()
        layers = []
        for _ in range(num_dense):
            layers.append(nn.Linear(in_features, num_fc))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU() if activation is F.relu else activation)
            in_features = num_fc
        self.core = nn.Sequential(*layers)
        self.out = nn.Linear(in_features, 2)
        self.inner_timesteps = inner_timesteps
        self.mc_dropout = mc_dropout

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1).repeat(1, self.inner_timesteps, 1)
        outs = []
        for i in range(self.inner_timesteps):
            h = self.core(x[:, i])
            h = F.dropout(
                h,
                p=self.core[1].p,
                training=self.mc_dropout or self.training,
            )
            outs.append(self.out(h).unsqueeze(1))
        return torch.cat(outs, dim=1)


class ConfidenceBlock(nn.Module):
    def __init__(self, in_features, num_fc, activation, dropout_rate, inner_timesteps, mc_dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, num_fc)
        self.fc2 = nn.Linear(num_fc, inner_timesteps)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.mc_dropout = mc_dropout

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = F.dropout(
            x,
            p=self.dropout.p,
            training=self.mc_dropout or self.training,
        )
        return self.activation(self.fc2(x))


class StandardModel(nn.Module):
    """PyTorch implementation of the DeepMReye model."""

    def __init__(self, input_shape, opts):
        super().__init__()
        act = nn.ReLU()
        self.activation = act
        self.gaussian_noise = opts.get("gaussian_noise", 0)
        self.mc_dropout = opts.get("mc_dropout", False)

        channels, x, y, z = input_shape

        self.first = Conv3dBlock(channels, opts["filters"], opts["kernel"], 1, act)
        self.dropout = nn.Dropout3d(opts["dropout_rate"])

        self.down = DownsampleBlock(
            opts["filters"],
            opts["filters"],
            opts["depth"],
            opts["multiplier"],
            opts["groups"],
            act,
        )

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, channels, x, y, z)
            h = self.first(dummy)
            h = self.dropout(h)
            h, _ = self.down(h)
            bottleneck_features = int(h.flatten(1).shape[1])
        self.regression = RegressionBlock(
            in_features=bottleneck_features,
            num_dense=opts["num_dense"],
            num_fc=opts["num_fc"],
            activation=act,
            dropout_rate=opts["dropout_rate"],
            inner_timesteps=opts["inner_timesteps"],
            mc_dropout=opts.get("mc_dropout", False),
        )
        self.confidence = ConfidenceBlock(
            in_features=bottleneck_features,
            num_fc=opts["num_fc"],
            activation=act,
            dropout_rate=opts["dropout_rate"],
            inner_timesteps=opts["inner_timesteps"],
            mc_dropout=opts.get("mc_dropout", False),
        )

    def forward(self, x):
        if self.gaussian_noise > 0:
            noise = torch.randn_like(x) * self.gaussian_noise
            x = x + noise
        x = self.first(x)
        x = F.dropout(
            x,
            p=self.dropout.p,
            training=self.mc_dropout or self.training,
        )
        x, _ = self.down(x)
        x = self.flatten(x)
        reg = self.regression(x)
        conf = self.confidence(x)
        return reg, conf


def compute_standard_loss(out_confidence, real_reg, pred_reg):
    loss_euclidean = torch.sqrt(torch.sum((real_reg - pred_reg) ** 2, dim=-1))
    loss_confidence = (loss_euclidean - out_confidence) ** 2
    return loss_euclidean.mean(), loss_confidence.mean()
