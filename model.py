import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from enum import Enum
from ukan_kan import UkanFeatureMapBlock


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpKAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.kan = UkanFeatureMapBlock(dim=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.kan(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ConcatSkip(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = BasicConv2d(c_in, c_out, kernel_size=1)

    def forward(self, rgb, topo):
        x = torch.cat([rgb, topo], dim=1)
        return self.conv(x)


class FusionMode(Enum):
    CONCAT = "concat"


class UNet(nn.Module):
    def __init__(
        self,
        rgb_backbone: str = "efficientnet_b5",
        pretrained_rgb: bool = True,
        topo_backbone: str = "efficientnet_b5",
        topo_in_chans: int = 1,
        pretrained_topo: bool = True,
        fusion: FusionMode = FusionMode.CONCAT,
        use_kan: bool = True,
        decoder_channels: int = 64,
    ) -> None:
        super(UNet, self).__init__()
        self.rgb_encoder = timm.create_model(
            rgb_backbone,
            pretrained=pretrained_rgb,
            in_chans=3,
            features_only=True,
        )

        self.topo_in_chans = topo_in_chans
        self.topo_encoder = timm.create_model(
            topo_backbone,
            pretrained=pretrained_topo,
            in_chans=topo_in_chans,
            features_only=True,
        )

        rgb_channels = self.rgb_encoder.feature_info.channels()
        topo_channels = self.topo_encoder.feature_info.channels()
        self.reductions = self.rgb_encoder.feature_info.reduction()
        self.num_stages = len(rgb_channels)

        if len(topo_channels) != self.num_stages:
            raise ValueError(
                f"Stage count mismatch: RGB backbone '{rgb_backbone}' has {self.num_stages} stages, "
                f"while Topo backbone '{topo_backbone}' has {len(topo_channels)} stages."
            )

        if fusion == FusionMode.CONCAT:
            self.fuses = nn.ModuleList([
                ConcatSkip(rgb_channels[i] + topo_channels[i], decoder_channels)
                for i in range(self.num_stages)
            ])
        else:
            raise NotImplementedError(f"Fusion mode {fusion} is not implemented.")

        UpBlock = UpKAN if use_kan else Up
        # Decoder performs (num_stages - 1) upsampling operations
        self.up_blocks = nn.ModuleList([
            UpBlock(2 * decoder_channels, decoder_channels)
            for _ in range(self.num_stages - 1)
        ])

        self.side1 = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.side2 = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(self, x, x_topo=None):
        if x_topo is None:
            if x.shape[1] > 3:
                x_rgb = x[:, :3, :, :]
                x_topo = x[:, 3:, :, :]
            else:
                raise ValueError(
                    f"Model is configured with {self.topo_in_chans} topography channels, "
                    f"but input has only {x.shape[1]} channels. Pass both RGB and topography channels, "
                    f"or pass x_topo explicitly."
                )
        else:
            x_rgb = x

        # 1. Optical RGB branch
        feats_rgb = self.rgb_encoder(x_rgb)

        # 2. Topography branch
        feats_topo = self.topo_encoder(x_topo)

        # 3. Multi-scale feature fusion
        fused = [self.fuses[i](feats_rgb[i], feats_topo[i]) for i in range(self.num_stages)]

        # 4. Decoder traversal from deepest level up to coarsest
        cur = fused[-1]

        # First upsampling step: deepest skip
        cur = self.up_blocks[0](cur, fused[-2])
        out1 = F.interpolate(self.side1(cur), scale_factor=self.reductions[-2], mode='bilinear', align_corners=False)

        # Second upsampling step: mid skip
        cur = self.up_blocks[1](cur, fused[-3])
        out2 = F.interpolate(self.side2(cur), scale_factor=self.reductions[-3], mode='bilinear', align_corners=False)

        # Remaining upsampling steps
        for k in range(2, self.num_stages - 1):
            cur = self.up_blocks[k](cur, fused[self.num_stages - 2 - k])

        # Final primary head projection
        out = F.interpolate(self.head(cur), scale_factor=self.reductions[0], mode='bilinear', align_corners=False)

        return out, out1, out2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model = UNet(rgb_backbone="efficientnet_b5", topo_backbone="efficientnet_b5", pretrained_rgb=False, pretrained_topo=False).to(device)
        x = torch.randn(1, 4, 128, 128).to(device)
        out, out1, out2 = model(x)
        print("EfficientNet-B5 UNet forward outputs:", out.shape, out1.shape, out2.shape)