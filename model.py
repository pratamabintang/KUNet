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
        rgb_backbone: str = "convnext_tiny",
        pretrained_rgb: bool = True,
        topo_backbone: str = "convnext_tiny",
        topo_in_chans: int = 1,
        pretrained_topo: bool = True,
        fusion: FusionMode = FusionMode.CONCAT,
        use_kan: bool = True
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

        # self.rfb1 = RFB_modified(144, 64)
        # self.rfb2 = RFB_modified(288, 64)
        # self.rfb3 = RFB_modified(576, 64)
        # self.rfb4 = RFB_modified(1152, 64)

        if fusion == FusionMode.CONCAT:
            self.fuse1 = ConcatSkip(192, 64)
            self.fuse2 = ConcatSkip(384, 64)
            self.fuse3 = ConcatSkip(768, 64)
            self.fuse4 = ConcatSkip(1536, 64)

        if use_kan:
            self.up1 = (UpKAN(128, 64))
            self.up2 = (UpKAN(128, 64))
            self.up3 = (UpKAN(128, 64))
            self.up4 = (UpKAN(128, 64))
        else:
            self.up1 = (Up(128, 64))
            self.up2 = (Up(128, 64))
            self.up3 = (Up(128, 64))
            self.up4 = (Up(128, 64))

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

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

        # 1. Optical RGB branch (timm ConvNeXt) (96, 192, 384, 768)
        x1, x2, x3, x4 = self.rgb_encoder(x_rgb)

        # 2. Topography branch (timm ConvNeXt) (96, 192, 384, 768)
        t1, t2, t3, t4 = self.topo_encoder(x_topo)

        # x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x1 = self.fuse1(x1, t1)
        x2 = self.fuse2(x2, t2)
        x3 = self.fuse3(x3, t3)
        x4 = self.fuse4(x4, t4)

        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model = UNet(pretrained_topo=False).to(device)
        x = torch.randn(1, 4, 512, 512).to(device)
        out, out1, out2 = model(x)
        print("Single tensor forward outputs:", out.shape, out1.shape, out2.shape)