"""Dual-branch (detail + spatial) segmentation head with a BiSeNet-style feature fusion, in 3D.

Author: Fl0rian
"""
import torch
import torch.nn as nn
import monai.networks.blocks as mn


class ConvBNReLU(nn.Module):
    """3D convolution followed by batch norm and ReLU."""

    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, groups: int = 1, bias: bool = False) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class ContentBranch(nn.Module):
    """Detail branch: a shallow stack of strided convs, truncated according to ``skip``."""

    def __init__(self, in_c: int, skip: int) -> None:
        super(ContentBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(in_c, 32, 3, stride=2),
            ConvBNReLU(32, 32, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(32, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

        self.branch = nn.Sequential(
            self.S1 if skip < 3 else nn.Identity(),
            self.S2 if skip < 2 else nn.Identity(),
            self.S3 if skip < 1 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.branch(x)
        return feat


class StemBlock(nn.Module):
    """Initial downsampling stem: a strided conv followed by a parallel conv/pool fusion."""

    def __init__(self, in_c: int) -> None:
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_c, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool3d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 32, 3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class C2Block(nn.Module):
    """Inverted-residual block with a stride-1 depthwise expansion and a residual add."""

    def __init__(self, in_chan: int, out_chan: int, exp_ratio: int = 6) -> None:
        super(C2Block, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv3d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm3d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class C1Block(nn.Module):
    """Inverted-residual block with a stride-2 depthwise expansion and a downsampled shortcut."""

    def __init__(self, in_chan: int, out_chan: int, exp_ratio: int = 6) -> None:
        super(C1Block, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv3d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm3d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm3d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv3d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.Conv3d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm3d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SpatialBranch(nn.Module):
    """Spatial branch: a stem block followed by a stack of C1/C2 blocks, truncated by ``skip``."""

    def __init__(self, in_c: int, skip: int) -> None:
        super(SpatialBranch, self).__init__()
        self.S1S2 = StemBlock(in_c)
        self.S3 = nn.Sequential(
            C1Block(32, 32),
            C2Block(32, 32),
        )
        self.S4 = nn.Sequential(
            C1Block(32, 64),
            C2Block(64, 64),
        )
        self.S5 = nn.Sequential(
            C1Block(64, 128),
            C2Block(128, 128),
            C2Block(128, 128),
            C2Block(128, 128),
        )

        self.branch = nn.Sequential(
            self.S1S2,
            self.S3 if skip < 3 else nn.Identity(),
            self.S4 if skip < 2 else nn.Identity(),
            self.S5 if skip < 1 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.branch(x)
        return feat


class MergeLayer(nn.Module):
    """Bilateral guided-aggregation layer fusing the detail and spatial branch features."""

    def __init__(self, in_c: int, skip: int) -> None:
        super(MergeLayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, groups=in_c, bias=False),
            nn.BatchNorm3d(128),
            nn.Conv3d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv3d(
                in_c, 128, kernel_size=3, stride=1,
                padding=1, groups=in_c, bias=False),
            nn.BatchNorm3d(128),
            nn.Conv3d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.up = mn.UpSample(3, in_channels=128, out_channels=128, scale_factor=4, kernel_size=3)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        """Fuse detail-branch features ``x_d`` with spatial-branch features ``x_s``."""
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up(right)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    """Upsampling segmentation head producing per-voxel class logits."""

    def __init__(self, in_chan: int, num_classes: int, skip: int) -> None:
        super(SegmentHead, self).__init__()
        self.conv_out = nn.Conv3d(
            in_chan, num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.bn_act = mn.ADN(ordering="NDA", in_channels=in_chan, dropout_dim=1, dropout=0.1)
        self.up = mn.UpSample(3, in_channels=in_chan, out_channels=in_chan, scale_factor=2, kernel_size=3)
        self.skip = skip

    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        feat = x
        for i in range(3 - self.skip):
            feat = self.up(feat)
            feat = self.bn_act(feat)
        feat = self.conv_out(feat)
        return feat


class DualDomainNet(nn.Module):
    """Full dual-branch (detail + spatial) segmentation network with BiSeNet-style fusion."""

    def __init__(self, num_classes: int, in_c: int, skip: int = 0) -> None:
        """
        Args:
            num_classes (int): Number of output segmentation classes.
            in_c (int): Number of input channels.
            skip (int): How many trailing stages to skip in the detail/spatial branches (0 = full depth).
        """
        super(DualDomainNet, self).__init__()
        m_in = [128, 64, 32, 16]
        self.detail = ContentBranch(in_c, skip)
        if skip < 3:
            self.segment = SpatialBranch(in_c, skip)
            self.merge = MergeLayer(m_in[skip], skip)
        in_seg = 128 if skip < 3 else m_in[::-1][skip - 1]
        if (in_c > 128) & (skip == 3):
            in_seg = in_c
        self.head = SegmentHead(in_seg, num_classes, skip)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        feat_d = self.detail(x)
        if self.skip < 3:
            feat_s = self.segment(x)
            feat_head = self.merge(feat_d, feat_s)
        else:
            feat_head = feat_d
        logits = self.head(feat_head, size)
        return logits

if __name__ == "__main__":
    import monai
    model = monai.networks.nets.DynUNet(spatial_dims=3, in_channels=1, out_channels=2, kernel_size=[3, 3, 3, 3, 3], strides=[1,2,2,2,2], upsample_kernel_size=[2,2,2,2,1])
    print(model)
