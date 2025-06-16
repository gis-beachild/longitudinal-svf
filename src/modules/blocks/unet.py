# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# isort: dont-add-import: from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock
from src.modules.blocks.DualDomainNet import DualDomainNet
import monai


class DyNUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Sequence[int]] = None,
        strides: Sequence[Sequence[int]] = None,
        upsample_kernel_size: Sequence[Sequence[int]] = None,
        upsample_stride: Sequence[Sequence[int]] = None,
        filters: Sequence[int] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = False,
    ):
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        super().__init__()
        self.embedding_dim = 64
        self.temp_fm = 3
        self.num_heads = 4
        self.num_layers = 2
        self.drop_out = dropout
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_stride = upsample_stride
        if filters is None:
            self.filters = [min(2 ** (5 + i), 320) for i in range(len(self.strides))]
        self.input_block = self.conv_block(spatial_dims=3, in_channels=in_channels, out_channels=self.filters[0],
                                           stride=1, kernel_size=[3, 3, 3], norm_name=norm_name, act_name=act_name,
                                           dropout=self.drop_out)
        self.down_blocks = nn.ModuleList([
            self.conv_block(spatial_dims=3, in_channels=self.filters[i], out_channels=self.filters[i + 1],
                            stride=self.strides[i], kernel_size=self.kernel_size[i], norm_name=norm_name,
                            act_name=act_name, dropout=self.drop_out) for i in range(4)
        ])
        self.attention = [32, 64]
        layers = []
        for out_c in self.filters:
            if out_c in self.attention:
                if out_c < 128:
                    layer = DualDomainNet(out_c, out_c)
                elif out_c < 256:
                    layer = DualDomainNet(out_c, out_c, skip=1)
                elif out_c < 320:
                    layer = DualDomainNet(out_c, out_c, skip=2)
                else:
                    layer = DualDomainNet(out_c, out_c, skip=3)

            else:
                layer = nn.Identity()
            layers.append(layer)

        self.attention_layers = nn.ModuleList(layers)


        self.bottleneck = self.conv_block(spatial_dims=3, in_channels=self.filters[-1], out_channels=self.filters[-1], stride=self.strides[-1], kernel_size=self.kernel_size[-1], norm_name=norm_name, act_name=act_name, dropout=self.drop_out)

        self.up_block = nn.ModuleList([
            UnetUpBlock(spatial_dims=3, in_channels=self.filters[-1], out_channels=self.filters[-1],
                        stride=self.upsample_stride[-1], kernel_size=[1, 1, 1],
                        upsample_kernel_size=self.upsample_kernel_size[4], norm_name=norm_name, act_name=act_name,
                        dropout=self.drop_out)
        ])
        for i in range(len(self.filters) - 1, 0, -1):
            self.up_block.append(
                UnetUpBlock(spatial_dims=3, in_channels=self.filters[i], out_channels=self.filters[i - 1],
                            stride=self.upsample_stride[i - 1], kernel_size=[1, 1, 1],
                            upsample_kernel_size=self.upsample_kernel_size[i - 1], norm_name=norm_name,
                            act_name=act_name, dropout=self.drop_out))
        self.output_block = UnetOutBlock(spatial_dims=3,in_channels=self.filters[0],out_channels=out_channels,dropout=self.drop_out)



        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.ReLU(),
            nn.Linear( self.embedding_dim,  self.embedding_dim)
        )

        self.to_weight = nn.Linear( self.embedding_dim, 3)  # 3 vector channels
        self.sigmoid = nn.Sigmoid()

        self.channel_proj = nn.Linear(self.filters[self.temp_fm],  self.embedding_dim)

        self.apply(self.initialize_weights)

    def forward(self, images):
        skips = [self.input_block(images)]
        for i, down in enumerate(self.down_blocks):
            skips.append(down(skips[-1]))
        y = self.bottleneck(skips[-1])
        for i, up in enumerate(self.up_block):
            y = up(y, skips[-(i + 1)])
        y = self.output_block(y)
        return y


    def temp_weight(self, t, encoder_features):
        B, C_enc, D_enc, H_enc, W_enc = encoder_features.shape  # (B, 320, 8, 8, 8)
        x = encoder_features.flatten(2).permute(0, 2, 1)  # (B, 512, 320)
        x = self.channel_proj(x)  # (B, 512, embedding_dim)

        t_emb = self.time_encoder(t).unsqueeze(0).repeat(1, x.size(1), 1)  # (B, 512, embedding_dim)
        x = x + t_emb

        x = self.transformer(x)  # (B, 512, embedding_dim)

        weights = self.to_weight(x)  # (B, 512, 3)
        weights = self.sigmoid(weights).permute(0, 2, 1).view(B, 3, D_enc, H_enc, W_enc)  # (B, 3, 8, 8, 8)
        return weights  # (B, 3, 128, 128, 128)


    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)






class GlobalTemporalUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,

        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block: bool = False,
    ):
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        super().__init__()
        self.embedding_dim = 64
        self.temp_fm = 3
        self.num_heads = 4
        self.num_layers = 2
        self.drop_out = dropout
        self.kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        self.strides = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.upsample_kernel_size = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.upsample_stride = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.filters = [16, 32, 64, 128]
        self.input_block = self.conv_block(spatial_dims=3, in_channels=in_channels, out_channels=self.filters[0],
                                           stride=1, kernel_size=[3, 3, 3], norm_name=norm_name, act_name=act_name,
                                           dropout=self.drop_out)
        self.down_blocks = nn.ModuleList([
            self.conv_block(spatial_dims=3, in_channels=self.filters[i], out_channels=self.filters[i + 1],
                            stride=self.strides[i], kernel_size=self.kernel_size[i], norm_name=norm_name,
                            act_name=act_name, dropout=self.drop_out) for i in range(len(self.filters) - 1)
        ])

        self.bottleneck = self.conv_block(spatial_dims=3, in_channels=self.filters[-1], out_channels=self.filters[-1], stride=self.strides[-1], kernel_size=self.kernel_size[-1], norm_name=norm_name, act_name=act_name, dropout=self.drop_out)

        self.up_block = nn.ModuleList([
            UnetUpBlock(spatial_dims=3, in_channels=self.filters[-1], out_channels=self.filters[-1],
                        stride=self.upsample_stride[-1], kernel_size=[1, 1, 1],
                        upsample_kernel_size=self.upsample_kernel_size[-1], norm_name=norm_name, act_name=act_name,
                        dropout=self.drop_out)
        ])
        for i in range(len(self.filters) - 1, 0, -1):
            self.up_block.append(
                UnetUpBlock(spatial_dims=3, in_channels=self.filters[i], out_channels=self.filters[i - 1],
                            stride=self.upsample_stride[i - 1], kernel_size=[1, 1, 1],
                            upsample_kernel_size=self.upsample_kernel_size[i - 1], norm_name=norm_name,
                            act_name=act_name, dropout=self.drop_out))
        self.output_block = UnetOutBlock(spatial_dims=3,in_channels=self.filters[0],out_channels=out_channels,dropout=self.drop_out)
        self.attention = [16, 32]
        layers = []
        for out_c in self.filters:
            if out_c in self.attention:
                if out_c < 128:
                    layer = DualDomainNet(out_c, out_c)
                elif out_c < 256:
                    layer = DualDomainNet(out_c, out_c, skip=1)
                elif out_c < 320:
                    layer = DualDomainNet(out_c, out_c, skip=2)
                else:
                    layer = DualDomainNet(out_c, out_c, skip=3)

            else:
                layer = nn.Identity()
            layers.append(layer)

        self.attention_layers = nn.ModuleList(layers)



        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.ReLU(),
            nn.Linear( self.embedding_dim,  self.embedding_dim)
        )

        self.to_weight = nn.Linear( self.embedding_dim, 3)  # 3 vector channels
        self.sigmoid = nn.Sigmoid()

        self.channel_proj = nn.Linear(self.filters[self.temp_fm],  self.embedding_dim)
        self.dvf2ddf = monai.networks.blocks.DVF2DDF(num_steps=9, mode='bilinear', padding_mode='zeros')  # Vector integration based on Runge-Kutta method

        self.apply(self.initialize_weights)

    def forward(self, images):
        skips = [self.input_block(images)]
        if not isinstance(self.attention_layers[0], nn.Identity):
            skips[0] = self.attention_layers[0](skips[0]) + skips[0]
        for i in range(len(self.down_blocks)):
            skips.append(self.down_blocks[i](skips[-1]))
            if not isinstance(self.attention_layers[i+1], nn.Identity):
                skips[-1] = self.attention_layers[i+1](skips[-1]) + skips[-1]
        spatial_features = skips[self.temp_fm]
        y = self.bottleneck(skips[-1])
        for i, up in enumerate(self.up_block):
            y = up(y, skips[-(i + 1)])
        y = self.output_block(y)
        return y, spatial_features


    def temp_weight(self, t, encoder_features):
        B, C_enc, D_enc, H_enc, W_enc = encoder_features.shape  # (B, 320, 8, 8, 8)
        x = encoder_features.flatten(2).permute(0, 2, 1)  # (B, 512, 320)
        x = self.channel_proj(x)  # (B, 512, embedding_dim)

        t_emb = self.time_encoder(t).unsqueeze(0).repeat(1, x.size(1), 1)  # (B, 512, embedding_dim)
        x = x + t_emb

        x = self.transformer(x)  # (B, 512, embedding_dim)

        weights = self.to_weight(x)  # (B, 512, 3)
        weights = self.sigmoid(weights).permute(0, 2, 1).view(B, 3, D_enc, H_enc, W_enc)  # (B, 3, 8, 8, 8)
        return weights  # (B, 3, 128, 128, 128)


    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def velocity2displacement(self, dvf):
        '''
            Convert the velocity field to a flow field
            :param dvf: Velocity field
            :return: Deformation field
        '''
        return self.dvf2ddf(dvf)


if __name__ == "__main__":
    model = NewTest(in_channels=16, out_channels=3)
    x = torch.randn(1, 16, 128, 128, 128)
    output, spatial_features = model(x)
    print("Output shape:", output.shape)
    print("Spatial features shape:", spatial_features.shape)

    temp = VectorialTemporalAttention(embedding_dim=64, num_heads=4, num_layers=2)
    time_vec = torch.randn(1)  # Example time vector
    a = temp(spatial_features, time_vec)
    print("Temporal attention output shape:", a.shape)