import torch
import torch.nn as nn

class myResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_features=32, mod='long_seg'):
        super(myResUNet3D, self).__init__()

        self.mod = mod
        self.encoder1 = self.res_block(in_channels, base_features)
        self.encoder2 = self.res_block(base_features, base_features * 2)
        self.encoder3 = self.res_block(base_features * 2, base_features * 4)
        self.encoder4 = self.res_block(base_features * 4, base_features * 8)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.bottleneck = self.res_block(base_features * 8, base_features * 16)

        self.upconv4 = nn.ConvTranspose3d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.decoder4 = self.res_block(base_features * 16, base_features * 8)
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self.res_block(base_features * 8, base_features * 4)
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.res_block(base_features * 4, base_features * 2)
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.decoder1 = self.res_block(base_features * 2, base_features)

        self.conv_final = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def res_block(self, in_channels, out_channels):
        return nn.Sequential(
            ResUnit(in_channels, out_channels),
            ResUnit(out_channels, out_channels)
        )

    def forward(self, x):
        # print(x.shape)
        
        if self.mod == 'single_seg':
            # 对两个timpoint的数据分别编码
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            e3 = self.encoder3(self.pool(e2))
            e4 = self.encoder4(self.pool(e3))

        elif self.mod == 'long_seg':
            # 输入x的维度为(2, 1, D, H, W)，2个timpoint，需拆分
            x1, x2 = x[0, :, :, :, :].unsqueeze(1), x[1, :, :, :, :].unsqueeze(1)
            
            # 对两个timpoint的数据分别编码
            e1_1 = self.encoder1(x1)
            e1_2 = self.encoder1(x2)
            e2_1 = self.encoder2(self.pool(e1_1))
            e2_2 = self.encoder2(self.pool(e1_2))
            e3_1 = self.encoder3(self.pool(e2_1))
            e3_2 = self.encoder3(self.pool(e2_2))
            e4_1 = self.encoder4(self.pool(e3_1))
            e4_2 = self.encoder4(self.pool(e3_2))

            # 特征相加
            e1 = e1_1 + e1_2
            e2 = e2_1 + e2_2
            e3 = e3_1 + e3_2
            e4 = e4_1 + e4_2

        else:
            raise ValueError("mod should be 'single_seg' or 'long_seg'")

        # bottleneck层
        bottleneck = self.bottleneck(self.pool(e4))
        # print(f"bottleneck shape: {bottleneck.shape}")

        # 解码器 skip connection
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((d4, e4), dim=1) ## FIXME
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        # 仅预测第二个时间节点的mask， 得到logits
        out = self.conv_final(d1)
        
        return out

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.conv1 =  nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True, momentum=0.1, track_running_stats=True)
        # self.norm1 = nn.BatchNorm3d(out_channels, affine=True, momentum=0.1, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True, momentum=0.1, track_running_stats=True)
        # self.norm2 = nn.BatchNorm3d(out_channels, affine=True, momentum=0.1, track_running_stats=True)
        
        if in_channels != out_channels:
            self.res_block = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.res_block = None
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        ## residual connection
        if self.res_block:
            residual = self.res_block(residual)
        out += residual
        out = self.relu(out)
        return out
