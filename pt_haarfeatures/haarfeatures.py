import math
import torch
import numpy as np
import torch.nn as nn

class HaarFeatures3d(nn.modules.Conv3d):
    def __init__(self, kernel_size, padding=None, stride=1, padding_mode='zeros'):
        haar_weights = self.initialise_haar_weights3d(kernel_size=kernel_size)

        in_channels = 1
        out_channels = haar_weights.shape[0]
        if not padding:
            padding = int(math.floor(haar_weights.shape[-1]/2))

        super(HaarFeatures3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode=padding_mode
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = haar_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)

    def initialise_haar_weights3d(self, kernel_size):

        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            assert len(kernel_size) == 3, "window size must be 3d"
        else:
            kernel_size = [kernel_size, kernel_size, kernel_size]

        centerdim = tuple(math.ceil(x/2) for x in kernel_size)
        kernel_size = tuple(kernel_size)

        # weights for 3d are of dimension Cout, Cin, D, H, W
        # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

        mean_volume = torch.ones(kernel_size) / \
            (kernel_size[0] * kernel_size[1] * kernel_size[2])

        full_volume = torch.ones(kernel_size)

        half_volume_x = torch.ones(kernel_size)
        half_volume_x[:, :, centerdim[2]:] = -1

        half_volume_y = torch.ones(kernel_size)
        half_volume_y[:, centerdim[1]:, :] = -1

        half_volume_z = torch.ones(kernel_size)
        half_volume_z[centerdim[0]:, :, :] = -1

        half_volume_hh_hw_xy = torch.ones(kernel_size)
        half_volume_hh_hw_xy[:, :centerdim[1], :centerdim[2]] = 1
        half_volume_hh_hw_xy[:, :centerdim[1], centerdim[2]:] = -1
        half_volume_hh_hw_xy[:, centerdim[1]:, :centerdim[2]] = -1
        half_volume_hh_hw_xy[:, centerdim[1]:, centerdim[2]:] = 1

        half_volume_hd_hh_zy = torch.ones(kernel_size)
        half_volume_hd_hh_zy[:centerdim[0], :centerdim[1], :] = 1
        half_volume_hd_hh_zy[:centerdim[0], centerdim[1]:, :] = -1
        half_volume_hd_hh_zy[centerdim[0]:, :centerdim[1], :] = -1
        half_volume_hd_hh_zy[centerdim[0]:, centerdim[1]:, :] = 1

        half_volume_hd_hw_zx = torch.ones(kernel_size)
        half_volume_hd_hw_zx[:centerdim[0], :, :centerdim[2]] = 1
        half_volume_hd_hw_zx[:centerdim[0], :, centerdim[2]:] = -1
        half_volume_hd_hw_zx[centerdim[0]:, :, :centerdim[2]] = -1
        half_volume_hd_hw_zx[centerdim[0]:, :, centerdim[2]:] = 1

        quarter_volume_zxy = torch.ones(kernel_size)
        quarter_volume_zxy[:centerdim[0], :centerdim[1], :centerdim[2]] = 1
        quarter_volume_zxy[:centerdim[0], :centerdim[1], centerdim[2]:] = -1
        quarter_volume_zxy[:centerdim[0], centerdim[1]:, :centerdim[2]] = -1
        quarter_volume_zxy[:centerdim[0], centerdim[1]:, centerdim[2]:] = 1

        quarter_volume_zxy[centerdim[0]:, :centerdim[1], :centerdim[2]] = -1
        quarter_volume_zxy[centerdim[0]:, :centerdim[1], centerdim[2]:] = 1
        quarter_volume_zxy[centerdim[0]:, centerdim[1]:, :centerdim[2]] = 1
        quarter_volume_zxy[centerdim[0]:, centerdim[1]:, centerdim[2]:] = -1

        mean_volume.unsqueeze_(dim=0)
        full_volume.unsqueeze_(dim=0)
        half_volume_x.unsqueeze_(dim=0)
        half_volume_y.unsqueeze_(dim=0)
        half_volume_z.unsqueeze_(dim=0)

        half_volume_hh_hw_xy.unsqueeze_(dim=0)
        half_volume_hd_hh_zy.unsqueeze_(dim=0)
        half_volume_hd_hw_zx.unsqueeze_(dim=0)

        quarter_volume_zxy.unsqueeze_(dim=0)

        all_features = torch.cat([mean_volume, full_volume, half_volume_x, half_volume_y, half_volume_z,
                                half_volume_hh_hw_xy, half_volume_hd_hh_zy, half_volume_hd_hw_zx, quarter_volume_zxy], dim=0)

        all_features.unsqueeze_(dim=1)

        return all_features

class HaarFeatures2d(nn.modules.Conv2d):
    def __init__(self, kernel_size, padding=None, stride=1, padding_mode='zeros'):
        haar_weights = self.initialise_haar_weights2d(kernel_size=kernel_size)

        in_channels = 1
        out_channels = haar_weights.shape[0]
        if not padding:
            padding = int(math.floor(haar_weights.shape[-1]/2))

        super(HaarFeatures2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode=padding_mode
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = haar_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)
    
    def initialise_haar_weights2d(self, kernel_size):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "window size must be 2d"
        else:
            kernel_size = [kernel_size, kernel_size]

        centerdim = tuple(math.ceil(x/2) for x in kernel_size)
        onethirddim = tuple(math.ceil(x/3) for x in kernel_size)
        kernel_size = tuple(kernel_size)

        # weights for 2d are of dimension Cout, Cin, H, W
        # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

        mean_volume = torch.ones(kernel_size) / \
            (kernel_size[0] * kernel_size[1])

        full_volume = torch.ones(kernel_size)

        two_vertical = torch.ones(kernel_size)
        two_vertical[:, centerdim[1]:] = -1

        two_horizontal = torch.ones(kernel_size)
        two_horizontal[centerdim[0]:, :] = -1

        three_vertical = torch.ones(kernel_size)
        # goes negative from 1/3 to 2/3 of the way
        three_vertical[:, onethirddim[1]:2*onethirddim[1]] = -1

        three_horizontal = torch.ones(kernel_size)
        # goes negative from 1/3 to 2/3 of the way
        three_horizontal[onethirddim[0]:2*onethirddim[0], :] = -1

        four_horizontal_vertical = torch.ones(kernel_size)
        # goes negative for second and third square
        four_horizontal_vertical[:centerdim[0], centerdim[1]:] = -1
        four_horizontal_vertical[centerdim[0]:, :centerdim[1]] = -1

        mean_volume.unsqueeze_(dim=0)
        full_volume.unsqueeze_(dim=0)
        two_vertical.unsqueeze_(dim=0)
        two_horizontal.unsqueeze_(dim=0)
        three_vertical.unsqueeze_(dim=0)
        three_horizontal.unsqueeze_(dim=0)
        four_horizontal_vertical.unsqueeze_(dim=0)

        all_features = torch.cat([mean_volume, full_volume, two_vertical, two_horizontal,
                                three_vertical, three_horizontal, four_horizontal_vertical])

        all_features.unsqueeze_(dim=1)

        return all_features

if __name__ == "__main__":
    conv3d = nn.Conv3d(in_channels=1, out_channels=8,
                    kernel_size=9, stride=1, padding=4)
    output_orig = conv3d(torch.zeros(size=(1, 1, 128, 128, 128)))
    print(output_orig.shape)

    haarfeat3d = HaarFeatures3d(kernel_size=(9, 9, 9), stride=1)
    output_haar3d = haarfeat3d(torch.rand(size=(1, 1, 128, 128, 128)))

    print(output_haar3d.shape)

    haarfeat2d = HaarFeatures2d(kernel_size=(9, 9), stride=1)
    output_haar2d = haarfeat2d(torch.rand(size=(1, 1, 128, 128)))
    print(output_haar2d.shape)