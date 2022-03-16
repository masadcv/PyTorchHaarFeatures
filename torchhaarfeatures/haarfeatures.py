import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import sparse


class HaarFeatures3d(nn.modules.Conv3d):
    def __init__(self, kernel_size, padding="same", stride=1, padding_mode="zeros"):
        # padding can be ["valid", "same"]
        haar_weights = self.initialise_haar_weights3d(kernel_size=kernel_size)

        in_channels = 1
        out_channels = haar_weights.shape[0]

        super(HaarFeatures3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode=padding_mode,
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

        # assert kernel_size[0]%2==1 and kernel_size[1]%2==1 and kernel_size[2]%2==1, "at the moment odd kernel sizes are supported, received {}".format(kernel_size)

        centerdim = tuple(math.ceil(x / 2) for x in kernel_size)
        kernel_size = tuple(kernel_size)

        # weights for 3d are of dimension Cout, Cin, D, H, W
        # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

        mean_volume = torch.ones(kernel_size) / (
            kernel_size[0] * kernel_size[1] * kernel_size[2]
        )

        full_volume = torch.ones(kernel_size)

        # sparse_volume = torch.from_numpy(np.array([float(x%2)*2-1 for x in range(full_volume.numel())], dtype=np.float32).reshape(kernel_size))
        sparse_volume = torch.ones(kernel_size)
        bit_write = True
        for k0 in range(kernel_size[0]):
            for k1 in range(kernel_size[1]):
                for k2 in range(kernel_size[2]):
                    sparse_volume[k0, k1, k2] = bit_write
                    bit_write = not bit_write

                if kernel_size[2] % 2 == 0:
                    bit_write = not bit_write

            if kernel_size[1] % 2 == 0:
                bit_write = not bit_write

        if kernel_size[0] % 2 == 0:
            bit_write = not bit_write

        sparse_volume = sparse_volume * 2 - 1
        assert sparse_volume.sum() == 0 or sparse_volume.sum(
        ) == 1, "sparse volume kernel not aligned"
        # print()
        # print(sparse_volume.sum())
        # print()

        half_volume_x = torch.ones(kernel_size)
        half_volume_x[:, :, centerdim[2]:] = -1

        half_volume_y = torch.ones(kernel_size)
        half_volume_y[:, centerdim[1]:, :] = -1

        half_volume_z = torch.ones(kernel_size)
        half_volume_z[centerdim[0]:, :, :] = -1

        half_volume_hh_hw_xy = torch.ones(kernel_size)
        half_volume_hh_hw_xy[:, : centerdim[1], : centerdim[2]] = 1
        half_volume_hh_hw_xy[:, : centerdim[1], centerdim[2]:] = -1
        half_volume_hh_hw_xy[:, centerdim[1]:, : centerdim[2]] = -1
        half_volume_hh_hw_xy[:, centerdim[1]:, centerdim[2]:] = 1

        half_volume_hd_hh_zy = torch.ones(kernel_size)
        half_volume_hd_hh_zy[: centerdim[0], : centerdim[1], :] = 1
        half_volume_hd_hh_zy[: centerdim[0], centerdim[1]:, :] = -1
        half_volume_hd_hh_zy[centerdim[0]:, : centerdim[1], :] = -1
        half_volume_hd_hh_zy[centerdim[0]:, centerdim[1]:, :] = 1

        half_volume_hd_hw_zx = torch.ones(kernel_size)
        half_volume_hd_hw_zx[: centerdim[0], :, : centerdim[2]] = 1
        half_volume_hd_hw_zx[: centerdim[0], :, centerdim[2]:] = -1
        half_volume_hd_hw_zx[centerdim[0]:, :, : centerdim[2]] = -1
        half_volume_hd_hw_zx[centerdim[0]:, :, centerdim[2]:] = 1

        quarter_volume_zxy = torch.ones(kernel_size)
        quarter_volume_zxy[: centerdim[0], : centerdim[1], : centerdim[2]] = 1
        quarter_volume_zxy[: centerdim[0], : centerdim[1], centerdim[2]:] = -1
        quarter_volume_zxy[: centerdim[0], centerdim[1]:, : centerdim[2]] = -1
        quarter_volume_zxy[: centerdim[0], centerdim[1]:, centerdim[2]:] = 1

        quarter_volume_zxy[centerdim[0]:, : centerdim[1], : centerdim[2]] = -1
        quarter_volume_zxy[centerdim[0]:, : centerdim[1], centerdim[2]:] = 1
        quarter_volume_zxy[centerdim[0]:, centerdim[1]:, : centerdim[2]] = 1
        quarter_volume_zxy[centerdim[0]:, centerdim[1]:, centerdim[2]:] = -1

        half_quarter_volume1_x = quarter_volume_zxy.detach().clone()
        half_quarter_volume1_x[:, :, :centerdim[2]] = 1

        half_quarter_volume2_x = quarter_volume_zxy.detach().clone()
        half_quarter_volume2_x[:, :, centerdim[2]:] = 1

        half_quarter_volume1_y = quarter_volume_zxy.detach().clone()
        half_quarter_volume1_y[:, :centerdim[1], :] = 1

        half_quarter_volume2_y = quarter_volume_zxy.detach().clone()
        half_quarter_volume2_y[:, centerdim[1]:, :] = 1

        half_quarter_volume1_z = quarter_volume_zxy.detach().clone()
        half_quarter_volume1_z[:centerdim[0], :, :] = 1

        half_quarter_volume2_z = quarter_volume_zxy.detach().clone()
        half_quarter_volume2_z[centerdim[0]:, :, :] = 1

        mean_volume.unsqueeze_(dim=0)
        full_volume.unsqueeze_(dim=0)
        sparse_volume.unsqueeze_(dim=0)

        half_volume_x.unsqueeze_(dim=0)
        half_volume_y.unsqueeze_(dim=0)
        half_volume_z.unsqueeze_(dim=0)

        half_volume_hh_hw_xy.unsqueeze_(dim=0)
        half_volume_hd_hh_zy.unsqueeze_(dim=0)
        half_volume_hd_hw_zx.unsqueeze_(dim=0)

        quarter_volume_zxy.unsqueeze_(dim=0)

        half_quarter_volume1_x.unsqueeze_(dim=0)
        half_quarter_volume2_x.unsqueeze_(dim=0)

        half_quarter_volume1_y.unsqueeze_(dim=0)
        half_quarter_volume2_y.unsqueeze_(dim=0)

        half_quarter_volume1_z.unsqueeze_(dim=0)
        half_quarter_volume2_z.unsqueeze_(dim=0)

        all_features = torch.cat(
            [
                mean_volume,
                full_volume,
                sparse_volume,
                half_volume_x,
                half_volume_y,
                half_volume_z,
                half_volume_hh_hw_xy,
                half_volume_hd_hh_zy,
                half_volume_hd_hw_zx,
                quarter_volume_zxy,
                half_quarter_volume1_x,
                half_quarter_volume2_x,
                half_quarter_volume1_y,
                half_quarter_volume2_y,
                half_quarter_volume1_z,
                half_quarter_volume2_z,
            ],
            dim=0,
        )

        all_features.unsqueeze_(dim=1)

        return all_features


class HaarFeatures2d(nn.modules.Conv2d):
    def __init__(self, kernel_size, padding="same", stride=1, padding_mode="zeros"):
        # padding can be ["valid", "same"]
        haar_weights = self.initialise_haar_weights2d(kernel_size=kernel_size)

        in_channels = 1
        out_channels = haar_weights.shape[0]

        super(HaarFeatures2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode=padding_mode,
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

        # assert kernel_size[0]%2==1 and kernel_size[1]%2==1, "at the moment odd kernel sizes are supported, received {}".format(kernel_size)
        centerdim = tuple(math.ceil(x / 2) for x in kernel_size)
        onethirddim = tuple(math.ceil(x / 3) for x in kernel_size)
        kernel_size = tuple(kernel_size)

        # weights for 2d are of dimension Cout, Cin, H, W
        # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

        mean_area = torch.ones(kernel_size) / \
            (kernel_size[0] * kernel_size[1])

        full_area = torch.ones(kernel_size)

        # sparse_area = torch.from_numpy(np.array([float(x%2)*2-1 for x in range(full_area.numel())], dtype=np.float32).reshape(kernel_size))
        sparse_area = torch.ones(kernel_size)
        bit_write = True
        for k0 in range(kernel_size[0]):
            for k1 in range(kernel_size[1]):
                sparse_area[k0, k1] = bit_write
                bit_write = not bit_write

            if kernel_size[1] % 2 == 0:
                bit_write = not bit_write

        if kernel_size[0] % 2 == 0:
            bit_write = not bit_write

        sparse_area = sparse_area * 2 - 1
        assert sparse_area.sum() == 0 or sparse_area.sum(
        ) == 1, "sparse area kernel not aligned"
        # print()
        # print(sparse_area.sum())
        # print()

        half_area_x = torch.ones(kernel_size)
        half_area_x[:, centerdim[1]:] = -1

        half_area_y = torch.ones(kernel_size)
        half_area_y[centerdim[0]:, :] = -1

        third_area_x = torch.ones(kernel_size)
        # goes negative from 1/3 to 2/3 of the way
        third_area_x[:, onethirddim[1]: 2 * onethirddim[1]] = -1

        third_area_y = torch.ones(kernel_size)
        # goes negative from 1/3 to 2/3 of the way
        third_area_y[onethirddim[0]: 2 * onethirddim[0], :] = -1

        quarter_area_xy = torch.ones(kernel_size)
        # goes negative for second and third square
        quarter_area_xy[: centerdim[0], centerdim[1]:] = -1
        quarter_area_xy[centerdim[0]:, : centerdim[1]] = -1

        half_quarter_area1_x = quarter_area_xy.detach().clone()
        half_quarter_area1_x[:, :centerdim[1]] = 1

        half_quarter_area2_x = quarter_area_xy.detach().clone()
        half_quarter_area2_x[:, centerdim[1]:] = 1

        half_quarter_area1_y = quarter_area_xy.detach().clone()
        half_quarter_area1_y[:centerdim[0], :] = 1

        half_quarter_area2_y = quarter_area_xy.detach().clone()
        half_quarter_area2_y[centerdim[0]:, :] = 1

        mean_area.unsqueeze_(dim=0)
        full_area.unsqueeze_(dim=0)
        sparse_area.unsqueeze_(dim=0)

        half_area_x.unsqueeze_(dim=0)
        half_area_y.unsqueeze_(dim=0)

        third_area_x.unsqueeze_(dim=0)
        third_area_y.unsqueeze_(dim=0)

        quarter_area_xy.unsqueeze_(dim=0)

        half_quarter_area1_x.unsqueeze_(dim=0)
        half_quarter_area2_x.unsqueeze_(dim=0)

        half_quarter_area1_y.unsqueeze_(dim=0)
        half_quarter_area2_y.unsqueeze_(dim=0)

        all_features = torch.cat(
            [
                mean_area,
                full_area,
                sparse_area,
                half_area_x,
                half_area_y,
                third_area_x,
                third_area_y,
                quarter_area_xy,
                half_quarter_area1_x,
                half_quarter_area2_x,
                half_quarter_area1_y,
                half_quarter_area2_y,
            ]
        )

        all_features.unsqueeze_(dim=1)

        return all_features


if __name__ == "__main__":
    input2d = torch.rand(size=(1, 1, 128, 128))
    input3d = torch.rand(size=(1, 1, 128, 128, 128))

    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        haarfeat2d = HaarFeatures2d(
            kernel_size=(i, i), padding="same", stride=1)
        output_haar2d = haarfeat2d(input2d)
        assert output_haar2d.shape[2:] == input2d.shape[2:]

        haarfeat2d = HaarFeatures2d(
            kernel_size=i, padding="same", stride=1)

        haarfeat3d = HaarFeatures3d(kernel_size=(
            i, i, i), padding="same", stride=1)
        output_haar3d = haarfeat3d(input3d)
        assert output_haar3d.shape[2:] == input3d.shape[2:]

        haarfeat3d = HaarFeatures3d(kernel_size=i, padding="same", stride=1)
