from typing import List
from numpy.core.numeric import full
import torch
import numpy as np
import math


def check_integral(orignal, integral):
    d, h, w = orignal.shape

    seld = np.random.randint(0, d)
    selh = np.random.randint(0, h)
    selw = np.random.randint(0, w)

    selorignal = orignal[:seld+1, :selh+1, :selw+1]

    assert selorignal.sum() == integral[seld, selh, selw], "not equal"


def get_integral3d(orignal):
    assert len(
        orignal.shape) == 3, 'get_integral3d() only works with 3D volume data'
    return orignal.cumsum(0).cumsum(1).cumsum(2)


def get_integral_at(integral, index):
    d, h, w = integral.shape

    if index[0] < 0 or index[1] < 0 or index[2] < 0:
        return 0

    if index[0] >= d or index[1] >= h or index[2] >= w:
        return 0

    return integral[index]


def get_cuboid_area(integral, topleftback, bottomrightfront):
    # z, y, x
    A = tuple(bottomrightfront)
    B = tuple([topleftback[0], bottomrightfront[1], topleftback[2]])
    C = tuple([B[0], B[1], A[2]])
    D = tuple([bottomrightfront[0], B[1], B[2]])

    return get_integral_at(integral, A) + get_integral_at(integral, B) - get_integral_at(integral, C) - get_integral_at(integral, D)


def get_haar_features(integral, center, windowsize):
    assert len(center) == 3, 'works on 3D spatial data only'
    assert len(windowsize) == 3, 'works on 3D spatial data only'

    # assert kdepth%2 == 0 and kheight%2 == 0 and kwidth%2 == 0, 'only even window size supported for now'

    # returns haar-like features for a window centered at cx, cy, cz with size(kwidth, kheight, kdepth)
    topleftback = [int(c-kw/2) for c, kw in zip(center, windowsize)]
    bottomrightfront = [int(c+kw/2) for c, kw in zip(center, windowsize)]

    # full volume
    full_volume = get_cuboid_area(integral, topleftback, bottomrightfront)

    # half volume x
    bottomrightfront_x = [bottomrightfront[0], bottomrightfront[1], center[2]]
    half_volume_plus_x = get_cuboid_area(
        integral, topleftback, bottomrightfront_x)
    topleftback_x = [topleftback[0], topleftback[1], center[2]]
    half_volume_minus_x = get_cuboid_area(
        integral, topleftback_x, bottomrightfront)
    half_volume_x = half_volume_plus_x - half_volume_minus_x

    # half volume y
    bottomrightfront_y = [bottomrightfront[0], center[1], bottomrightfront[2]]
    half_volume_plus_y = get_cuboid_area(
        integral, topleftback, bottomrightfront_y)
    topleftback_y = [topleftback[0], center[1], topleftback[2]]
    half_volume_minus_y = get_cuboid_area(
        integral, topleftback_y, bottomrightfront)
    half_volume_y = - half_volume_plus_y + half_volume_minus_y

    # half volume z
    bottomrightfront_z = [center[0], bottomrightfront[1], bottomrightfront[2]]
    half_volume_plus_z = get_cuboid_area(
        integral, topleftback, bottomrightfront_z)
    topleftback_z = [center[0], topleftback[1], topleftback[2]]
    half_volume_minus_z = get_cuboid_area(
        integral, topleftback_z, bottomrightfront)
    half_volume_z = -half_volume_plus_z + half_volume_minus_z

    # cuboid half width/height xy
    bottomrightfront_hh_hw_xy = [bottomrightfront[0], center[1], center[2]]
    half_volume_box1_xy = get_cuboid_area(
        integral, topleftback, bottomrightfront_hh_hw_xy)
    half_volume_box2_xy = get_cuboid_area(
        integral, topleftback_x, bottomrightfront_y)
    half_volume_box3_xy = get_cuboid_area(
        integral, topleftback_y, bottomrightfront_x)
    topleftback_hh_hw_xy = [topleftback[0], center[1], center[2]]
    half_volume_box4_xy = get_cuboid_area(
        integral, topleftback_hh_hw_xy, bottomrightfront)
    half_volume_hh_hw_xy = -half_volume_box1_xy + \
        half_volume_box2_xy + half_volume_box3_xy - half_volume_box4_xy

    # cuboid half width/height zy
    bottomrightfront_hd_hh_zy = [center[0], center[1], bottomrightfront[2]]
    half_volume_box1_zy = get_cuboid_area(
        integral, topleftback, bottomrightfront_hd_hh_zy)
    half_volume_box2_zy = get_cuboid_area(
        integral, topleftback_z, bottomrightfront_y)
    half_volume_box3_zy = get_cuboid_area(
        integral, topleftback_y, bottomrightfront_z)
    topleftback_hd_hh_zy = [center[0], center[1], topleftback[2]]
    half_volume_box4_zy = get_cuboid_area(
        integral, topleftback_hd_hh_zy, bottomrightfront)
    half_volume_hd_hh_zy = +half_volume_box1_zy - \
        half_volume_box2_zy - half_volume_box3_zy + half_volume_box4_zy

    # cuboid half width/height zx
    bottomrightfront_hd_hw_zx = [center[0], bottomrightfront[1], center[2]]
    half_volume_box1_zx = get_cuboid_area(
        integral, topleftback, bottomrightfront_hd_hw_zx)
    half_volume_box3_zx = get_cuboid_area(
        integral, topleftback_x, bottomrightfront_z)
    half_volume_box2_zx = get_cuboid_area(
        integral, topleftback_z, bottomrightfront_x)
    topleftback_hd_hw_zx = [center[0], topleftback[1], center[2]]
    half_volume_box4_zx = get_cuboid_area(
        integral, topleftback_hd_hw_zx, bottomrightfront)
    half_volume_hd_hw_zx = -half_volume_box1_zx + \
        half_volume_box2_zx + half_volume_box3_zx - half_volume_box4_zx

    # 8 squares zxy
    bottomrightfront_hd_hh_hw_zxy = [center[0], center[1], center[2]]
    quarter_volume_box1_zxy = get_cuboid_area(
        integral, topleftback, bottomrightfront_hd_hh_hw_zxy)
    quarter_volume_box2_zxy = get_cuboid_area(
        integral, topleftback_x, bottomrightfront_hd_hh_zy)
    quarter_volume_box3_zxy = get_cuboid_area(
        integral, topleftback_z, bottomrightfront_hh_hw_xy)
    quarter_volume_box4_zxy = get_cuboid_area(
        integral, topleftback_hd_hw_zx, bottomrightfront_hh_hw_xy)

    quarter_volume_box5_zxy = get_cuboid_area(
        integral, topleftback_y, bottomrightfront_hd_hw_zx)
    quarter_volume_box6_zxy = get_cuboid_area(
        integral, topleftback_hh_hw_xy, bottomrightfront_z)
    quarter_volume_box7_zxy = get_cuboid_area(
        integral, topleftback_hd_hh_zy, bottomrightfront_x)
    quarter_volume_box8_zxy = get_cuboid_area(
        integral, bottomrightfront_hd_hh_hw_zxy, bottomrightfront)

    quarter_volume_zxy = + quarter_volume_box1_zxy - quarter_volume_box2_zxy - quarter_volume_box3_zxy + quarter_volume_box4_zxy + \
        - quarter_volume_box5_zxy + quarter_volume_box6_zxy + \
        quarter_volume_box7_zxy - quarter_volume_box8_zxy

    solid_quarter_zy = half_volume_plus_x - quarter_volume_box2_zxy + \
        quarter_volume_box4_zxy + quarter_volume_box6_zxy - quarter_volume_box8_zxy
    solid_quarter_zx = half_volume_minus_y + quarter_volume_box1_zxy - \
        quarter_volume_box2_zxy - quarter_volume_box3_zxy + quarter_volume_box4_zxy
    solid_quarter_xy = half_volume_minus_z + quarter_volume_box1_zxy - \
        quarter_volume_box2_zxy - quarter_volume_box5_zxy + quarter_volume_box6_zxy

    return [full_volume, half_volume_x, half_volume_y, half_volume_z,
            half_volume_hh_hw_xy, half_volume_hd_hh_zy, half_volume_hd_hw_zx,
            quarter_volume_zxy,
            solid_quarter_zy, solid_quarter_zx, solid_quarter_xy]


def initialise_haar_weights3d(windowsize):

    if isinstance(windowsize, List):
        assert len(windowsize) == 3, "window size must be 3d"
    else:
        windowsize = [windowsize, windowsize, windowsize]

    centerdim = tuple(math.ceil(x/2) for x in windowsize)
    windowsize = tuple(windowsize)

    # weights for 3d are of dimension Cout, Cin, D, H, W
    # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

    mean_volume = np.ones(windowsize) / \
        (windowsize[0] * windowsize[1] * windowsize[2])

    full_volume = np.ones(windowsize)

    half_volume_x = np.ones(windowsize)
    half_volume_x[:, :, centerdim[2]:] = -1
    print(half_volume_x)

    half_volume_y = np.ones(windowsize)
    half_volume_y[:, centerdim[1]:, :] = -1
    print(half_volume_y)

    half_volume_z = np.ones(windowsize)
    half_volume_z[centerdim[0]:, :, :] = -1
    print(half_volume_z)

    half_volume_hh_hw_xy = np.ones(windowsize)
    half_volume_hh_hw_xy[:, :centerdim[1], :centerdim[2]] = 1
    half_volume_hh_hw_xy[:, :centerdim[1], centerdim[2]:] = -1
    half_volume_hh_hw_xy[:, centerdim[1]:, :centerdim[2]] = -1
    half_volume_hh_hw_xy[:, centerdim[1]:, centerdim[2]:] = 1

    half_volume_hd_hh_zy = np.ones(windowsize)
    half_volume_hd_hh_zy[:centerdim[0], :centerdim[1], :] = 1
    half_volume_hd_hh_zy[:centerdim[0], centerdim[1]:, :] = -1
    half_volume_hd_hh_zy[centerdim[0]:, :centerdim[1], :] = -1
    half_volume_hd_hh_zy[centerdim[0]:, centerdim[1]:, :] = 1

    half_volume_hd_hw_zx = np.ones(windowsize)
    half_volume_hd_hw_zx[:centerdim[0], :, :centerdim[2]] = 1
    half_volume_hd_hw_zx[:centerdim[0], :, centerdim[2]:] = -1
    half_volume_hd_hw_zx[centerdim[0]:, :, :centerdim[2]] = -1
    half_volume_hd_hw_zx[centerdim[0]:, :, centerdim[2]:] = 1

    quarter_volume_zxy = np.ones(windowsize)
    quarter_volume_zxy[:centerdim[0], :centerdim[1], :centerdim[2]] = 1
    quarter_volume_zxy[:centerdim[0], :centerdim[1], centerdim[2]:] = -1
    quarter_volume_zxy[:centerdim[0], centerdim[1]:, :centerdim[2]] = -1
    quarter_volume_zxy[:centerdim[0], centerdim[1]:, centerdim[2]:] = 1

    quarter_volume_zxy[centerdim[0]:, :centerdim[1], :centerdim[2]] = -1
    quarter_volume_zxy[centerdim[0]:, :centerdim[1], centerdim[2]:] = 1
    quarter_volume_zxy[centerdim[0]:, centerdim[1]:, :centerdim[2]] = 1
    quarter_volume_zxy[centerdim[0]:, centerdim[1]:, centerdim[2]:] = -1

    all_features = np.array([mean_volume, full_volume, half_volume_x, half_volume_y, half_volume_z,
                            half_volume_hh_hw_xy, half_volume_hd_hh_zy, half_volume_hd_hw_zx, quarter_volume_zxy])

    all_features = np.expand_dims(all_features, axis=1)

    print(all_features.shape)
    return all_features


def initialise_haar_weights2d(kernel_size):

    if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "window size must be 2d"
    else:
        kernel_size = [kernel_size, kernel_size]

    centerdim = tuple(math.ceil(x/2) for x in kernel_size)
    onethirddim = tuple(math.ceil(x/3) for x in kernel_size)
    kernel_size = tuple(kernel_size)

    # weights for 2d are of dimension Cout, Cin, H, W
    # Cin is always 1 for us, Cout is the dimension to concatenate all haar filters

    mean_volume = np.ones(kernel_size) / \
        (kernel_size[0] * kernel_size[1])

    full_volume = np.ones(kernel_size)

    two_vertical = np.ones(kernel_size)
    two_vertical[:, centerdim[1]:] = -1

    two_horizontal = np.ones(kernel_size)
    two_horizontal[centerdim[0]:, :] = -1

    three_vertical = np.ones(kernel_size)
    # goes negative from 1/3 to 2/3 of the way
    three_vertical[:, onethirddim[1]:2*onethirddim[1]] = -1

    three_horizontal = np.ones(kernel_size)
    # goes negative from 1/3 to 2/3 of the way
    three_horizontal[onethirddim[0]:2*onethirddim[0], :] = -1

    four_horizontal_vertical = np.ones(kernel_size)
    # goes negative for second and third square
    four_horizontal_vertical[:centerdim[0], centerdim[1]:] = -1
    four_horizontal_vertical[centerdim[0]:, :centerdim[1]] = -1

    all_features = np.array([mean_volume, full_volume, two_vertical, two_horizontal,
                            three_vertical, three_horizontal, four_horizontal_vertical])

    all_features = np.expand_dims(all_features, axis=1)

    return all_features


if __name__ == "__main__":
    sl = 128
    myarray = np.array([x for x in range(sl*sl*sl)]).reshape((sl, sl, sl))
    print(myarray)

    integral = get_integral3d(myarray)
    print(integral)

    check_integral(myarray, integral)

    initialise_haar_weights3d(9)
    aabc = initialise_haar_weights2d(9)
    print(aabc)


    # for d in range(sl):
    #     for h in range(sl):
    #         for w in range(sl):

    #             get_haar_features(integral, [d, h, w], [9, 9, 9])
