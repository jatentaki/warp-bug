import torch, kornia, h5py, imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# This is a failure mode, with different aspect ratios
#fname1 = '47698078_3766965066'
#fname2 = '18698491_4586522698'

# This works
fname1 = '271147142_778c4e7999_o'
fname2 = '275191466_a33f8c30b7_o'

def get_K_Rt(K_: [3, 3], R: [3, 3], T: [3]):
    B = 1

    # add batch dimension
    K = K_[None]         # [B, 3, 3]
    R = R[None]          # [B, 3, 3]
    T = T[None, :, None] # [B, 3, 1]

    Rt_3x4 = kornia.projection_from_Rt(R, T)
    Rt = torch.zeros(B, 4, 4)
    Rt[:, :3, :] = Rt_3x4
    Rt[:,  3, 3] = 1.

    return K, Rt

def build_pinhole(K_3x3: ['B', 3, 3], Rt: ['B', 4, 4], shape):
    K_4x4 = torch.zeros(K_3x3.shape[0], 4, 4)
    K_4x4[:, :3, :3] = K_3x3
    K_4x4[:,  3,  3] = 1.

    width  = torch.tensor([shape[0]], dtype=torch.float32)
    height = torch.tensor([shape[1]], dtype=torch.float32)

    return kornia.PinholeCamera(K_4x4, Rt, width, height)

def read_data(fname):
    ''' read the files and return a
        * image tensor
        * depth tensor
        * kornia.PinholeCamera instance
    '''
    img = imageio.imread(f'data/images/{fname}.jpg')
    img = torch.from_numpy(img).to(torch.float32) / 255

    with h5py.File(f'data/depth_maps/{fname}.h5', 'r') as hdf:
        depth = torch.from_numpy(hdf['depth'][()]).to(torch.float32)

    with h5py.File(f'data/calibration/calibration_{fname}.h5', 'r') as hdf:
        K = torch.from_numpy(hdf['K'][()]).to(torch.float32)
        R = torch.from_numpy(hdf['R'][()]).to(torch.float32)
        T = torch.from_numpy(hdf['T'][()]).to(torch.float32)

    K, Rt = get_K_Rt(K, R, T)

    return img, depth, K, Rt

def rescale(img: ['H', 'W', 'C'], size) -> ['h', 'w', 'C']:
    _3d = img.permute(2, 0, 1)
    _4d = _3d[None, ...]
    scaled = F.interpolate(
        _4d,
        size=size,
        mode='bilinear',
        align_corners=False
    )
    return scaled.squeeze(0).permute(1, 2, 0)

def pad(img: ['H', 'W', 'C'], size, value=0.) -> ['h', 'w', 'C']:
    cropped = img[:size[0], :size[1]]
    # now bitmap cannot be bigger than target size

    xpad = size[0] - cropped.shape[0]
    ypad = size[1] - cropped.shape[1]

    # not that F.pad takes sizes starting from the last dimension
    padded = F.pad(cropped, (0, 0, 0, ypad, 0, xpad), mode='constant')

    assert padded.shape[:2] == size

    return padded

def scale_and_pad(img, depth, K, size):
    '''
    scale to at most `size` while preserving aspect ratio and then pad
    the remaining for `img` and `depth`. Adjust `K` accordingly
    '''

    x_factor = img.shape[0] / size[0]
    y_factor = img.shape[1] / size[1]

    f = 1 / max(x_factor, y_factor)
    if x_factor > y_factor:
        scale_size = (size[0], int(f * img.shape[1]))
    else:
        scale_size = (int(f * img.shape[0]), size[1])

    K     = kornia.geometry.epipolar.scale_intrinsics(K, f)
    img   = pad(rescale(img, scale_size), size)
    depth = pad(rescale(depth.unsqueeze(-1), scale_size), size).squeeze(-1)

    return img, depth, K

img1, dep1, K1, Rt1 = read_data(fname1)
img2, dep2, K2, Rt2 = read_data(fname2)

shape = 1024, 1024

img1, dep1, K1 = scale_and_pad(img1, dep1, K1, shape)
img2, dep2, K2 = scale_and_pad(img2, dep2, K2, shape)

pin1 = build_pinhole(K1, Rt1, img1.shape)
pin2 = build_pinhole(K2, Rt2, img2.shape)

img1_bchw = img1.permute(2, 0, 1)[None]
img2_bchw = img2.permute(2, 0, 1)[None]
dep1_bchw = dep1[None, None]
dep2_bchw = dep2[None, None]

warp_1_to_2 = kornia.geometry.warp.depth_warp(
    pin1, pin2, dep2_bchw, img1_bchw, *shape
)

warp_2_to_1 = kornia.geometry.warp.depth_warp(
    pin2, pin1, dep1_bchw, img2_bchw, *shape
)

fig, axes = plt.subplots(2, 3, constrained_layout=True)
axes[0, 0].imshow(img1.numpy())
axes[1, 0].imshow(img2.numpy())
axes[0, 1].imshow(dep1.numpy())
axes[1, 1].imshow(dep2.numpy())
axes[0, 2].imshow(warp_2_to_1[0].permute(1, 2, 0).numpy())
axes[1, 2].imshow(warp_1_to_2[0].permute(1, 2, 0).numpy())
plt.show()
