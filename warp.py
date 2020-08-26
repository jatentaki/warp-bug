import torch, kornia, h5py, imageio
import numpy as np

prefix = 'data'
fname1 = '271147142_778c4e7999_o'
fname2 = '275191466_a33f8c30b7_o'
#fname1 = '47698078_3766965066'
#fname2 = '18698491_4586522698'

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

def read_data(fname):
    ''' read the files and return a
        * image tensor
        * depth tensor
        * kornia.PinholeCamera instance
    '''
    img = imageio.imread(f'{prefix}/images/{fname}.jpg')
    img = torch.from_numpy(img).to(torch.float32) / 255

    with h5py.File(f'{prefix}/depth_maps/{fname}.h5', 'r') as hdf:
        depth = torch.from_numpy(hdf['depth'][()]).to(torch.float32)

    with h5py.File(f'{prefix}/calibration/calibration_{fname}.h5', 'r') as hdf:
        K = torch.from_numpy(hdf['K'][()]).to(torch.float32)
        R = torch.from_numpy(hdf['R'][()]).to(torch.float32)
        T = torch.from_numpy(hdf['T'][()]).to(torch.float32)

    K, Rt = get_K_Rt(K, R, T)

    return img, depth, K, Rt

img1, dep1, K1, Rt1 = read_data(fname1)
img2, dep2, K2, Rt2 = read_data(fname2)

img1_bchw = img1.permute(2, 0, 1)[None]
img2_bchw = img2.permute(2, 0, 1)[None]

trans_12 = Rt2 @ torch.inverse(Rt1)
trans_21 = Rt1 @ torch.inverse(Rt2)

# warp from image 2 to image 1
warp_12 = kornia.geometry.warp_frame_depth(
    img2_bchw, dep1[None, None], trans_12, K1
)

# warp from image 1 to image 2
warp_21 = kornia.geometry.warp_frame_depth(
    img1_bchw, dep2[None, None], trans_21, K2
)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, constrained_layout=True)
axes[0, 0].imshow(img1.numpy())
axes[1, 0].imshow(img2.numpy())
axes[0, 1].imshow(dep1.numpy())
axes[1, 1].imshow(dep2.numpy())
axes[0, 2].imshow(warp_12[0].permute(1, 2, 0).numpy())
axes[1, 2].imshow(warp_21[0].permute(1, 2, 0).numpy())
plt.show()
