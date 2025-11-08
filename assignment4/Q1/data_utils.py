import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

SH_C0 = 0.28209479177387814
CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 5.0, 7.0

class TruckDataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split: {self.split}")

        self.masks = []
        self.points = []
        self.images = []
        self.cameras = []

        imgs_root = os.path.join(root, "imgs")
        poses_root = os.path.join(root, "poses")
        points_root = os.path.join(root, "points")
        self.points_path = os.path.join(points_root, "points_10000.npy")

        data_img_size = None
        num_files = len(os.listdir(imgs_root))
        test_idxs = np.linspace(0, num_files, 7).astype(np.int32)[1:-1]
        test_idxs_set = set(test_idxs.tolist())
        train_idxs = [i for i in range(num_files) if i not in test_idxs_set]
        idxs = train_idxs if self.split == "train" else test_idxs

        for i in idxs:
            img_path = os.path.join(imgs_root, f"frame{i+1:06d}.jpg")
            npy_path = os.path.join(poses_root, f"frame{i+1:06d}.npy")

            img_ = imageio.v3.imread(img_path).astype(np.float32) / 255.0

            mask = None
            if img_.shape[-1] == 3:
                img = torch.tensor(img_)  # (H, W, 3)
            else:
                img = torch.tensor(img_[..., :3])  # (H, W, 3)
                mask = torch.tensor(img_[..., 3:4])  # (H, W, 1)
                
            img_size = img.shape[:2]
            h, w = img_size
            
            # Checking if all data samples have the same image size
            if data_img_size is None:
                data_img_size = (w,h) 
            else:
                if data_img_size[0] != img_size[1] or data_img_size[1] != img_size[0]:
                    raise RuntimeError

            pose = np.load(npy_path)
            R, T, F, C = pose[:9].reshape((3,3)), pose[9:12], pose[12:14], pose[14:16]
            
            # Screen space camera
            F = F * min(img_size) / 2 
            C = w / 2 - C[0] * min(img_size) / 2, h / 2 - C[1] * min(img_size) / 2  

            camera = PerspectiveCameras(
                focal_length=torch.tensor(F, dtype=torch.float)[None], 
                principal_point=torch.tensor(C, dtype=torch.float)[None],
                R=torch.tensor(R, dtype=torch.float)[None], 
                T=torch.tensor(T, dtype=torch.float)[None],
                in_ndc=False,
                image_size=((h,w),)
            )

            self.images.append(img)
            self.cameras.append(camera)
            if mask is not None:
                self.masks.append(mask)

        self.img_size = data_img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        masks = None
        if len(self.masks) > 0:
            masks = self.masks[idx]
        return self.images[idx], self.cameras[idx], masks

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch], dim=0)
        cameras = [x[1] for x in batch]

        masks = [x[2] for x in batch if x[2] is not None]
        if len(masks) == 0:
            masks = None
        else:
            masks = torch.stack(masks, dim=0)

        return images, cameras, masks


def colour_depth_q1_render(depth):
    normalized_depth = (depth - CMAP_MIN_NORM) / (CMAP_MAX_NORM - CMAP_MIN_NORM + 1e-8)
    coloured_depth = CMAP_JET(normalized_depth)[:, :, :3]  # (H, W, 3)
    coloured_depth = (np.clip(coloured_depth, 0.0, 1.0) * 255.0).astype(np.uint8)

    return coloured_depth

def visualize_renders(scene, gt_viz_img, cameras, img_size):

    imgs = []
    viz_size = (256, 256)
    with torch.no_grad():
        for cam in cameras:
            pred_img, _, _ = scene.render(
                cam, img_size=img_size,
                bg_colour=(0.0, 0.0, 0.0),
                per_splat=-1,
            )
            img = torch.clamp(pred_img, 0.0, 1.0) * 255.0
            img = img.clone().detach().cpu().numpy().astype(np.uint8)

            if img_size[0] != viz_size[0] or img_size[1] != viz_size[1]:
                img = np.array(Image.fromarray(img).resize(viz_size))

            imgs.append(img)

    pred_viz_img = np.concatenate(imgs, axis=1)
    viz_frame = np.concatenate((pred_viz_img, gt_viz_img), axis=0)
    return viz_frame

def load_gaussians_from_ply(path):
    # Modified from https://github.com/thomasantony/splat
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * SH_C0 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_colours
    }
    return output

import torch

def colours_from_spherical_harmonics(spherical_harmonics: torch.Tensor,
                                     gaussian_dirs: torch.Tensor) -> torch.Tensor:
    """
    Computes view-dependent colour given spherical harmonic coefficients
    and direction vectors for each Gaussian.

    Args:
        spherical_harmonics : (N, 48) SH coefficients (16 coefficients × 3 channels)
        gaussian_dirs       : (N, 3) direction vectors from camera to Gaussian

    Returns:
        colours             : (N, 3) RGB in [0, 1]
    """

    device = spherical_harmonics.device
    sh = spherical_harmonics.view(-1, 16, 3)  # (N, 16, 3)

    # ---- Spherical harmonic constants ----
    SH0 = 0.282095  # Y_0^0

    # l = 1
    SH1 = 0.488603  # all l=1 terms share this scale

    # l = 2
    C2 = torch.tensor(
        [ 1.092548,  # xy
         -1.092548,  # yz
          0.315392,  # 2z^2 - x^2 - y^2
         -1.092548,  # xz
          0.546274], # x^2 - y^2
        device=device
    )

    # l = 3
    C3 = torch.tensor(
        [-0.590044,  # y (3x^2 - y^2)
          2.890611,  # xyz
         -0.457046,  # y (4z^2 - x^2 - y^2)
          0.373176,  # z (2z^2 - 3x^2 - 3y^2)
         -0.457046,  # x (4z^2 - x^2 - y^2)
          1.445306,  # z (x^2 - y^2)
         -0.590044], # x (x^2 - 3y^2)
        device=device
    )

    # ---- Directions and monomials ----
    x = gaussian_dirs[:, 0]
    y = gaussian_dirs[:, 1]
    z = gaussian_dirs[:, 2]

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # ---- 0th order ----
    c0 = sh[:, 0]                          # (N, 3)
    colour = SH0 * c0                      # (N, 3)

    # ---- 1st order (3 coeffs: c1, c2, c3) ----
    c1 = sh[:, 1]
    c2 = sh[:, 2]
    c3 = sh[:, 3]

    colour = (
        colour
        - SH1 * y.unsqueeze(-1) * c1
        + SH1 * z.unsqueeze(-1) * c2
        - SH1 * x.unsqueeze(-1) * c3
    )

    # ---- 2nd order (5 coeffs: c4..c8) ----
    c4  = sh[:, 4]
    c5  = sh[:, 5]
    c6  = sh[:, 6]
    c7  = sh[:, 7]
    c8  = sh[:, 8]

    colour = (
        colour
        + C2[0] * (xy).unsqueeze(-1)                      * c4
        + C2[1] * (yz).unsqueeze(-1)                      * c5
        + C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1)      * c6
        + C2[3] * (xz).unsqueeze(-1)                      * c7
        + C2[4] * (xx - yy).unsqueeze(-1)                 * c8
    )

    # ---- 3rd order (7 coeffs: c9..c15) ----
    c9  = sh[:,  9]
    c10 = sh[:, 10]
    c11 = sh[:, 11]
    c12 = sh[:, 12]
    c13 = sh[:, 13]
    c14 = sh[:, 14]
    c15 = sh[:, 15]

    colour = (
        colour
        + C3[0] * (y * (3.0 * xx - yy)).unsqueeze(-1)                 * c9
        + C3[1] * (xy * z).unsqueeze(-1)                              * c10
        + C3[2] * (y * (4.0 * zz - xx - yy)).unsqueeze(-1)            * c11
        + C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(-1)* c12
        + C3[4] * (x * (4.0 * zz - xx - yy)).unsqueeze(-1)            * c13
        + C3[5] * (z * (xx - yy)).unsqueeze(-1)                       * c14
        + C3[6] * (x * (xx - 3.0 * yy)).unsqueeze(-1)                 * c15
    )

    # Shift & clamp like your original
    colours = torch.clamp(colour + 0.5, 0.0, 1.0)

    return colours