#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2
import torch.nn.functional as F
import roma
# import kornia


def erode_mask_torch(masks, kernel_size=5):
    # masks: [B, H, W]
    assert len(masks.shape) == 3
    if (masks.dtype == torch.bool):
        masks = masks.float() 
    else:
        assert 0, masks.max()
    # if not (masks.dtype == torch.bool):
    #     assert masks.max() <= 1
    #     masks = masks.to(torch.bool)

    from kornia.morphology import erosion
    erosion_kernel = torch.ones(kernel_size, kernel_size).to(masks.device)#.cuda()

    #erorion function input: b,1{or3},h,w
    erosion_results = erosion(masks.unsqueeze(1), kernel = erosion_kernel).squeeze(1)
    # return erosion_results
    return erosion_results>0



def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, step=0.5) -> torch.Tensor:
    # https://github.com/clemense/quaternion-conventions
    # 3D Gaussian Format: w-x-y-z Roma Format: x-y-z-w

    ndim = q0.ndim
    if ndim == 1:
        q0 = q0.unsqueeze(0)
        q1 = q1.unsqueeze(0)
        
    q0 = torch.nn.functional.normalize(q0)
    q1 = torch.nn.functional.normalize(q1)
    q0 = q0[..., [1, 2, 3, 0]]
    q1 = q1[..., [1, 2, 3, 0]]
    steps = torch.tensor([step], device=q1.device).float()
    q = roma.utils.unitquat_slerp(q0, q1, steps) 
    q = q[..., [3, 0, 1, 2]].squeeze(0)
    
    if ndim == 1:
        q = q.squeeze(0)
    
    return q


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def inpaint_rgb(rgb_image, mask):
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Inpaint missing regions
    inpainted_image = cv2.inpaint(rgb_image, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return inpainted_image

def inpaint_depth(depth_image, mask):
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Inpaint missing regions
    inpainted_depth_image = cv2.inpaint((depth_image).astype(np.uint8), mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return inpainted_depth_image 

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000,warmup_steps = 0, disable = False
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or (step < warmup_steps) or disable:
            # Disable this parameter
            return 0.0
        # assert 0,f'{step} {lr_init} {lr_final} {lr_delay_steps} {lr_delay_mult} {max_steps} {warmup_steps}'
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        # if step>=max_steps:
            # return 0.0
        # else:
            # t = 1
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))



def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    src: streetgs
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix_numpy(r):
    q = r / np.linalg.norm(r)

    R = np.zeros((3, 3))

    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - r*z)
    R[0, 2] = 2 * (x*z + r*y)
    R[1, 0] = 2 * (x*y + r*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - r*x)
    R[2, 0] = 2 * (x*z - r*y)
    R[2, 1] = 2 * (y*z + r*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    return R

def quaternion_to_matrix(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
  



def RotMat2RPY(rot_matrices):
    """
    Convert a batch of rotation matrices (B x N x 3 x 3) to RPY (Euler angles, B x N x 3).
    Assumes rotation matrix uses ZYX convention.

    Args:
        rot_matrices (torch.Tensor): Rotation matrices of shape (B, N, 3, 3).

    Returns:
        torch.Tensor: RPY angles of shape (B, N, 3) [roll, pitch, yaw].
    """
    # Extract components of the rotation matrix
    R11 = rot_matrices[..., 0, 0]
    R12 = rot_matrices[..., 0, 1]
    R13 = rot_matrices[..., 0, 2]
    R21 = rot_matrices[..., 1, 0]
    R31 = rot_matrices[..., 2, 0]
    R32 = rot_matrices[..., 2, 1]
    R33 = rot_matrices[..., 2, 2]

    # Compute roll, pitch, and yaw
    roll = torch.atan2(R32, R33)  # Rotation around x-axis
    pitch = torch.asin(-R31)      # Rotation around y-axis
    yaw = torch.atan2(R21, R11)   # Rotation around z-axis

    # Stack angles into a tensor of shape (B, N, 3)
    rpy = torch.stack([roll, pitch, yaw], dim=-1)

    return rpy


def startswith_any(k, l):
    for s in l:
        if k.startswith(s):
            return True
    return False

    
def NumpytoTorch(image, resolution, resize_mode=cv2.INTER_AREA):
    if resolution is not None:
        image = cv2.resize(image, resolution, interpolation=resize_mode)
    
    image = torch.from_numpy(np.array(image))
    if len(image.shape) == 2:
        image = image[..., None].permute(2, 0, 1) # [1, H, W]
    elif len(image.shape) == 3:
        image = image.permute(2, 0, 1)
    
    return image