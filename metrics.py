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

from pathlib import Path 
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from utils.image_utils import rmse
from argparse import ArgumentParser
import numpy as np
import glob


def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
lpips = LPIPS()
def cal_lpips(a, b, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)

    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = lpips(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean


def readImages_lp(renders_dir, gt_dir, depth_dir, gtdepth_dir, masks_dir,tgt = None):
    renders = []
    gts = []
    image_names = []
    depths = []
    gt_depths = []
    masks = []

    # renders_dir = Path(renders_dir)  # Ensure it's a Path object
    # renders_list = sorted(renders_dir.iterdir()) 
    # renders_dir = sorted(renders_dir)
    if tgt is None:
        assert 0
        tgt =  os.listdir(renders_dir)
        tgt = sorted(tgt)
    else:
        # only used for faked IP
        tgt = sorted(tgt)
        assert tgt 

    # #the tgt id is the abs id!
    # renders_dir = renders_dir.replace('test','video').replace('train','video')
    # gt_dir = gt_dir.replace('test','video').replace('train','video')
    # depth_dir = depth_dir.replace('test','video').replace('train','video')
    # gtdepth_dir = gtdepth_dir.replace('test','video').replace('train','video')
    # masks_dir = masks_dir.replace('test','video').replace('train','video')

    # for fname in os.listdir(renders_dir):
    for fname in tgt:
        #read the lp results as it ti
        render = np.array(Image.open(renders_dir / fname))
        depth = np.array(Image.open(depth_dir / fname))#[:-12]
        # remove 0 in the front of the fname
        if fname[0] == '0':
            fname = fname[1:]

        # they hard crop the 12 px already in the render img!
        # we need to do the same for gts so to preceed the same
        gt = np.array(Image.open(gt_dir / fname))[:-12]
        gt_depth = np.array(Image.open(gtdepth_dir / fname))[:-12]
        mask = np.array(Image.open(masks_dir / fname))[:-12]
        
        # assert 0,f'{gt.shape} {render.shape} {depth.shape} {gt_depth.shape} {mask.shape}'

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)[:, :, :, :].cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        
        image_names.append(fname)
    return renders, gts, depths, gt_depths, masks, image_names

def readImages(renders_dir, gt_dir, depth_dir, gtdepth_dir, masks_dir):
    renders = []
    gts = []
    image_names = []
    depths = []
    gt_depths = []
    masks = []

    tgt =  os.listdir(renders_dir)
    tgt = sorted(tgt)

    for fname in tgt:
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))
        depth = np.array(Image.open(depth_dir / fname))
        gt_depth = np.array(Image.open(gtdepth_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)[:, :, :, :].cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        
        image_names.append(fname)
    return renders, gts, depths, gt_depths, masks, image_names

def evaluate(model_paths, target_masks = ['full'],metric_save_root = None,metric_save_name = None, skip_save = False, verbose = True,
             ):

    os.makedirs(metric_save_root, exist_ok=True)
    results_file = os.path.join(metric_save_root,f'{metric_save_name}.json')
    per_view_results_file = os.path.join(metric_save_root,f'{metric_save_name}_PERVIEW.json')

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    with torch.no_grad():
        NOT_RENDERED_LIST = []
        for tartget_mask in target_masks:
            print(f'////////////{tartget_mask}//////////////')
            for scene_dir in sorted(model_paths):
                
                gs_method = scene_dir.split('/')[-2]
                model_varient = scene_dir.split('/')[-1]
                seq_name = scene_dir.split('/')[-3]
                assert gs_method in ['ttgs','deform3dgs','endog','surggs'], f'{gs_method}'

                # create dict if the key no exist
                if seq_name not in full_dict:
                    full_dict[seq_name] = {}
                    per_view_dict[seq_name] = {}
                    full_dict_polytopeonly[seq_name] = {}
                    per_view_dict_polytopeonly[seq_name] = {}

                if model_varient not in full_dict[seq_name]:
                    full_dict[seq_name][model_varient] = {}
                    per_view_dict[seq_name][model_varient] = {}
                    full_dict_polytopeonly[seq_name][model_varient] = {}
                    per_view_dict_polytopeonly[seq_name][model_varient] = {}

                test_dir = Path(scene_dir) / args.phase

                if not os.path.isdir(test_dir):
                    NOT_RENDERED_LIST.append(test_dir)
                    continue

                for method in os.listdir(test_dir):
                    if verbose:
                        print("Method:",model_varient,model_varient)
                    assert len(os.listdir(test_dir))== 1, f'{os.listdir(test_dir)} {test_dir/method}... more than one iter varients for the model?'

                    method_dir = test_dir / method

                    if gs_method in ['ttgs','deform3dgs']:
                        # gt_dir = method_dir/ "gt"
                        gt_dir = method_dir/ "gt_images"
                        # renders_dir = method_dir / "renders"
                        renders_dir = method_dir / "Renders"
                        depth_dir = method_dir / "Depth"
                        gt_depth_dir = method_dir / "Depth"
                        # gt_depth_dir = method_dir / "gt_depth"
                        # masks_dir = method_dir / "masks"
                        # masks_dir = method_dir / "masks_merged"
                        masks_dir = method_dir / "masks_raw_tissue"#!!!

                    elif gs_method == 'surggs':
                        gt_dir = method_dir / "gt_color"
                        renders_dir = method_dir / "renders"
                        gt_depth_dir = method_dir / "gt_depth"
                        depth_dir = method_dir / "depth"
                        # masks_dir = method_dir / "masks"
                        masks_dir = method_dir / "masks_raw_tissue"

                    elif gs_method == 'endog':
                        gt_dir = method_dir / "gt"
                        renders_dir = method_dir / "renders"
                        gt_depth_dir = method_dir / "gt_depth"
                        depth_dir = method_dir / "depth"
                        # masks_dir = method_dir / "masks"
                        masks_dir = method_dir / "masks_raw_tissue"
                    
                    else:
                        assert 0, f'unknown gs_method: {gs_method}'
                    
                    # hard code rendered results for eval nerf based methods
                    
                    renders, gts, depths, gt_depths, masks, image_names = readImages(renders_dir, gt_dir, depth_dir, gt_depth_dir, masks_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []
                    rmses = []
                    
                    loop_tgt =  tqdm(range(len(renders)), desc='Metric evaluation progress') if verbose else range(len(renders))
                    
                    for idx in loop_tgt:
                        render, gt, depth, gt_depth, mask = renders[idx], gts[idx], depths[idx], gt_depths[idx], masks[idx]
                        if tartget_mask == 'tool':
                            mask = (1-mask).to(torch.uint8)
                        elif tartget_mask == 'tissue':
                            mask = mask
                        elif tartget_mask == 'full':
                            mask = torch.ones_like(mask)
                        else:
                            assert 0

                        render = render * mask
                        gt = gt * mask
                        psnrs.append(psnr(render, gt))
                        ssims.append(ssim(render, gt))
                        lpipss.append(cal_lpips(render, gt))
                        if (gt_depth!=0).sum() < 10:
                            continue
                        rmses.append(rmse(depth, gt_depth, mask))

                    if verbose:
                        print("Scene: ", seq_name,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                        print("Scene: ", seq_name,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("Format: ", seq_name, tartget_mask, model_varient)
                    print('&', 
                          "{:.2f}".format(torch.tensor(psnrs).mean().item()),
                          "&",
                          "{:.2f}".format(torch.tensor(ssims).mean().item()*100),
                    )

                    full_dict[seq_name][model_varient].update({f"SSIM_{tartget_mask}": torch.tensor(ssims).mean().item(),
                                                            f"PSNR_{tartget_mask}": torch.tensor(psnrs).mean().item(),
                                                            # f"LPIPS_{tartget_mask}": torch.tensor(lpipss).mean().item(),
                                                            # f"RMSE_{tartget_mask}": torch.tensor(rmses).mean().item(),
                                                            }
                                                            )
                    per_view_dict[seq_name][model_varient].update({f"SSIM_{tartget_mask}": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                f"PSNR_{tartget_mask}": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                # f"LPIPS_{tartget_mask}": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                # f"RMSES_{tartget_mask}": {name: lp for lp, name in zip(torch.tensor(rmses).tolist(), image_names)}
                                                                }
                                                                )

        print('Not rendered:',len(NOT_RENDERED_LIST))
        print(NOT_RENDERED_LIST)
        if not skip_save:
            with open(results_file, 'w') as fp:
                avg_full_dict = {}
                for seq_name in full_dict:
                    for method in full_dict[seq_name].keys():
                        if method not in avg_full_dict:
                            avg_full_dict[method] = {
                                'SSIM_full_avg': [],
                                'PSNR_full_avg': [],
                                'SSIM_tool_avg': [],
                                'PSNR_tool_avg': []
                            }
                        method_data = full_dict[seq_name][method]
                        for metric_type in ['full', 'tool']:
                            if f'SSIM_{metric_type}' in method_data:
                                avg_full_dict[method][f'SSIM_{metric_type}_avg'].append(method_data[f'SSIM_{metric_type}'])
                            if f'PSNR_{metric_type}' in method_data:
                                avg_full_dict[method][f'PSNR_{metric_type}_avg'].append(method_data[f'PSNR_{metric_type}'])
                
                # Compute averages
                for method in avg_full_dict:
                    for key in avg_full_dict[method]:
                        if avg_full_dict[method][key]:
                            avg_full_dict[method][key] = sum(avg_full_dict[method][key]) / len(avg_full_dict[method][key])
                        else:
                            avg_full_dict[method][key] = None
                
                full_dict.update(avg_full_dict)
                json.dump(full_dict, fp, indent=True)

                print('per_model and avg resullts saved in',results_file)

            # with open(per_view_results_file, 'a') as fp:
            #     # json.dump(per_view_dict[scene_dir], fp, indent=True)
            #     # json.dump(per_view_dict[seq_name], fp, indent=True)
            #     json.dump(per_view_dict, fp, indent=True)

            # print('resullts saved in',results_file)
            # print('perivew resullts saved in',per_view_results_file)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', 
    # required=True, 
    nargs="+", type=str, default=[])
    parser.add_argument('--phase', '-p', type=str, default='test')
    parser.add_argument('--exp_root', '-r', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    # parser.add_argument('--dataset_subdir', '-d', type=str, default='EndoNeRF_metric')
    parser.add_argument('--seq_subdirs', '-s', nargs="+", type=str, default=['pulling','cutting'])
    parser.add_argument('--method_subdir', type=str, default='ttgs',choices=['deform3dgs','ttgs','endog','surggs'])
    parser.add_argument('--method_varient_subdir', type=str, default='ALL',choices=['ToolOnly','ALL'])
    args = parser.parse_args()

    # dataset_subdir = args.dataset_subdir
    seq_subdirs = args.seq_subdirs
    method_subdir = args.method_subdir
    method_varient_subdir = args.method_varient_subdir
    metric_save_root = args.save_root

    model_paths = []
    for seq_subdir in seq_subdirs:
        # for subdir in method_varient_subdir:
        # model_paths_given_subdir_i = glob.glob(f'{args.exp_root}/{dataset_subdir}/{seq_subdir}/{method_subdir}/{method_varient_subdir}')
        model_paths_given_subdir_i = glob.glob(f'{args.exp_root}/{seq_subdir}/{method_subdir}/{method_varient_subdir}')
        model_paths.extend(model_paths_given_subdir_i)

    assert len(model_paths) == len(seq_subdirs), f'only allow computation on one method at a time for cleaness, while found {len(model_paths)} models for {len(seq_subdirs)} sequences: {model_paths} given {seq_subdirs} and {method_subdir} and {method_varient_subdir}'

    args.model_paths = set(model_paths)

    print('There are total',len(args.model_paths),f'models to evaluate given seq {seq_subdirs}..')
    
    # metric_save_name = f'{dataset_subdir}_{"_".join(seq_subdirs)}_{method_subdir}_{method_varient_subdir}'
    metric_save_name = f'{"_".join(seq_subdirs)}_{method_subdir}_{method_varient_subdir}'

    evaluate(args.model_paths,
             target_masks=['full','tool','tissue'], #full, tool, tissue area metrics
             metric_save_root=metric_save_root,
             metric_save_name=metric_save_name, 
             skip_save = False,
             verbose=False,
             )


 
