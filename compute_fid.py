# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import os
from os import path as osp
from tqdm import tqdm
from pathlib import Path
import shutil

import torch
import numpy as np
from PIL import Image

from third_party.genesis_v2.geco import GECO
from sri.model import SRI
from sri.fixed_order_model import FixedOrderSRI
from third_party.pytorch_fid import fid_score as FID
from third_party.genesis_v2.shapestacks_config import ShapeStacksDataset
import argparse

torch.set_printoptions(threshold=10000, linewidth=300)


def restore_from_checkpoint(args):

    state = torch.load(args['checkpoint'])
    
    if args['model'] == 'SRI':
        model = SRI(args['K'], args['pixel_bound'], args['z_dim'], 
                    args['semiconv'], args['kernel'], args['pixel_std'],
                    args['dynamic_K'], args['image_likelihood'], False,
                    args['img_size'], args['s_dim'], args['L'])
    elif args['model'] == 'FixedOrderSRI':
        model = FixedOrderSRI(args['slot_attention_iters'], args['K'],
                          args['pixel_bound'], args['z_dim'], 
                          args['pixel_std'], args['image_likelihood'], 
                          args['img_size'], args['s_dim'], args['L'])
    model = model.to(args['device'])
    new_state_dict = {}
    for k,v in state['model_state_dict'].items():
        # remove string 'module.' from the key
        if 'module.' in k:
            new_state_dict[k.replace('module.', '')] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    print(f'loaded {args["checkpoint"]}')
    return model


def tensor_to_png(tensor, save_dir, count, stop):
    np_images = tensor.cpu().numpy()
    np_images = np.moveaxis(np_images, 1, 3)
    for i in range(len(np_images)):
        im = Image.fromarray(np.uint8(255*np_images[i]))
        fn = osp.join(save_dir, str(count).zfill(6) + '.png')
        im.save(fn)
        count += 1
        if count >= stop:
            return count
    return count


def fid_from_model(model, test_loader, out_dir, batch_size=10,
                    num_images=10000, feat_dim=2048, img_dir='/tmp',
                    verbose=True):
    # Save images from test set as pngs
    test_dir = osp.join(img_dir, 'test_images')
    if not os.path.isdir(test_dir):
        if verbose:
            print("Saving images from test set as pngs.")
        os.makedirs(test_dir)
        count = 0
        for bidx, batch in enumerate(test_loader):
            batch['imgs'] = (batch['imgs'] + 1) / 2.
            count = tensor_to_png(batch['imgs'], test_dir, count, num_images)
            if count >= num_images:
                break

    # Generate images and save as pngs
    print("Generate images and save as pngs.")
    gen_dir = osp.join(img_dir, 'generated_images')
    # Delete the folder if it already exists
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)
    count = 0
    for _ in tqdm(range(num_images // batch_size + 1), disable=not verbose):
        with torch.no_grad():
            recons, _ = model.sample(batch_size=batch_size, K_steps=model.K_steps)

        count = tensor_to_png(recons, gen_dir, count, num_images)
        if count >= num_images:
            break
    # Compute FID
    print("Computing FID.")
    gpu = next(model.parameters()).is_cuda
    fid_value = FID.calculate_fid_given_paths(
        [test_dir, gen_dir], batch_size, gpu, feat_dim, verbose)
    
    print(f"FID: {fid_value}")

   
    with open(out_dir / 'fid.txt', 'a+') as f:
        f.write('{}\n'.format(fid_value))
    return fid_value


def fid_from_folder(model_folder, test_loader, out_dir, img_dir='/tmp', batch_size=10, num_images=10000,
                    feat_dim=2048, verbose=False):
     # Save images from test set as pngs
    
    test_dir = osp.join(img_dir, 'test_images')
    if not os.path.isdir(test_dir):
        if verbose:
            print("Saving images from test set as pngs.")
        os.makedirs(test_dir)
        count = 0
        for bidx, batch in enumerate(test_loader):
            batch['imgs'] = (batch['imgs'] + 1) / 2.
            count = tensor_to_png(batch['imgs'], test_dir, count, num_images)
            if count >= num_images:
                break
    # Compute FID
    if verbose:
        print("Computing FID.")

    fid_value = FID.calculate_fid_given_paths(
        [test_dir, model_folder], batch_size, True, feat_dim, verbose)
    
    print(f"FID: {fid_value}")

    with open(out_dir / 'fid.txt', 'a+') as f:
        f.write('{}\n'.format(fid_value))
    return fid_value


def main(args):

    # Fix random seed
    seed = args['seed']
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)


    te_dataset = ShapeStacksDataset(data_dir=args['data_dir'], mode='test', load_instances=False)
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=args['batch_size'],
                                                shuffle=True, num_workers=args['num_workers'],
                                                drop_last=False)
    
    out_dir = Path(args['out_dir'], 'results', 
                   Path(args['checkpoint']).stem + f'-test-seed={seed}')
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    if args['from_folder'] == "":
        model = restore_from_checkpoint(args)
        model.eval()
        # Compute FID
        fid_from_model(model, te_dataloader, out_dir, args['batch_size'],
                    args['num_fid_images'], args['feat_dim'], args['img_dir'])
    else:
        fid_from_folder(args['from_folder'], te_dataloader,out_dir, args['img_dir'],
                        args['batch_size'], args['num_fid_images'], args['feat_dim'], verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('segregate-relate-imagine FID')


    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Workers for dataloading')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--from_folder', type=str, default='',
                        help='Provide a folder containing generated images. Used ' \
                             'for computing FID for a model defined external to this repo.')
    parser.add_argument('--feat_dim', type=int, default=2048,
                        help='FID net feature dim')
    parser.add_argument('--num_fid_images', type=int, default=10000,
                        help='# of images to use for FID calculation')
    parser.add_argument('--img_dir', type=str, default='/tmp',
                        help='.png directory')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Generate images in batches')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Parent folder of data')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Trained model weights')
    parser.add_argument('--out_dir', type=str, default='experiments',
                        help='output folder for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to run on') 
    
    ### Model
    parser.add_argument('--z_dim', type=int, default=64,
                        help='slot dim')
    parser.add_argument('--s_dim', type=int, default=128,
                        help='s dim')
    parser.add_argument('--L', type=int, default=3,
                        help='number of relation layers in scene encoder')
    parser.add_argument('--K', type=int, default=9,
                        help='number of slots')       
    ###########################################                                                             
    parser.add_argument('--kernel', type=str, default='gaussian',
                        help='{laplacian,gaussian,epanechnikov}')
    parser.add_argument('--semiconv', action='store_true', default=True,
                        help='semiconv for GENESISv2')
    parser.add_argument('--dynamic_K', action='store_true', default=False,
                        help='dynamic_K for GENESISv2')
    parser.add_argument('--pixel_bound',  action='store_true', default=True,
                        help='constrain decoder outputs')
    parser.add_argument('--img_size', type=int, default=64,
                        help='side length of square image inputs')
    parser.add_argument('--pixel_std', type=float, default=0.7,
                        help='Global std dev for image likelihood')
    parser.add_argument('--image_likelihood', type=str, default='MoG-Normed',
                        help='MoG or MoG-Normed (use normalization constant)')
    parser.add_argument('--slot_attention_iters', type=int, default=3, 
                        help='number of iterations in slot attention')
                        
    args = vars(parser.parse_args())

    main(args)
