import argparse
import torch
import numpy as np
from sri.model import SRI
from sri.fixed_order_model import FixedOrderSRI
from sri.visualization import create_mask_image
import torchvision
from pathlib import Path
from PIL import Image


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



@torch.no_grad()
def main(args):

    # load model
    model = restore_from_checkpoint(args)
    model.eval()

    seed = args['seed']
    print(f'Setting random seed to {seed}')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)

    out_dir = Path(args['out_dir'], 'results', 
                   Path(args['checkpoint']).stem + f'-test-seed={seed}')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    print(f'saving results in {out_dir}')

    temp = args['temperature']
    all_imgs, all_masks = [], []
    grids = []
    if args['save_images_as'] == 'grid':
        batch_size = 4 
        iters = 4 
        padding = 1
    elif args['save_images_as'] == 'folder':
        batch_size = 10
        iters = 10
        padding=0


    for i in range(iters):

        recons, stats = model.sample(batch_size=batch_size, temp=temp, K_steps=model.K_steps)
        m = []
        for idx in range(batch_size):
            m_ = []
            for _, val in enumerate(stats['log_m_k']):
                val = val.exp()
                m_ += [val]
            m += [create_mask_image(torch.stack(m_)[:,idx])]
            
        if args['save_images_as'] == 'grid':
            all_imgs += [ torchvision.utils.make_grid(recons,padding=padding) ] # [C,H,bs*W]
            all_masks += [ torchvision.utils.make_grid(m,padding=padding) ] # [C,H,bs*W]
            grids += [ torch.cat((all_imgs[-1], all_masks[-1]), 1) ]
        elif args['save_images_as'] == 'folder':
            for j in range(batch_size):
                to_png = recons[j].permute(1,2,0).data.cpu().numpy()
                to_png = Image.fromarray(np.uint8((to_png)*255))
                to_png.save(out_dir / f'temp={temp}_{((i*10)+j):04d}.png')

    if args['save_images_as'] == 'grid':
        all_imgs = torchvision.utils.make_grid(
                        torch.stack(all_imgs).permute(0,1,3,2),padding=padding).permute(0,2,1)
        all_masks = torchvision.utils.make_grid(
                        torch.stack(all_masks).permute(0,1,3,2),padding=padding).permute(0,2,1)
        grids = torchvision.utils.make_grid(
                        torch.stack(grids).permute(0,1,3,2),nrow=10,padding=padding).permute(0,2,1)  

        all_imgs = all_imgs.permute(1,2,0).data.cpu().numpy()
        Image.fromarray(np.uint8((all_imgs)*255)).save(out_dir / f'all_imgs_temp={temp}.png')

        all_masks = all_masks.permute(1,2,0).data.cpu().numpy()
        Image.fromarray(np.uint8((all_masks)*255)).save(out_dir / f'all_masks_temp={temp}.png')

        grids = grids.permute(1,2,0).data.cpu().numpy()
        Image.fromarray(np.uint8((grids)*255)).save(out_dir / f'grids_temp={temp}.png')
        
        # rinfo = {}
        # rinfo["outputs"] = {
        #     "recons": all_imgs.permute(1,2,0).data.cpu().numpy(),
        #     "masks": all_masks.permute(1,2,0).data.cpu().numpy(),
        #     "combined": grids.permute(1,2,0).data.cpu().numpy()
        #     #"pred_mask": masks.permute(0,1,3,4,2).data.cpu().numpy(),
        #     #"pred_mask_logits": mask_logits.permute(0,1,3,4,2).data.cpu().numpy(),
        #     #"components": components.permute(0,1,3,4,2).data.cpu().numpy()
        #     #"rgbs": all_rgbs.permute(1,2,0).data.cpu().numpy()
        # }
        # pkl.dump(rinfo, open(out_dir / f'rinfo.pkl', 'wb'))       


if __name__ == '__main__':
    parser = argparse.ArgumentParser('segregate-relate-imagine training')
    
    # test
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default='shapestacks',
                        help='the training dataset name')
    parser.add_argument('--data_dir', type=str, 
                        help='$PATH_TO/shapestacks')
    parser.add_argument('--model', type=str, default='SRI', choices=['SRI', 'FixedOrderSRI'],
                        help='which model to test')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='experiments',
                        help='output folder for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to run on')    

    ###########################################
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
    ###########################################
    parser.add_argument('--save_images_as', type=str, default='grid', choices=['grid', 'individual'],
                        help='how to present the samples')  
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for sampling')
    args = vars(parser.parse_args())

    main(args)    