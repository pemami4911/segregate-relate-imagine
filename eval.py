import argparse
import torch
import numpy as np
from sri.model import SRI
from sri.fixed_order_model import FixedOrderSRI
from sri.metrics import adjusted_rand_index
from third_party.genesis_v2.shapestacks_config import ShapeStacksDataset
from third_party.genesis_v2.geco import GECO
from pathlib import Path


def restore_from_checkpoint(args):
    
    state = torch.load(args['checkpoint'])    
    model_state_dict = state['model_state_dict']

    if args['model'] == 'SRI':
        pass
    elif args['model'] == 'FixedOrderSRI':
        model = FixedOrderSRI(args['slot_attention_iters'], args['K'],
                          args['pixel_bound'], args['z_dim'], 
                          args['pixel_std'], args['image_likelihood'], 
                          args['img_size'], args['s_dim'], args['L'])
    model = model.to(args['device'])
    new_state_dict = {}
    for k,v in model_state_dict.items():
        # remove string 'module.' from the key
        if 'module.' in k:
            new_state_dict[k.replace('module.', '')] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)          
    num_elements = 3 * model.img_size**2  # Assume three input channels
    # Goal is specified per pixel & channel so it doesn't need to
    # be changed for different resolutions etc.
    geco_goal = args['g_goal'] * num_elements
    # Scale step size to get similar update at different resolutions
    geco_lr = args['g_lr'] * (64**2 / model.img_size**2)
    geco_sri = GECO(geco_goal, geco_lr, args['g_alpha'], args['g_init'],
            args['g_min'], args['g_speedup'])
    geco_sri = geco_sri.to(args['device'])
    geco_sri.beta = state['beta']
    geco_sri.err_ema = state['err_ema']

    return model, geco_sri


@torch.no_grad()
def main(args):

    # load model
    model, geco = restore_from_checkpoint(args)
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

    results = {'bpd_nelbo': [], 'fg_ari': []}
   
    #################### Evaluate ELBO ####################
    total_images = 0

    bpd_coeff = 1. / np.log(2.) / (3*64*64)

    if args['dataset'] == 'shapestacks':
        dataset = ShapeStacksDataset(data_dir=args['data_dir'], mode='val')
    else:
        raise ValueError(f'Unsupported dataset {args["dataset"]}')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                shuffle=True, num_workers=4,
                                drop_last=False)

    # 5K validation images from validation set
    for i, batch in enumerate(dataloader):
        if total_images >= 5000:
            break

        imgs = batch['imgs'].to(args['device'])
        imgs = (imgs + 1) / 2.

        if args['model'] == 'FixedOrderSRI':
            sri_output, losses, stats = model(imgs, use_normalized_nll=True)

            # Reconstruction errors
            err_sri = losses['sri_err'].mean(0)
            # KL divergences
            # -- KL scene
            kl_scene = losses['sri_kl_scene']
            if kl_scene is not None:
                kl_scene = kl_scene.mean(0)

            kl_slot = losses['sri_kl_slot'].mean(0)
            kl = kl_scene + kl_slot
            # log p(x|z) - log_(q(z|x)) + log_p(z)
            nelbo = (err_sri + kl).detach()
        elif args['model'] == 'SRI':
            pass # TODO

        bpd_nelbo = bpd_coeff * nelbo
        results['bpd_nelbo'] += [bpd_nelbo.data.cpu().numpy()]

        total_images += imgs.shape[0]
    
    #################### Evaluate FG-ARI ####################
    if args['dataset'] == 'shapestacks':
        dataset = ShapeStacksDataset(data_dir=args['data_dir'], mode='test')
    else:
        raise ValueError(f'Unsupported dataset {args["dataset"]}')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=4,
                                drop_last=False)

    total_images = 0
    for i,batch in enumerate(dataloader):
        if total_images >= 320:
            break

        imgs = batch['imgs'].to('cuda')
        true_masks = batch['masks'].to('cuda')

        if args['model'] == 'FixedOrderSRI':
            imgs = (imgs + 1) / 2.
            sri_output, losses, stats = model(imgs, use_normalized_nll=True)
      
            if 'log_m_r_k' in stats:
                pred_masks = torch.stack(stats['log_m_r_k'], 4).exp()
            else:
                pred_masks = torch.stack(stats['log_m_k'], 4).exp()
            pred_masks = pred_masks.permute(0, 4, 1, 2,3)
        elif args['model'] == 'SRI':
            pass

        # ARI
        ari = adjusted_rand_index(true_masks, pred_masks,
                                  background_idxs=dataloader.dataset.bg_indices)
        ari = ari.data.cpu().numpy().reshape(-1)
        results['fg_ari'] += [ari]

        total_images += imgs.shape[0]
             
    with open(out_dir / 'results.txt', 'w+') as f:
        for k,v in results.items():
            print('{} : {} +- {}'.format(
                k, np.mean(v), np.std(v)
            ))
            f.write(f'{k},{np.mean(v)},{np.std(v)}\n')


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
    # model
    parser.add_argument('--g_goal', type=float, default=0.5655,
                        help='GECO recon goal')
    parser.add_argument('--g_lr', type=float, default=1e-6, 
                        help='GECO learning rate')
    parser.add_argument('--g_alpha', type=float, default=0.99,
                        help='GECO momentum for error')
    parser.add_argument('--g_init', type=float, default=1.0,
                        help='GECO init Lagrange factor')
    parser.add_argument('--g_min', type=float, default=1e-10,
                        help='GECO min Lagrange factor')
    parser.add_argument('--g_speedup', type=float, default=10.,
                        help='Scale GECO Lr if delta positive')
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
    parser.add_argument('--slot_attention_iters', type=int, default=3, 
                        help='number of iterations in slot attention')
    parser.add_argument('--pixel_bound',  action='store_true', default=True,
                        help='constrain decoder outputs')
    parser.add_argument('--img_size', type=int, default=64,
                        help='side length of square image inputs')
    parser.add_argument('--pixel_std', type=float, default=0.7,
                        help='Global std dev for image likelihood')
    parser.add_argument('--image_likelihood', type=str, default='MoG-Normed',
                        help='MoG or MoG-Normed (use normalization constant)')

    args = vars(parser.parse_args())

    main(args)    