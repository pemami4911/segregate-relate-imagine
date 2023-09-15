import os
from pathlib import Path
import datetime
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from third_party.genesis_v2.geco import GECO
from third_party.genesis_v2.shapestacks_config import ShapeStacksDataset
from sri.visualization import visualise_outputs_fixed_order_model
from sri.fixed_order_model import FixedOrderSRI
import argparse

local_rank = os.environ['LOCAL_RANK']

def save_checkpoint(ckpt_file, model, optimiser, beta, geco_err_ema,
                    iter_idx, verbose=True):
    if verbose:
        print(f"Saving model training checkpoint to: {ckpt_file}")
    ckpt_dict = {'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'beta': beta[0],
                'err_ema': geco_err_ema[0],
                'iter_idx': iter_idx}

    torch.save(ckpt_dict, ckpt_file)


def run(training, seed):
    global local_rank
    local_rank = f'cuda:{local_rank}'

    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    for dir_ in [checkpoint_dir, tb_dir]:
        if local_rank == 'cuda:0' and not dir_.exists():
            os.makedirs(dir_)

    tb_dbg = tb_dir / (training['run_suffix'] + '_' + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

    if local_rank == 'cuda:0':
        print(f'Creating SummaryWriter! ({local_rank})')
        writer = SummaryWriter(tb_dbg)
        
    
    print(f'Local rank: {local_rank}. Setting random seed to {seed}')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(training['DDP_port'])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    if training['dataset'] == 'shapestacks':
        tr_dataset = ShapeStacksDataset(data_dir=training['data_dir'], mode=training['mode'])
    else:
        raise ValueError(f'Unsupported dataset {training["dataset"]}')

    batch_size = training['batch_size']
    tr_sampler = DistributedSampler(dataset=tr_dataset)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
            batch_size=batch_size, sampler=tr_sampler, 
            num_workers=training['num_workers'],
            worker_init_fn=worker_init_fn,
            drop_last=True)

    model = FixedOrderSRI(training['slot_attention_iters'], training['K'],
                          training['pixel_bound'], training['z_dim'], 
                          training['pixel_std'], training['image_likelihood'], 
                          training['img_size'], training['s_dim'], training['L'])
    num_elements = 3 * model.img_size**2  # Assume three input channels

    # Goal is specified per pixel & channel so it doesn't need to
    # be changed for different resolutions etc.
    geco_goal = training['g_goal'] * num_elements
    # Scale step size to get similar update at different resolutions
    geco_lr = training['g_lr'] * (64**2 / model.img_size**2)

    geco_sri = GECO(geco_goal, geco_lr, training['g_alpha'], training['g_init'],
            training['g_min'], training['g_speedup'])
    beta_sri = geco_sri.beta
    
    model = model.to(local_rank)
    
    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    geco_sri.to_cuda()
    # Try to restore model and optimiser from checkpoint
    iter_idx = 0
    if training['load_from_checkpoint']:
        checkpoint = checkpoint_dir / training['checkpoint']
        map_location = {'cuda:0': local_rank}
        state = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state['model_state_dict'], strict=True)
        model_opt.load_state_dict(state['optimiser_state_dict'])
        iter_idx = state['iter_idx'] + 1
        geco_sri.beta = state['beta']
        geco_sri.err_ema = state['err_ema']

        
        # Update starting iter
        iter_idx = state['iter_idx'] + 1
    print(f"Starting training at iter = {iter_idx}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=False)
    model.train()
    if local_rank == 'cuda:0':
        print(model)

    # ------------------------
    # TRAINING
    # ------------------------
    model.train()
    epoch = 0
    
    while iter_idx <= training['iters']:
        
         # Re-shuffle every epoch
        tr_sampler.set_epoch(epoch)
        if training['tqdm'] and local_rank == 'cuda:0':
            data_iter = tqdm(tr_dataloader)
        else:
            data_iter = tr_dataloader
        for train_batch in data_iter:
            
            # Parse data
            train_input = train_batch['imgs']
            # back to 0-1
            train_input = (train_input + 1) / 2.
            train_input = train_input.to(local_rank)

            # Forward propagation
            model_opt.zero_grad()
            sri_output, losses, stats = model(train_input)
           
            # Reconstruction errors
            err_sri = losses['sri_err'].mean(0)
            # KL divergences
            # -- KL scene
            kl_scene = losses['sri_kl_scene']
            if kl_scene is not None:
                kl_scene = kl_scene.mean(0)

            kl_slot = losses['sri_kl_slot'].mean(0)


            # Compute MSE / RMSE
            mse_batched = ((train_input-sri_output)**2).mean((1, 2, 3)).detach()
            rmse_batched = mse_batched.sqrt()
            mse, rmse = mse_batched.mean(0), rmse_batched.mean(0)
            
            kl = kl_scene + kl_slot
            sri_elbo = (err_sri + kl)
            
            # Main objective
            
            beta_sri = geco_sri.beta
            loss_sri = geco_sri.loss(err_sri, kl)

            # Backprop and optimise
            loss_sri.backward()

            model_opt.step()
            
            # Heartbeat log
            if iter_idx % training['tensorboard_freq'] == 0 and iter_idx > 0 and local_rank == 'cuda:0':
                # TensorBoard logging
                # -- Optimisation stats
                writer.add_scalar('optim/beta', beta_sri, iter_idx)
                writer.add_scalar('optim/geco_err_ema',
                                    geco_sri.err_ema, iter_idx)
                writer.add_scalar('optim/geco_err_ema_element',
                                    geco_sri.err_ema/num_elements, iter_idx)
                # -- Main loss terms

                writer.add_scalar('train/mse', mse, iter_idx)
                writer.add_scalar('train/rmse', rmse, iter_idx)
                writer.add_scalar('train/err_sri', err_sri, iter_idx)
                if kl_scene is not None:
                    writer.add_scalar('train/kl_scene', kl_scene, iter_idx)
                writer.add_scalar('train/kl_slot', kl_slot, iter_idx)
                writer.add_scalar('train/elbo_sri', sri_elbo, iter_idx)
                writer.add_scalar('train/loss_loss', loss_sri, iter_idx)

                # Visualise model outputs
                visualise_outputs_fixed_order_model(model, train_input, writer, 'train', iter_idx)

            if (iter_idx % training['checkpoint_freq'] == 0 and 
                local_rank == 'cuda:0'):
                # Save the model
                prefix = training['run_suffix']
                save_checkpoint(
                    checkpoint_dir / f'{prefix}-state-{iter_idx}.pth',
                    model.module, model_opt, [beta_sri],
                    [geco_sri.err_ema], iter_idx, False)

            
            if iter_idx >= training['iters']:
                iter_idx += 1
                break
            iter_idx += 1
        epoch += 1

    # Close writer
    if local_rank == 'cuda:0':
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('segregate-reglate-imagine training')
    
    # training
    parser.add_argument('--DDP_port', type=int, default=29500,
                        help='torch.distributed config')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training mini-batch size')
    parser.add_argument('--dataset', type=str, default='shapestacks',
                        help='the training dataset name')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of pytorch dataloader workers')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/val/test')
    parser.add_argument('--data_dir', type=str, default='',
                        help='$PATH_TO/shapestacks')
    parser.add_argument('--iters', type=int, default=500000, 
                        help='number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate for Adam')
    parser.add_argument('--tensorboard_freq', type=int, default=4000,
                        help='how often to write to TB')
    parser.add_argument('--checkpoint_freq', type=int, default=50000,
                        help='how often to save a model ckpt')
    parser.add_argument('--load_from_checkpoint', action='store_true', default=False,
                        help='whether to load training from a checkpoint')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='name of a .pth file')
    parser.add_argument('--run_suffix', type=str, default='debug', 
                        help='string to attach to model name')
    parser.add_argument('--out_dir', type=str, default='experiments',
                        help='output folder for results')
    parser.add_argument('--tqdm', action='store_true', default=False,
                        help='show training progress in CLI')

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

    run(args, args['seed'])
