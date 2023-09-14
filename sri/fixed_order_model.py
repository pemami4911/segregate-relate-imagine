from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from third_party.genesis_v2.modules.unet import UNet
import third_party.genesis_v2.modules.blocks as B
from third_party.genesis_v2.monet_config import MONet
from sri.utils import mvn, std_mvn
from sri.model import SceneEncoder, AutoregressivePrior, SRI


class FixedOrderSlotAttention(nn.Module):
    """FixedOrderSlotAttention module."""
    def __init__(self, img_size, num_iterations, num_slots, slot_size,
                 mlp_hidden_size, slot_stddev=1.0, seed=1, epsilon=1e-8):
        """Builds the FixedOrderSlotAttention module.
        Args:
            img_size (int): Image size (assumed square).
            num_iterations: Number of iterations.
            num_slots: Number of slots.
            slot_size: Dimensionality of slot feature vectors.
            mlp_hidden_size: Hidden layer size of MLP.
            epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.slot_stddev = slot_stddev
        self.seed = seed
        self.epsilon = epsilon

        self.positional_embedding = FixedOrderSlotAttention.create_positional_embedding(
            img_size, img_size)
        self.positional_embedding_projection = nn.Linear(4, self.slot_size)

        self.encoder = UNet(
            num_blocks=int(np.log2(img_size)-1),
            img_size=img_size,
            filter_start=min(self.slot_size, 64),
            in_chnls=3,
            out_chnls=self.slot_size,
            norm='gn')
        self.encoder.final_conv = nn.Identity()

        self.norm_inputs = nn.LayerNorm(self.slot_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Parameters for Gaussian init (no weight-sharing across slots)
        self.slots_mu = torch.FloatTensor(1, self.num_slots, self.slot_size)
        # init a torch.FloatTensor with 'he_uniform'
        nn.init.kaiming_uniform_(self.slots_mu)
        self.slots_mu = nn.Parameter(self.slots_mu)

        self.slots_log_sigma = torch.FloatTensor(1, self.num_slots, self.slot_size)
        nn.init.kaiming_uniform_(self.slots_log_sigma)
        self.slots_log_sigma = nn.Parameter(self.slots_log_sigma)
                                     
        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRU(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )


    @staticmethod
    def create_positional_embedding(h, w):
        dist_right = torch.linspace(1, 0, w).view(w,1,1).repeat(1,h,1)  # [w,h,1]
        dist_left = torch.linspace(0, 1, w).view(w,1,1).repeat(1,h,1)
        dist_top = torch.linspace(0, 1, h).view(1,h,1).repeat(w,1,1)
        dist_bottom = torch.linspace(1, 0, h).view(1,h,1).repeat(w,1,1)
        return torch.cat([dist_right, dist_left, dist_top, dist_bottom],2).unsqueeze(0)


    def forward(self, inputs):
        """
        Args:
            inputs: image tensor [N, C, H, W]

        Returns:
            slots: Slot tensor [N, num_slots, slot_size].
        """
        slots, unorm_attn = None, None
        
        pos_embed = self.positional_embedding.to(inputs.device).repeat(inputs.shape[0],1,1,1)
        pos_embed = self.positional_embedding_projection(pos_embed)

        # --- Extract features ---
        enc_feat, _ = self.encoder(inputs)
        enc_feat = enc_feat.permute(0,2,3,1).contiguous()  # [batch_size,H,W,self.slot_size]
        enc_feat = F.relu(enc_feat)
        enc_feat = enc_feat + pos_embed
        inputs = self.norm_inputs(enc_feat)
        inputs = inputs.view(inputs.shape[0], -1, self.slot_size)  # [batch_size, H*W,self.slot_size]

        k = self.project_k(inputs)  # Shape: [batch_size, H*W, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, H*W, slot_size].
       
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_mu = self.slots_mu.repeat(inputs.shape[0], 1, 1)
        slots_log_sigma = self.slots_log_sigma.repeat(inputs.shape[0], 1, 1)
        slots = torch.distributions.normal.Normal(slots_mu, torch.exp(slots_log_sigma)).rsample()

        # Multiple rounds of attention.
        # return slots, unorm_attn
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            # Attention
            q = self.project_q(slots)  # [batch_size, num_slots, slot_size]
            q *= self.slot_size ** -0.5  # Normalization

            # k is [batch_size, H*W, slot_size]
            # q is [batch_size, num_slots, slot_size]
            dots = torch.einsum('bid,bjd->bij', q, k)  # [batch_size, H*W, H*W]
            attn = dots.softmax(dim=1) + self.epsilon
            # Slot Attention normalization
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [batch_size, num_slots, slot_size]
            
            slots, _ = self.gru(
                    updates.reshape(1,-1,self.slot_size),  # [1, batch_size*num_slots, slot_size]
                    slots_prev.reshape(1,-1,self.slot_size))  
            
            slots = slots.reshape(inputs.shape[0], -1, self.slot_size)  # [batch_size, num_slots, slot_size]
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class FixedOrderSRI(nn.Module):
    def __init__(self, slot_attention_iters, K_steps, pixel_bound, feat_dim,
                 pixel_std, image_likelihood, img_size, scene_latent_dim,
                 scene_encoder_layers):
        super(FixedOrderSRI, self).__init__()
        # Configuration
        self.K_steps = K_steps
        self.feat_dim = feat_dim
        self.image_likelihood = image_likelihood
        self.img_size = img_size
        self.scene_latent_dim = scene_latent_dim
        self.pixel_bound = pixel_bound

        self.z_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 2*feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*feat_dim, 2*feat_dim))

        # self.slot_attn = SlotAttention(num_iters, num_slots, slot_size,
        #                                2*slot_size, slot_stddev, seed)
        self.fixed_order_slot_attention = FixedOrderSlotAttention(
            img_size, slot_attention_iters, K_steps, feat_dim, feat_dim
        )

        self.scene_encoder = SceneEncoder(feat_dim, scene_latent_dim, scene_encoder_layers)
        self.prior_lstm = AutoregressivePrior(scene_latent_dim, feat_dim)
        c = feat_dim

        self.decoder = nn.Sequential(
            B.BroadcastLayer(img_size // 16),
            nn.ConvTranspose2d(feat_dim+2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1))

        # --- Output pixel distribution ---
        self.std = pixel_std

    def decode_latents(self, z_k):
        
        # --- Reconstruct components and image ---
        x_r_k, m_r_logits_k = [], []
        z_k_batched = torch.stack(z_k)  # [K, batch_size, z_dim]
        batch_size = z_k_batched.shape[1]
        z_k_batched = z_k_batched.view(self.K_steps * batch_size, -1)


        dec = self.decoder(z_k_batched)
        x_r = dec[:,:3].view(self.K_steps, batch_size, 3, self.img_size, self.img_size)
        m_r_logits = dec[:,3:].view(self.K_steps, batch_size, 1, self.img_size, self.img_size)
        x_r_k = torch.tensor_split(x_r, self.K_steps, 0)
        m_r_logits_k = torch.tensor_split(m_r_logits, self.K_steps, 0)
        x_r_k = [_[0] for _ in x_r_k]
        m_r_logits_k = [_[0] for _ in m_r_logits_k]

        # Optional: Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        # --- Reconstruct masks ---
        log_m_r_stack = MONet.get_mask_recon_stack(
            m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k
    
    def forward(self, x, use_normalized_nll=False):
        losses = {}

        batch_size = x.size(0)

        ######################## SRI Set-to-sequence Inference
        # 1. Segregate with fixed order slot attention
        slots = self.fixed_order_slot_attention(x)
        # Map the ordered set of output slots to posteriors over z
        mu, sigma_ps = self.z_head(slots).reshape(batch_size * self.K_steps, -1).chunk(2, dim=1)
        sigma = B.to_sigma(sigma_ps)
        mu = mu.reshape(batch_size, self.K_steps, -1)
        sigma = sigma.reshape(batch_size, self.K_steps, -1)
        mu = torch.tensor_split(mu, self.K_steps, dim=1)
        mu = [_.squeeze(1) for _ in mu]
        sigma = torch.tensor_split(sigma, self.K_steps, dim=1)
        sigma = [_.squeeze(1) for _ in sigma]
        q_z = [mvn(mu_i, sigma_i) for mu_i,sigma_i in zip(mu, sigma)]
        #q_z = Normal(mu, sigma)
        #z = q_z.rsample()
        #z = z.reshape(batch_size, self.K_steps, -1)
        z_as_list = [q_z_i.rsample() for q_z_i in q_z]
        z = torch.stack(z_as_list, dim=1)

        # 2. Relate
        q_scene = self.scene_encoder(z)
        scene_latent = q_scene.rsample() 
       
        ######################## Compute L_SRI_fixed_order
                
        scene_prior = std_mvn(shape=[batch_size, self.scene_latent_dim], device=x.device)
        # compute scene distribution kl
        scene_kl = torch.distributions.kl.kl_divergence(q_scene, scene_prior)  # [batch_size]
        slot_kl = self.prior_lstm.KL_loss(scene_latent, z_as_list, q_z)
                    
        # --- Decode latents ---
        recon_ordered, x_r_k_ordered, log_m_r_k_ordered = self.decode_latents(z_as_list)
        mx_r_k_ordered = [x*logm.exp() for x, logm in zip(x_r_k_ordered, log_m_r_k_ordered)]
    
        # -- Reconstruction loss
        if self.image_likelihood == 'MoG':
            sri_nll = SRI.gmm_negative_loglikelihood(x,
                                log_m_r_k_ordered, x_r_k_ordered, self.std, use_normalized_nll)
        elif self.image_likelihood == 'MoG-Normed':
            sri_nll = SRI.gmm_negative_loglikelihood(x,
                                log_m_r_k_ordered, x_r_k_ordered, self.std, use_normalized_nll=True)

        losses['sri_err'] = sri_nll
        losses['sri_kl_scene'] = scene_kl
        losses['sri_kl_slot'] = slot_kl

        sri_stats = AttrDict(
            recon=recon_ordered, x_r_k=x_r_k_ordered,
            log_m_r_k=log_m_r_k_ordered, mx_r_k=mx_r_k_ordered,
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k_ordered, dim=1), dim=1))
    
        return recon_ordered, losses, sri_stats



    def sample(self, batch_size, temp=1.0, device='cuda', K_steps=None, scene_latent=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        
        if scene_latent is None:
            scene_prior = std_mvn(shape=[batch_size, self.scene_latent_dim], device=device, temperature=temp)
            s = scene_prior.sample()
        else:
            s = scene_latent
        # slots
        z_k, _ = self.prior_lstm(s, K_steps, t=temp)
      
        # Decode latents
        recon, x_r_k, log_m_r_k = self.decode_latents(z_k)

        stats = AttrDict(x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)],
                         z_k=z_k)
        return recon, stats
