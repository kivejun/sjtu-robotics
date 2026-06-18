# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class TransformerActorCritic(nn.Module):
    """Actor-critic with a small transformer over explicit dynamic-obstacle tokens.

    Observation layout follows the SEA-Nav Go2 dynamic obstacle task:
        props | rays | K * [rel_x, rel_y, rel_vx, rel_vy, dist, ttc] | global_dynamic | goal

    The transformer only encodes the K obstacle tokens. Static obstacle context
    still comes from the ray observations, and the temporal navigation latent is
    kept from the original SEA-Nav MLP encoder.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.5,
        num_props=12,
        num_rays=31,
        num_dynamic_obstacle_obs=0,
        his_len=10,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        transformer_dropout=0.0,
        dynamic_token_dim=6,
        dynamic_global_dim=2,
        **kwargs,
    ):
        if kwargs:
            print(
                "TransformerActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        activation_fn = get_activation(activation)

        self.his_len = his_len
        self.num_rays = num_rays
        self.num_props = num_props
        self.num_dynamic_obstacle_obs = num_dynamic_obstacle_obs
        self.num_obs_one_step = num_rays + num_props + num_dynamic_obstacle_obs + 2
        self.num_obs_hist = self.num_obs_one_step * his_len
        self.num_actions = num_actions
        self.num_latent = 16
        self.dynamic_token_dim = int(dynamic_token_dim)
        self.dynamic_global_dim = min(int(dynamic_global_dim), int(num_dynamic_obstacle_obs))

        dynamic_token_obs = max(0, int(num_dynamic_obstacle_obs) - self.dynamic_global_dim)
        if dynamic_token_obs > 0 and dynamic_token_obs % self.dynamic_token_dim == 0:
            self.num_dynamic_tokens = dynamic_token_obs // self.dynamic_token_dim
        else:
            self.num_dynamic_tokens = 0
            self.dynamic_global_dim = int(num_dynamic_obstacle_obs)

        self.uses_dynamic_transformer = self.num_dynamic_tokens > 0
        self.transformer_dim = int(transformer_dim) if self.uses_dynamic_transformer else 0

        if self.uses_dynamic_transformer:
            self.dynamic_token_proj = nn.Linear(self.dynamic_token_dim, self.transformer_dim)
            self.dynamic_cls_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_dim,
                nhead=int(transformer_heads),
                dim_feedforward=self.transformer_dim * 4,
                dropout=float(transformer_dropout),
                activation="gelu",
                batch_first=True,
            )
            self.dynamic_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))

        base_step_dim = self.num_props + self.num_rays + 2
        actor_input_dim = base_step_dim + self.dynamic_global_dim + self.transformer_dim + self.num_latent
        critic_input_dim = actor_input_dim

        self.actor = self._make_mlp(actor_input_dim, actor_hidden_dims, num_actions, activation_fn)
        self.critic = self._make_mlp(critic_input_dim, critic_hidden_dims, 1, activation_fn)
        self.encoder = self._make_mlp(self.num_obs_hist, encoder_hidden_dims, self.num_latent, activation_fn)

        print(f"TransformerActorCritic dynamic tokens: {self.num_dynamic_tokens} x {self.dynamic_token_dim}")
        print(f"TransformerActorCritic dynamic global dim: {self.dynamic_global_dim}")
        print(f"Dynamic Transformer: {getattr(self, 'dynamic_encoder', None)}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder MLP: {self.encoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation_fn):
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation_fn]
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def extract(self, observations):
        obs_hist = observations
        obs_buf = observations[:, -self.num_obs_one_step :]
        props = obs_buf[:, : self.num_props]
        rays_start = self.num_props
        rays_end = rays_start + self.num_rays
        rays = obs_buf[:, rays_start:rays_end]
        dynamic_start = rays_end
        dynamic_end = dynamic_start + self.num_dynamic_obstacle_obs
        dynamic_obs = obs_buf[:, dynamic_start:dynamic_end]
        goals = obs_buf[:, -2:]

        token_dim_total = self.num_dynamic_tokens * self.dynamic_token_dim
        if self.uses_dynamic_transformer:
            dynamic_tokens = dynamic_obs[:, :token_dim_total].reshape(
                observations.shape[0], self.num_dynamic_tokens, self.dynamic_token_dim
            )
            dynamic_global = dynamic_obs[:, token_dim_total : token_dim_total + self.dynamic_global_dim]
        else:
            dynamic_tokens = None
            dynamic_global = dynamic_obs[:, : self.dynamic_global_dim]

        base_step = torch.cat((props, rays, goals), dim=-1)
        return obs_hist, base_step, dynamic_tokens, dynamic_global

    def encode_dynamic_obstacles(self, dynamic_tokens):
        if not self.uses_dynamic_transformer:
            return None

        token_embed = self.dynamic_token_proj(dynamic_tokens)
        batch_size = token_embed.shape[0]
        cls = self.dynamic_cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, token_embed), dim=1)

        token_padding = dynamic_tokens.abs().sum(dim=-1) < 1.0e-6
        cls_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=dynamic_tokens.device)
        key_padding_mask = torch.cat((cls_padding, token_padding), dim=1)

        encoded = self.dynamic_encoder(sequence, src_key_padding_mask=key_padding_mask)
        return encoded[:, 0]

    def build_actor_critic_input(self, observations):
        obs_hist, base_step, dynamic_tokens, dynamic_global = self.extract(observations)
        latent = self.encoder(obs_hist)
        inputs = [base_step]
        if self.dynamic_global_dim > 0:
            inputs.append(dynamic_global)
        dynamic_context = self.encode_dynamic_obstacles(dynamic_tokens)
        if dynamic_context is not None:
            inputs.append(dynamic_context)
        inputs.append(latent)
        return torch.cat(inputs, dim=-1)

    def update_distribution(self, observations):
        policy_input = self.build_actor_critic_input(observations)
        mean = self.actor(policy_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.mean = mean

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, im_heights=None, im_rays=None):
        policy_input = self.build_actor_critic_input(observations)
        return self.actor(policy_input)

    def evaluate(self, observations, **kwargs):
        policy_input = self.build_actor_critic_input(observations)
        return self.critic(policy_input)
