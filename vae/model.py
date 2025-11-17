"""
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.configs.schema import ModelConfig
from vae.modules.transformer import AttentionBlock, FlashQueryLayer
from vae.utils import (
    DiagonalGaussianDistribution,
    DummyLatent,
    calculate_iou,
    calculate_metrics,
    construct_grid_points,
    extract_mesh,
    sync_timer,
)


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.precision = torch.bfloat16  # manually handle low-precision training, always use bf16

        # point encoder
        self.proj_input = nn.Linear(3 + config.point_fourier_dim, config.hidden_dim)

        self.perceiver = AttentionBlock(
            config.hidden_dim,
            num_heads=config.num_heads,
            dim_context=config.hidden_dim,
            qknorm=config.qknorm,
            qknorm_type=config.qknorm_type,
        )

        if self.config.salient_attn_mode == "dual":
            self.perceiver_dorases = AttentionBlock(
                config.hidden_dim,
                num_heads=config.num_heads,
                dim_context=config.hidden_dim,
                qknorm=config.qknorm,
                qknorm_type=config.qknorm_type,
            )

        # self-attention encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    config.hidden_dim, config.num_heads, qknorm=config.qknorm, qknorm_type=config.qknorm_type
                )
                for _ in range(config.num_enc_layers)
            ]
        )

        # vae bottleneck
        self.norm_down = nn.LayerNorm(config.hidden_dim)
        self.proj_down_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        if not self.config.use_ae:
            self.proj_down_std = nn.Linear(config.hidden_dim, config.latent_dim)
        self.proj_up = nn.Linear(config.latent_dim, config.dec_hidden_dim)

        # self-attention decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    config.dec_hidden_dim, config.dec_num_heads, qknorm=config.qknorm, qknorm_type=config.qknorm_type
                )
                for _ in range(config.num_dec_layers)
            ]
        )

        # cross-attention query
        if self.config.use_flash_query:
            # Triplane decoder head
            self.triplane_head = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(config.dec_hidden_dim),
            nn.Linear(config.dec_hidden_dim, config.triplane_resolution * config.triplane_resolution * config.triplane_channels)
            )
            for _ in range(3)  # XY, XZ, YZ planes
            ])

            # MLP query decoder
            self.mlp_query = nn.Sequential(
            nn.Linear(config.triplane_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

        # preload from a checkpoint (NOTE: this happens BEFORE checkpointer loading latest checkpoint!)
        if self.config.pretrain_path is not None:
            try:
                ckpt = torch.load(self.config.pretrain_path)  # local path
                self.load_state_dict(ckpt["model"], strict=True)
                del ckpt
                print(f"Loaded VAE from {self.config.pretrain_path}")
            except Exception as e:
                print(
                    f"Failed to load VAE from {self.config.pretrain_path}: {e}, make sure you resumed from a valid checkpoint!"
                )

        # log
        n_params = 0
        for p in self.parameters():
            n_params += p.numel()
        print(f"Number of parameters in VAE: {n_params / 1e6:.2f}M")

    # override to support tolerant loading (only load matched shape)
    def load_state_dict(self, state_dict, strict=True, assign=False):
        local_state_dict = self.state_dict()
        seen_keys = {k: False for k in local_state_dict.keys()}
        for k, v in state_dict.items():
            if k in local_state_dict:
                seen_keys[k] = True
                if local_state_dict[k].shape == v.shape:
                    local_state_dict[k].copy_(v)
                else:
                    print(f"mismatching shape for key {k}: loaded {local_state_dict[k].shape} but model has {v.shape}")
            else:
                print(f"unexpected key {k} in loaded state dict")
        for k in seen_keys:
            if not seen_keys[k]:
                print(f"missing key {k} in loaded state dict")

    def fourier_encoding(self, points: torch.Tensor):
        # points: [B, N, 3], float32 for precision
        # assert points.dtype == torch.float32, "Query points must be float32"

        F = self.config.point_fourier_dim // (2 * points.shape[-1])

        if self.config.fourier_version == "v1":  # default
            exponent = torch.arange(1, F + 1, device=points.device, dtype=torch.float32) / F  # [F], range from 0 to 1
            freq_band = 512**exponent  # [F], min frequency is 1, max frequency is 1/freq
            freq_band *= torch.pi
        elif self.config.fourier_version == "v2":
            exponent = torch.arange(F, device=points.device, dtype=torch.float32) / (F - 1)  # [F], range from 0 to 1
            freq_band = 1024**exponent  # [F]
            freq_band *= torch.pi
        elif self.config.fourier_version == "v3":  # hunyuan3d-2
            freq_band = 2 ** torch.arange(F, device=points.device, dtype=torch.float32)  # [F]

        spectrum = points.unsqueeze(-1) * freq_band  # [B,...,3,F]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,3,F]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,3,2,F]
        input_enc = input_enc.view(*points.shape[:-1], -1)  # [B,...,6F] = [B,...,dim]
        return torch.cat([input_enc, points], dim=-1).to(dtype=self.precision)  # [B,...,dim+input_dim]

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format=memory_format)
        self.to(dtype=self.precision, memory_format=memory_format)  # use bfloat16 for training

    def encode(self, data: dict[str, torch.Tensor]):
        # uniform points
        pointcloud = data["pointcloud"]  # [B, N, 3]

        # fourier embed and project
        pointcloud = self.fourier_encoding(pointcloud)  # [B, N, 3+C]
        pointcloud = self.proj_input(pointcloud)  # [B, N, hidden_dim]

        # salient points
        if self.config.use_salient_point:
            pointcloud_dorases = data["pointcloud_dorases"]  # [B, M, 3]

            # fourier embed and project (shared weights)
            pointcloud_dorases = self.fourier_encoding(pointcloud_dorases)  # [B, M, 3+C]
            pointcloud_dorases = self.proj_input(pointcloud_dorases)  # [B, M, hidden_dim]

        # gather fps point
        fps_indices = data["fps_indices"]  # [B, N']
        pointcloud_query = torch.gather(pointcloud, 1, fps_indices.unsqueeze(-1).expand(-1, -1, pointcloud.shape[-1]))

        if self.config.use_salient_point:
            fps_indices_dorases = data["fps_indices_dorases"]  # [B, M']

            if fps_indices_dorases.shape[1] > 0:
                pointcloud_query_dorases = torch.gather(
                    pointcloud_dorases,
                    1,
                    fps_indices_dorases.unsqueeze(-1).expand(-1, -1, pointcloud_dorases.shape[-1]),
                )

                # combine both fps points as the query
                pointcloud_query = torch.cat(
                    [pointcloud_query, pointcloud_query_dorases], dim=1
                )  # [B, N'+M', hidden_dim]

            # dual cross-attention
            if self.config.salient_attn_mode == "dual_shared":
                hidden_states = self.perceiver(pointcloud_query, pointcloud) + self.perceiver(
                    pointcloud_query, pointcloud_dorases
                )  # [B, N'+M', hidden_dim]
            elif self.config.salient_attn_mode == "dual":
                hidden_states = self.perceiver(pointcloud_query, pointcloud) + self.perceiver_dorases(
                    pointcloud_query, pointcloud_dorases
                )
            else:  # single, hunyuan3d-2 style
                hidden_states = self.perceiver(pointcloud_query, torch.cat([pointcloud, pointcloud_dorases], dim=1))
        else:
            hidden_states = self.perceiver(pointcloud_query, pointcloud)  # [B, N', hidden_dim]

        # encoder
        for block in self.encoder:
            hidden_states = block(hidden_states)

        # bottleneck
        hidden_states = self.norm_down(hidden_states)
        latent_mean = self.proj_down_mean(hidden_states).float()
        if not self.config.use_ae:
            latent_std = self.proj_down_std(hidden_states).float()
            posterior = DiagonalGaussianDistribution(latent_mean, latent_std)
        else:
            posterior = DummyLatent(latent_mean)

        return posterior

    def decode(self, latent: torch.Tensor):
        latent = latent.to(dtype=self.precision)
        hidden_states = self.proj_up(latent)  # [B, latent_size, dec_hidden_dim]

        # Self-attention decoder (unchanged)
        for block in self.decoder:
            hidden_states = block(hidden_states)

            # Project to 3 triplanes
            # Average pool tokens to get global features for each plane
            global_features = hidden_states.mean(dim=1, keepdim=True)  # [B, 1, dec_hidden_dim]

        triplanes = []
        for plane_head in self.triplane_head:
        # Project global features to plane
            plane_flat = plane_head(global_features)  # [B, 1, res*res*C]
            plane = plane_flat.view(
            -1,
            self.config.triplane_resolution,
            self.config.triplane_resolution,
            self.config.triplane_channels
            )  # [B, res, res, C]
            triplanes.append(plane)

        return triplanes  # List of 3 tensors [B, 256, 256, 32]

    def query(self, query_points: torch.Tensor, triplanes: list[torch.Tensor]):
# query_points: [B, N, 3], float32 in [-1, 1]
# triplanes: list of 3 tensors [B, res, res, C]

        B, N, _ = query_points.shape
        res = self.config.triplane_resolution
        C = self.config.triplane_channels

        # Normalize coordinates from [-1, 1] to [0, 1] for grid_sample
        coords_01 = (query_points + 1) / 2  # [B, N, 3]
        coords_01 = coords_01.clamp(0, 1)

        # Sample each triplane using bilinear interpolation
        # XY plane: sample using (x, y) coordinates
        xy_coords = coords_01[:, :, [0, 1]]  # [B, N, 2]
        xy_coords_grid = xy_coords * 2 - 1  # Back to [-1, 1] for grid_sample
        xy_coords_grid = xy_coords_grid.unsqueeze(1)  # [B, 1, N, 2]
        triplane_xy = triplanes[0].permute(0, 3, 1, 2)  # [B, C, res, res]
        feat_xy = F.grid_sample(
        triplane_xy, xy_coords_grid,
        mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N]
        feat_xy = feat_xy.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # XZ plane: sample using (x, z) coordinates
        xz_coords = coords_01[:, :, [0, 2]]  # [B, N, 2]
        xz_coords_grid = xz_coords * 2 - 1
        xz_coords_grid = xz_coords_grid.unsqueeze(1)  # [B, 1, N, 2]
        triplane_xz = triplanes[1].permute(0, 3, 1, 2)  # [B, C, res, res]
        feat_xz = F.grid_sample(
        triplane_xz, xz_coords_grid,
        mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N]
        feat_xz = feat_xz.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # YZ plane: sample using (y, z) coordinates
        yz_coords = coords_01[:, :, [1, 2]]  # [B, N, 2]
        yz_coords_grid = yz_coords * 2 - 1
        yz_coords_grid = yz_coords_grid.unsqueeze(1)  # [B, 1, N, 2]
        triplane_yz = triplanes[2].permute(0, 3, 1, 2)  # [B, C, res, res]
        feat_yz = F.grid_sample(
        triplane_yz, yz_coords_grid,
        mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, 1, N]
        feat_yz = feat_yz.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # Compute adaptive weights based on gradient estimation
        # Use finite difference to estimate SDF gradient direction
        eps = 0.01
        with torch.enable_grad():
        # For weight computation, use simple average first (bootstrap)
            feat_avg = (feat_xy + feat_xz + feat_yz) / 3
            sdf_vals = self.mlp_query(feat_avg.to(self.precision))  # [B, N, 1]

        # Estimate gradient via central differences
        # Sample 6 neighboring points (±eps in x, y, z)
        offsets = torch.tensor([
        [eps, 0, 0], [-eps, 0, 0],
        [0, eps, 0], [0, -eps, 0],
        [0, 0, eps], [0, 0, -eps]
        ], device=query_points.device, dtype=torch.float32)

        # For efficiency, approximate gradient using subset of points
        # In practice, can sample fewer neighboring points or use analytical gradients
        grad_x = (self._quick_sample(query_points + offsets[0:1], triplanes) -
        self._quick_sample(query_points + offsets[1:2], triplanes)) / (2 * eps)
        grad_y = (self._quick_sample(query_points + offsets[2:3], triplanes) -
        self._quick_sample(query_points + offsets[3:4], triplanes)) / (2 * eps)
        grad_z = (self._quick_sample(query_points + offsets[4:5], triplanes) -
        self._quick_sample(query_points + offsets[5:6], triplanes)) / (2 * eps)

        grad = torch.cat([grad_x, grad_y, grad_z], dim=-1)  # [B, N, 3]
        grad_norm = grad.norm(dim=-1, keepdim=True) + 1e-6  # [B, N, 1]
        grad_normalized = grad / grad_norm  # [B, N, 3]

        # Compute weights: higher weight for planes perpendicular to gradient (parallel to surface)
        # Plane normals: XY=(0,0,1), XZ=(0,1,0), YZ=(1,0,0)
        weight_xy = torch.abs(grad_normalized[:, :, 2:3])  # |grad_z| [B, N, 1]
        weight_xz = torch.abs(grad_normalized[:, :, 1:2])  # |grad_y| [B, N, 1]
        weight_yz = torch.abs(grad_normalized[:, :, 0:1])  # |grad_x| [B, N, 1]

        # Normalize weights
        weight_sum = weight_xy + weight_xz + weight_yz + 1e-6
        weight_xy = weight_xy / weight_sum
        weight_xz = weight_xz / weight_sum
        weight_yz = weight_yz / weight_sum

        # Weighted aggregation
        feat_aggregated = (
        weight_xy * feat_xy +
        weight_xz * feat_xz +
        weight_yz * feat_yz
        )  # [B, N, C]

        # MLP decoder
        pred = self.mlp_query(feat_aggregated.to(self.precision))  # [B, N, 1]

        return pred

    def _quick_sample(self, points: torch.Tensor, triplanes: list[torch.Tensor]):
# Helper method for fast triplane sampling (simple average)
# Used only for gradient estimation
        coords_01 = (points + 1) / 2
        coords_01 = coords_01.clamp(0, 1)

        # Sample all 3 planes and average (no gradient weighting in helper)
        xy_coords = coords_01[:, :, [0, 1]] * 2 - 1
        xz_coords = coords_01[:, :, [0, 2]] * 2 - 1
        yz_coords = coords_01[:, :, [1, 2]] * 2 - 1

        feat_xy = F.grid_sample(triplanes[0].permute(0, 3, 1, 2), xy_coords.unsqueeze(1),
        mode='bilinear', padding_mode='border', align_corners=False)
        feat_xz = F.grid_sample(triplanes[1].permute(0, 3, 1, 2), xz_coords.unsqueeze(1),
        mode='bilinear', padding_mode='border', align_corners=False)
        feat_yz = F.grid_sample(triplanes[2].permute(0, 3, 1, 2), yz_coords.unsqueeze(1),
        mode='bilinear', padding_mode='border', align_corners=False)

        feat_avg = (feat_xy + feat_xz + feat_yz) / 3
        feat_avg = feat_avg.squeeze(2).permute(0, 2, 1)

        return self.mlp_query(feat_avg.to(self.precision))


        # *Edge cases handled:*

        # - Coordinates clamped to [0, 1] to handle boundary queries
        # - Gradient normalization with epsilon to avoid division by zero
        # - Weight normalization ensures sum = 1
        # - Helper method for efficient gradient estimation



    def training_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output = {}

        # cut off fps point during training for progressive flow
        if self.training:
            # randomly choose from a set of cutoff candidates
            cutoff_index = np.random.choice(len(self.config.cutoff_fps_prob), p=self.config.cutoff_fps_prob)
            cutoff_fps_point = self.config.cutoff_fps_point[cutoff_index]
            cutoff_fps_salient_point = self.config.cutoff_fps_salient_point[cutoff_index]
            # prefix of FPS points are still FPS points
            data["fps_indices"] = data["fps_indices"][:, :cutoff_fps_point]
            if self.config.use_salient_point:
                data["fps_indices_dorases"] = data["fps_indices_dorases"][:, :cutoff_fps_salient_point]

        loss = 0

        # encode
        posterior = self.encode(data)
        latent_geom = posterior.sample() if self.training else posterior.mode()

        # decode
        triplanes = self.decode(latent_geom)

        # cross-attention query
        query_points = data["query_points"]  # [B, N, 3], float32

        pred = self.query(query_points, triplanes).squeeze(-1).float()
        gt = data["query_gt"].float()  # [B, N], in [-1, 1]

        # main loss
        loss_mse = F.mse_loss(pred, gt, reduction="mean")
        loss += loss_mse

        loss_l1 = F.l1_loss(pred, gt, reduction="mean")
        loss += loss_l1

        # kl loss
        loss_kl = posterior.kl().mean()
        loss += self.config.kl_weight * loss_kl

        # metrics
        with torch.no_grad():
            output["scalar"] = {}  # for wandb logging
            output["scalar"]["loss_mse"] = loss_mse.detach()
            output["scalar"]["loss_l1"] = loss_l1.detach()
            output["scalar"]["loss_kl"] = loss_kl.detach()
            output["scalar"]["iou_fg"] = calculate_iou(pred, gt, target_value=1)
            output["scalar"]["iou_bg"] = calculate_iou(pred, gt, target_value=0)
            output["scalar"]["precision"], output["scalar"]["recall"], output["scalar"]["f1"] = calculate_metrics(
                pred, gt, target_value=1
            )

        return output, loss

    @torch.no_grad()
    def validation_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return self.training_step(data, iteration)

    @torch.inference_mode()
    @sync_timer("vae forward")
    def forward(
        self,
        data: dict[str, torch.Tensor],
        mode: Literal["dense", "hierarchical"] = "hierarchical",
        max_samples_per_iter: int = 8192,
        resolution: int = 512,
        min_resolution: int = 64,  # for hierarchical
    ) -> dict[str, torch.Tensor]:
        output = {}

        # encode
        if "latent" in data:
            latent = data["latent"]
        else:
            posterior = self.encode(data)
            output["posterior"] = posterior
            latent = posterior.mode()

        output["latent"] = latent
        B = latent.shape[0]

        # decode
        triplanes = self.decode(latent)
        output["triplanes"] = triplanes # List of 3 tensors [B, 256, 256, 32]

        # query
        def chunked_query(grid_points):
            if grid_points.shape[0] <= max_samples_per_iter:
                return self.query(grid_points.unsqueeze(0), triplanes).squeeze(-1)
            all_pred = []
            for i in range(0, grid_points.shape[0], max_samples_per_iter):
                grid_chunk = grid_points[i : i + max_samples_per_iter]
                pred_chunk = self.query(grid_chunk.unsqueeze(0), triplanes)
                all_pred.append(pred_chunk)
            return torch.cat(all_pred, dim=1).squeeze(-1)


        if mode == "dense":
            grid_points = construct_grid_points(resolution).to(latent.device)
            grid_points = grid_points.contiguous().view(-1, 3)
            grid_vals = chunked_query(grid_points).float().view(B, resolution + 1, resolution + 1, resolution + 1)

        elif mode == "hierarchical":
            assert resolution >= min_resolution, "Resolution must be greater than or equal to min_resolution"
            assert B == 1, "Only one batch is supported for hierarchical mode"

            resolutions = []
            res = resolution
            while res >= min_resolution:
                resolutions.append(res)
                res = res // 2
            resolutions.reverse()  # e.g., [64, 128, 256, 512]

            # dense-query the coarsest resolution
            res = resolutions[0]
            grid_points = construct_grid_points(res).to(latent.device)
            grid_points = grid_points.contiguous().view(-1, 3)
            grid_vals = chunked_query(grid_points).float().view(res + 1, res + 1, res + 1)

            # sparse-query finer resolutions
            dilate_kernel_3 = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32, device=latent.device)
            dilate_kernel_5 = torch.ones(1, 1, 5, 5, 5, dtype=torch.float32, device=latent.device)
            for i in range(1, len(resolutions)):
                res = resolutions[i]
                # get the boundary grid mask in the coarser grid (where the grid_vals have different signs with at least one of its neighbors)
                grid_signs = grid_vals >= 0
                mask = torch.zeros_like(grid_signs)
                mask[1:, :, :] += grid_signs[1:, :, :] != grid_signs[:-1, :, :]
                mask[:-1, :, :] += grid_signs[:-1, :, :] != grid_signs[1:, :, :]
                mask[:, 1:, :] += grid_signs[:, 1:, :] != grid_signs[:, :-1, :]
                mask[:, :-1, :] += grid_signs[:, :-1, :] != grid_signs[:, 1:, :]
                mask[:, :, 1:] += grid_signs[:, :, 1:] != grid_signs[:, :, :-1]
                mask[:, :, :-1] += grid_signs[:, :, :-1] != grid_signs[:, :, 1:]
                # empirical: also add those with abs(grid_vals) < 0.95
                mask += grid_vals.abs() < 0.95
                mask = (mask > 0).float()
                # empirical: dilate the coarse mask
                if res < 512:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask = F.conv3d(mask, weight=dilate_kernel_3, padding=1)
                    mask = mask.squeeze(0).squeeze(0)
                # get the coarse coordinates
                cidx_x, cidx_y, cidx_z = torch.nonzero(mask, as_tuple=True)
                # fill to the fine indices
                mask_fine = torch.zeros(res + 1, res + 1, res + 1, dtype=torch.float32, device=latent.device)
                mask_fine[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
                # empirical: dilate the fine mask
                if res < 512:
                    mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                    mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_3, padding=1)
                    mask_fine = mask_fine.squeeze(0).squeeze(0)
                else:
                    mask_fine = mask_fine.unsqueeze(0).unsqueeze(0)
                    mask_fine = F.conv3d(mask_fine, weight=dilate_kernel_5, padding=2)
                    mask_fine = mask_fine.squeeze(0).squeeze(0)
                # get the fine coordinates
                fidx_x, fidx_y, fidx_z = torch.nonzero(mask_fine, as_tuple=True)
                # convert to float query points
                query_points = torch.stack([fidx_x, fidx_y, fidx_z], dim=-1)  # [N, 3]
                query_points = query_points * 2 / res - 1  # [N, 3], in [-1, 1]
                # query
                pred = chunked_query(query_points).float()
                # fill to the fine indices
                grid_vals = torch.full((res + 1, res + 1, res + 1), -100.0, dtype=torch.float32, device=latent.device)
                grid_vals[fidx_x, fidx_y, fidx_z] = pred
                # print(f"[INFO] hierarchical: resolution: {res}, valid coarse points: {len(cidx_x)}, valid fine points: {len(fidx_x)}")

            grid_vals = grid_vals.unsqueeze(0)  # [1, res+1, res+1, res+1]
            grid_vals[grid_vals <= -100.0] = float("nan")  # use nans to ignore invalid regions

        # extract mesh
        meshes = []
        for b in range(B):
            vertices, faces = extract_mesh(grid_vals[b], resolution)
            meshes.append((vertices, faces))
        output["meshes"] = meshes

        return output
