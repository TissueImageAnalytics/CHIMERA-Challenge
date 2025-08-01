# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.proj(x)
    
## softmax weights per modality
class ModalityAttentionFusion(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(len(input_dims)))

    def forward(self, features):
        stacked = torch.stack(features, dim=1)  # [B, N_modalities, D]
        weights = torch.softmax(self.weights, dim=0)  # [N_modalities]
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        return fused

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim[d], num_heads=num_heads, batch_first=True)
            for d in range(len(dim))
        ])

    def forward(self, features):
        B = features[0].size(0)
        N = len(features)

        keys_values = torch.stack(features, dim=1)  # [B, N, D]
        attended = []

        for i in range(N):
            query = features[i].unsqueeze(1)  # [B, 1, D]
            attn = self.attn_layers[i]
            # Exclude self from keys/values to force intermodal learning
            kv = torch.stack([features[j] for j in range(N) if j != i], dim=1)  # [B, N-1, D]
            out, _ = attn(query, kv, kv)
            attended.append(out.squeeze(1))  # [B, D]

        fused = torch.mean(torch.stack(attended, dim=1), dim=1)  # [B, D]
        return fused

class CrossAttentionGatedFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim[d], num_heads=num_heads, batch_first=True)
            for d in range(len(dim))
        ])
        self.gate_layer = nn.Sequential(
            nn.Linear(sum(int(d) for d in dim), len(dim)),
            nn.Sigmoid()
        )

    def forward(self, features):
        B = features[0].size(0)
        N = len(features)
        keys_values = torch.stack(features, dim=1)  # [B, N, D]
        attended = []

        for i in range(N):
            query = features[i].unsqueeze(1)  # [B, 1, D]
            attn = self.attn_layers[i]
            # Exclude self from keys/values to force intermodal learning
            kv = torch.stack([features[j] for j in range(N) if j != i], dim=1)  # [B, N-1, D]
            out, _ = attn(query, kv, kv)
            attended.append(out.squeeze(1))  # [B, D]

        # Gated weighting of attended features
        concat = torch.cat(attended, dim=1)  # [B, D * N]
        gates = self.gate_layer(concat)      # [B, N]
        gated = [gates[:, i:i+1] * attended[i] for i in range(N)]
        fused = torch.sum(torch.stack(gated, dim=1), dim=1)  # [B, D]
        return fused

class CrossAttentionWithSelfAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim[d], num_heads=num_heads, batch_first=True)
            for d in range(len(dim))
        ])

        # Self-attention block over attended outputs
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim[0], nhead=num_heads, batch_first=True)
        self.self_attn_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features):
        B = features[0].size(0)
        N = len(features)

        keys_values = torch.stack(features, dim=1)  # [B, N, D]
        attended = []

        for i in range(N):
            query = features[i].unsqueeze(1)  # [B, 1, D]
            attn = self.attn_layers[i]
            # Exclude self from keys/values to force intermodal learning
            kv = torch.stack([features[j] for j in range(N) if j != i], dim=1)  # [B, N-1, D]
            out, _ = attn(query, kv, kv)
            attended.append(out.squeeze(1))  # [B, D]

        # fused = torch.mean(torch.stack(attended, dim=1), dim=1)  # [B, D]
        stacked = torch.stack(attended, dim=1)  # [B, N, D]
        refined = self.self_attn_block(stacked)  # [B, N, D]
        fused = refined.mean(dim=1)  # [B, D]
        return fused

## linear layer after concat
class LinearFusion(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        total_dim = sum(input_dims)
        self.fc = nn.Linear(total_dim, max(input_dims))

    def forward(self, features):
        concat = torch.cat(features, dim=1)
        return self.fc(concat)

## concat with no learnable params
class SimpleConcatFusion(nn.Module):
    def forward(self, features):
        return torch.cat(features, dim=1)

class CoxHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)

class DeepHitHead(nn.Module):
    def __init__(self, in_dim, time_bins):
        super().__init__()
        self.fc = nn.Linear(in_dim, time_bins)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

class MultimodalSurvivalModel(nn.Module):
    def __init__(self, clin_dim, mri_dim, wsi_dim, fusion_type='modality', survival_model='cox', time_bins=30):
        super().__init__()

        self.clinical_proj = ProjectionHead(clin_dim, 128) if clin_dim > 0 else None
        self.mri_proj = ProjectionHead(mri_dim, 128) if mri_dim > 0 else None
        self.wsi_proj = ProjectionHead(wsi_dim, 128) if wsi_dim > 0 else None

        input_dims = []
        if self.clinical_proj is not None:
            input_dims.append(128)
        if self.mri_proj is not None:
            input_dims.append(128)
        if self.wsi_proj is not None:
            input_dims.append(128)

        if fusion_type == 'simple':
            self.fusion = SimpleConcatFusion()
            fusion_dim = sum(input_dims)
        elif fusion_type == 'modality':
            self.fusion = ModalityAttentionFusion(input_dims)
            fusion_dim = 128  # fixed projection size
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(input_dims)
            fusion_dim = 128
        elif fusion_type == 'cross_attention_gated':
            self.fusion = CrossAttentionGatedFusion(input_dims)
            fusion_dim = 128
        elif fusion_type == 'cross_attention_with_self':
            self.fusion = CrossAttentionWithSelfAttentionFusion(input_dims)
            fusion_dim = 128
        else:
            self.fusion = LinearFusion(input_dims)
            fusion_dim = 128  # weighted fusion projects down too

        if survival_model == 'cox':
            self.head = CoxHead(fusion_dim)
        else:
            self.head = DeepHitHead(fusion_dim, time_bins)

        self.survival_model = survival_model
        self.time_bins = time_bins

    def forward(self, clinical_feat=None, mri_feat=None, wsi_feat=None):
        device = next(self.parameters()).device
        features = []

        B = None

        if self.clinical_proj is not None:
            if clinical_feat is None:
                B = mri_feat.size(0) if mri_feat is not None else 1
                clinical_feat = torch.zeros(B, self.clinical_proj.proj[0].in_features, device=device)
            else:
                clinical_feat = clinical_feat.to(device)
            features.append(self.clinical_proj(clinical_feat))

        if self.mri_proj is not None:
            if mri_feat is None:
                B = clinical_feat.size(0) if clinical_feat is not None else 1
                mri_feat = torch.zeros(B, self.mri_proj.proj[0].in_features, device=device)
            else:
                mri_feat = mri_feat.to(device)
            features.append(self.mri_proj(mri_feat))

        if self.wsi_proj is not None:
            if wsi_feat is None:
                B = clinical_feat.size(0) if clinical_feat is not None else 1
                wsi_feat = torch.zeros(B, self.wsi_proj.proj[0].in_features, device=device)
            else:
                wsi_feat = wsi_feat.to(device)
            features.append(self.wsi_proj(wsi_feat))

        fused = self.fusion(features)
        return self.head(fused)

# def infer_time(model, mri_feat, clinical_feat):
#     model.eval()
#     with torch.no_grad():
#         preds = model(clinical_feat=clinical_feat, mri_feat=mri_feat)
#         if isinstance(model.head, DeepHitHead):
#             survival_time = torch.sum(preds * torch.arange(1, preds.shape[1] + 1, device=preds.device), dim=1)
#             return survival_time
#         else:
#             return preds.squeeze()
