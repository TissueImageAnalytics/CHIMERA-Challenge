# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

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
    
## Gated, instance-wise modality fusion
class GatedModalityFusion(nn.Module):
    """
    For each modality we learn a small gating MLP that outputs a scalar in [0,1]
    per sample. The fused vector is sum_i gate_i(x_i) * x_i.
    input_dims: list of per-modality feature dims (after projection)
    """
    def __init__(self, input_dims, hidden_gate_dim=64):
        super().__init__()
        self.num_modalities = len(input_dims)
        # Create one small gate MLP per modality
        self.gates = nn.ModuleList()
        for d in input_dims:
            # small two-layer gate: d -> hidden_gate_dim -> 1 (sigmoid)
            gate = nn.Sequential(
                nn.Linear(d, max(d // 2, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(d // 2, 8), 1),
                nn.Sigmoid()
            )
            self.gates.append(gate)

    def forward(self, features):
        # features: list of tensors [B, D]
        gated = []
        # optional: collect gate values for monitoring if needed
        for feat, gate_net in zip(features, self.gates):
            g = gate_net(feat)  # [B,1]
            gated.append(feat * g)  # [B,D]
        fused = torch.stack(gated, dim=1).sum(dim=1)
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
        #B = features[0].size(0)
        N = len(features)

        #keys_values = torch.stack(features, dim=1)  # [B, N, D]
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
    def __init__(self, clin_dim, rna_dim, wsi_dim, fusion_type='modality', survival_model='cox', time_bins=30):
        super().__init__()

        self.clinical_proj = ProjectionHead(clin_dim, HIDDEN_DIM) if clin_dim > 0 else None
        self.rna_proj = ProjectionHead(rna_dim, HIDDEN_DIM) if rna_dim > 0 else None
        self.wsi_proj = ProjectionHead(wsi_dim, HIDDEN_DIM) if wsi_dim > 0 else None

        input_dims = []
        if self.clinical_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.rna_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.wsi_proj is not None:
            input_dims.append(HIDDEN_DIM)

        # Fusion selection
        if fusion_type == 'simple':
            self.fusion = SimpleConcatFusion()
            fusion_dim = sum(input_dims)
        elif fusion_type == 'linear':
            self.fusion = LinearFusion(input_dims)
            fusion_dim = HIDDEN_DIM
        elif fusion_type == 'cross_att':
            self.fusion = CrossAttentionFusion(input_dims)
            fusion_dim = HIDDEN_DIM
        elif fusion_type == 'gated':
            # instance-wise learned gates per modality
            self.fusion = GatedModalityFusion(input_dims)
            fusion_dim = HIDDEN_DIM
        else:
            raise ValueError(f"Unknown fusion_type {fusion_type}")

        if survival_model == 'cox':
            self.head = CoxHead(fusion_dim)
        else:
            self.head = DeepHitHead(fusion_dim, time_bins)

        self.survival_model = survival_model
        self.time_bins = time_bins

    def forward(self, clinical_feat=None, rna_feat=None, wsi_feat=None):
        device = next(self.parameters()).device
        features = []

        B = None

        if self.clinical_proj is not None:
            if clinical_feat is None:
                B = rna_feat.size(0) if rna_feat is not None else 1
                clinical_feat = torch.zeros(B, self.clinical_proj.proj[0].in_features, device=device)
            else:
                clinical_feat = clinical_feat.to(device)
            features.append(self.clinical_proj(clinical_feat))

        if self.rna_proj is not None:
            if rna_feat is None:
                B = clinical_feat.size(0) if clinical_feat is not None else 1
                rna_feat = torch.zeros(B, self.rna_proj.proj[0].in_features, device=device)
            else:
                rna_feat = rna_feat.to(device)
            features.append(self.rna_proj(rna_feat))

        if self.wsi_proj is not None:
            if wsi_feat is None:
                B = clinical_feat.size(0) if clinical_feat is not None else 1
                wsi_feat = torch.zeros(B, self.wsi_proj.proj[0].in_features, device=device)
            else:
                wsi_feat = wsi_feat.to(device)
            features.append(self.wsi_proj(wsi_feat))

        fused = self.fusion(features)
        return self.head(fused)

## modality confidence network
class dis_MultimodalSurvivalModel(nn.Module):
    def __init__(self, clin_dim, rna_dim, wsi_dim, fusion_type='linear', survival_model='cox', time_bins=30):
        super().__init__()

        self.clinical_proj = ProjectionHead(clin_dim, HIDDEN_DIM) if clin_dim > 0 else None
        self.rna_proj = ProjectionHead(rna_dim, HIDDEN_DIM) if rna_dim > 0 else None
        self.wsi_proj = ProjectionHead(wsi_dim, HIDDEN_DIM) if wsi_dim > 0 else None

        # Modality-specific heads
        if survival_model == 'cox':
            self.clinical_head = CoxHead(HIDDEN_DIM) if self.clinical_proj is not None else None
            self.rna_head = CoxHead(HIDDEN_DIM) if self.rna_proj is not None else None
            self.wsi_head = CoxHead(HIDDEN_DIM) if self.wsi_proj is not None else None
        else:
            self.clinical_head = DeepHitHead(HIDDEN_DIM, time_bins) if self.clinical_proj is not None else None
            self.rna_head = DeepHitHead(HIDDEN_DIM, time_bins) if self.rna_proj is not None else None
            self.wsi_head = DeepHitHead(HIDDEN_DIM, time_bins) if self.wsi_proj is not None else None

        # Confidence networks per modality (small MLP outputting [0,1])
        def make_confidence_net():
            return nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM//2),
                nn.ReLU(inplace=True),
                nn.Linear(HIDDEN_DIM//2, 1),
                nn.Sigmoid()
            )
        self.clinical_confidence_net = make_confidence_net() if self.clinical_proj is not None else None
        self.rna_confidence_net = make_confidence_net() if self.rna_proj is not None else None
        self.wsi_confidence_net = make_confidence_net() if self.wsi_proj is not None else None

        self.survival_model = survival_model
        self.time_bins = time_bins

    def forward(self, clinical_feat=None, rna_feat=None, wsi_feat=None):
        device = next(self.parameters()).device

        B = None
        clinical_risk = None
        rna_risk = None
        wsi_risk = None
        clinical_conf = None
        rna_conf = None
        wsi_conf = None

        # Clinical branch
        if self.clinical_proj is not None:
            if clinical_feat is None:
                B = wsi_feat.size(0) if wsi_feat is not None else 1
                clinical_feat = torch.zeros(B, self.clinical_proj.proj[0].in_features, device=device)
            else:
                clinical_feat = clinical_feat.to(device)
            clinical_emb = self.clinical_proj(clinical_feat)
            clinical_risk = self.clinical_head(clinical_emb)  # [B, 1] or [B, time_bins]
            clinical_conf = self.clinical_confidence_net(clinical_emb)  # [B,1]

        # RNA branch
        if self.rna_proj is not None:
            if rna_feat is None:
                B = rna_feat.size(0) if rna_feat is not None else 1
                rna_feat = torch.zeros(B, self.rna_proj.proj[0].in_features, device=device)
            else:
                rna_feat = rna_feat.to(device)
            rna_emb = self.rna_proj(rna_feat)
            rna_risk = self.rna_head(rna_emb)  # [B,1] or [B, time_bins]
            rna_conf = self.rna_confidence_net(rna_emb)  # [B,1]

        # WSI branch
        if self.wsi_proj is not None:
            if wsi_feat is None:
                B = clinical_feat.size(0) if clinical_feat is not None else 1
                wsi_feat = torch.zeros(B, self.wsi_proj.proj[0].in_features, device=device)
            else:
                wsi_feat = wsi_feat.to(device)
            wsi_emb = self.wsi_proj(wsi_feat)
            wsi_risk = self.wsi_head(wsi_emb)  # [B,1] or [B, time_bins]
            wsi_conf = self.wsi_confidence_net(wsi_emb)  # [B,1]

        # Stack risks and confidences (only existing modalities)
        risks = []
        confs = []

        if clinical_risk is not None:
            risks.append(clinical_risk)
            confs.append(clinical_conf)
        if rna_risk is not None:
            risks.append(rna_risk)
            confs.append(rna_conf)
        if wsi_risk is not None:
            risks.append(wsi_risk)
            confs.append(wsi_conf)

        risks = torch.stack(risks, dim=1)  # [B, num_modalities, ...]
        confs = torch.stack(confs, dim=1)  # [B, num_modalities, 1]

        # Normalize confidence weights to sum to 1 per sample
        confs_norm = confs / (confs.sum(dim=1, keepdim=True) + 1e-8)  # [B, num_modalities, 1]

        # Weighted sum of risks
        # For Cox: risks shape = [B, num_mod, 1], output shape [B,1]
        # For DeepHit: risks shape = [B, num_mod, time_bins], weighted sum along dim=1
        fused_risk = (risks * confs_norm).sum(dim=1)

        return fused_risk
