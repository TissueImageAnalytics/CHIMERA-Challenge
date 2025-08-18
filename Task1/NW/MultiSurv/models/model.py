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


# ------------------ Fusion Strategies ------------------ #

class GatedCrossModalFusion(nn.Module):
    """
    Combines sample-specific gating + cross-modal attention.
    Clinical is used as the primary query modality.
    """
    def __init__(self, input_dims, hidden_dim=128):
        super().__init__()
        self.num_modalities = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Gating per modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for d in input_dims
        ])
        
        # Cross-modal attention (Clinical as Query)
        self.query_proj = nn.Linear(input_dims[0], hidden_dim)  # Clinical
        self.key_proj = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in input_dims[1:]
        ])
        self.value_proj = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in input_dims[1:]
        ])

    def forward(self, features):
        # Step 1: Gating
        gated = [f * g(f) for f, g in zip(features, self.gates)]
        
        # Step 2: Clinical as Query for Cross-Modal Attention
        clinical = gated[0]
        query = self.query_proj(clinical).unsqueeze(1)  # [B, 1, H]

        keys = [proj(f) for proj, f in zip(self.key_proj, gated[1:])]
        values = [proj(f) for proj, f in zip(self.value_proj, gated[1:])]
        
        if len(keys) > 0:
            keys = torch.stack(keys, dim=1)      # [B, N_other, H]
            values = torch.stack(values, dim=1)  # [B, N_other, H]
            
            attn = torch.softmax(
                (query @ keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5), dim=-1
            )  # [B, 1, N_other]
            fused_other = (attn @ values).squeeze(1)  # [B, H]
            
            return torch.cat([clinical, fused_other], dim=1)  # [B, 128+H]
        else:
            return clinical

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

class LinearFusion(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        total_dim = sum(input_dims)
        self.fc = nn.Linear(total_dim, max(input_dims))

    def forward(self, features):
        concat = torch.cat(features, dim=1)
        return self.fc(concat)


class SimpleConcatFusion(nn.Module):
    def forward(self, features):
        return torch.cat(features, dim=1)

class GuidedModalityFusion(nn.Module):
    """
    Fusion where each modality predicts its own risk,
    and those predictions are used to weight the features in fusion.
    """
    def __init__(self, input_dims, hidden_dim=128):
        super().__init__()
        self.num_modalities = len(input_dims)
        
        # Risk heads: MLPs for each modality
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for d in input_dims
        ])
        
        # Optional final fusion FC
        self.fuse_fc = nn.Linear(input_dims[0], hidden_dim)  # you can change this to sum(input_dims)

    def forward(self, features):
        # Step 1: compute modality-specific risk scores
        risks = [head(f) for head, f in zip(self.risk_heads, features)]  # [B,1] each
        risk_scores = torch.cat(risks, dim=1)  # [B, num_modalities]

        # Step 2: convert to independent weights via sigmoid
        weights = torch.sigmoid(risk_scores)  # [B, num_modalities]

        # Step 3: weight features
        weighted_features = [
            f * weights[:, i].unsqueeze(1) for i, f in enumerate(features)
        ]

        # Step 4: sum fuse and project
        fused = torch.stack(weighted_features, dim=1).sum(dim=1)  # [B, D]
        fused = self.fuse_fc(fused)
        return fused

# ------------------ Survival Heads ------------------ #

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


# ------------------ Multimodal Model ------------------ #

class MultimodalSurvivalModel(nn.Module):
    def __init__(self, clin_dim, mri_dim, wsi_dim, fusion_type='modality', survival_model='cox', time_bins=30):
        super().__init__()

        self.clinical_proj = ProjectionHead(clin_dim, HIDDEN_DIM) if clin_dim > 0 else None
        self.mri_proj = ProjectionHead(mri_dim, HIDDEN_DIM) if mri_dim > 0 else None
        self.wsi_proj = ProjectionHead(wsi_dim, HIDDEN_DIM) if wsi_dim > 0 else None

        input_dims = []
        input_dims = []
        if self.clinical_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.mri_proj is not None:
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
            if len(input_dims) < 2:
                # Fallback: no cross-att possible, just use identity or linear
                print("Error: CrossAttentionFusion requires >= 2 modalities. Please use linear or simple.")
                exit()
            else:
                self.fusion = CrossAttentionFusion(input_dims)
                fusion_dim = HIDDEN_DIM

        else:
            raise ValueError(f"Unknown fusion_type {fusion_type}")


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