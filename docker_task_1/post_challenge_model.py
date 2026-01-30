# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from post_challenge_config import *

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(PROJECTION_DROPOUT)
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

class CrossModalBlock(nn.Module):
    """Cross-attention block with residuals, FFN, and LayerNorm."""
    def __init__(self, dim, num_heads=4, ff_mult=4):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Residual 1
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_mult * dim, dim)
        )
        
        # Residual 2
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # Attention: query attends to key/value
        attn_out, _ = self.attn(query, key_value, key_value)
        
        # Residual + Norm
        x = self.norm1(query + attn_out)
        
        # FFN + Residual + Norm
        x = self.norm2(x + self.ff(x))
        
        return x

class CrossAttentionFusion(nn.Module):
    """
    Generic cross-attention fusion for N modalities.
    - Expects input `features` as list of tensors [B, D] (D == dim).
    - Each modality queries the other modalities (no self-attention).
    - Uses CrossModalBlock for attention + FFN + residuals.
    - Computes sample-specific gating weights from attended vectors.
    - Returns fused [B, dim].
    """
    def __init__(self, dim=128, num_modalities=2, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        # one CrossModalBlock per modality (query block)
        self.blocks = nn.ModuleList([
            CrossModalBlock(dim, num_heads=num_heads) for _ in range(num_modalities)
        ])

        # gating network: from concatenated attended features -> softmax weights over modalities
        self.gate = nn.Sequential(
            nn.Linear(num_modalities * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_modalities),
            nn.Softmax(dim=-1)
        )

        # final projection (optional)
        self.proj = nn.Linear(dim, dim)

    def forward(self, features):
        """
        features: list of length N, each tensor of shape [B, dim]
        returns: fused tensor [B, dim]
        """
        if not isinstance(features, (list, tuple)):
            raise ValueError("CrossAttentionFusion expects a list of modality tensors")

        N = len(features)
        if N != self.num_modalities:
            # allow flexible behavior: if number differs, adapt by rebuilding blocks? here assert
            # simpler: allow runtime mismatch by using min(self.num_modalities, N)
            # but safer to raise so initialization matches usage.
            raise ValueError(f"Expected {self.num_modalities} modalities, got {N}")

        B = features[0].size(0)
        # Convert each to sequence format [B, 1, D]
        seq = [f.unsqueeze(1) for f in features]  # list of [B,1,D]

        attended = []
        # For each modality i, make it query the remaining modalities (stack keys/values)
        for i in range(N):
            query = seq[i]  # [B,1,D]
            # collect keys/values from j != i
            kv_list = [seq[j] for j in range(N) if j != i]
            if len(kv_list) == 0:
                # single modality -> identity
                attended.append(query.squeeze(1))
                continue
            kv = torch.cat(kv_list, dim=1)  # [B, N-1, D] (concatenate along sequence dim)
            # Use the i-th block: query attends to kv
            out = self.blocks[i](query, kv)  # returns [B,1,D]
            attended.append(out.squeeze(1))   # store [B,D]

        # Stack attended: [B, N, D]
        attended_stack = torch.stack(attended, dim=1)

        # Compute gating weights per sample from concatenated attended vectors
        concat_att = attended_stack.view(B, N * self.dim)  # [B, N*D]
        weights = self.gate(concat_att)  # [B, N], sums to 1 across modalities

        # Weighted sum across modalities
        weights = weights.unsqueeze(-1)  # [B, N, 1]
        fused = (attended_stack * weights).sum(dim=1)  # [B, D]

        fused = self.proj(fused)
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

class DeepSurvHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # must output scalar risk
        )

    def forward(self, x):
        out = self.mlp(x)
        assert out.shape[-1] == 1, f"Expected risk score shape [B,1], got {out.shape}"
        return out

# ------------------ Multimodal Model ------------------ #

# === Multimodal Survival Model ===
class MultimodalSurvivalModel(nn.Module):
    def __init__(
        self,
        clin_dim=None,
        mri_dim=None,
        wsi_dim=None,
    ):
        super().__init__()

        self.clinical_proj = ProjectionHead(clin_dim, HIDDEN_DIM) if clin_dim > 0 else None
        self.mri_proj = ProjectionHead(mri_dim, HIDDEN_DIM) if mri_dim > 0 else None
        self.wsi_proj = ProjectionHead(wsi_dim, HIDDEN_DIM) if wsi_dim > 0 else None

        input_dims = []
        if self.clinical_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.mri_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.wsi_proj is not None:
            input_dims.append(HIDDEN_DIM)

        # Fusion strategy
        if FUSION_TYPE == "concat":
            fusion_dim = (HIDDEN_DIM if (clin_dim and clin_dim > 0) else 0) \
                    + (HIDDEN_DIM if (mri_dim and mri_dim > 0) else 0) \
                    + (HIDDEN_DIM if (wsi_dim and wsi_dim > 0) else 0)
            self.fusion = nn.Identity()
        elif FUSION_TYPE == 'simple':
            self.fusion = SimpleConcatFusion()
            fusion_dim = sum(input_dims)
        elif FUSION_TYPE == "linear":
            fusion_dim = HIDDEN_DIM
            self.fusion = LinearFusion(input_dims)
        elif FUSION_TYPE == "cross_att":
            #fusion_dim = HIDDEN_DIM
            #self.fusion = CrossAttentionFusion(input_dims)
            self.fusion = CrossAttentionFusion(dim=HIDDEN_DIM, num_modalities=len(input_dims))
            fusion_dim = HIDDEN_DIM
        else:
            raise ValueError(f"Unknown fusion method: {FUSION_TYPE}")

        # Post-fusion dropout
        self.fusion_dropout = FUSION_DROPOUT

        if SURVIVAL_MODEL == 'cox':
            self.head = CoxHead(fusion_dim)
        elif SURVIVAL_MODEL == 'deepsurv':
            self.head = DeepSurvHead(fusion_dim)
        elif SURVIVAL_MODEL == 'deephit':
            self.head = DeepHitHead(fusion_dim, TIME_BINS)
        else:
            raise ValueError(f"Unknown survival model {SURVIVAL_MODEL}")

    def forward(self, clinical_feat=None, mri_feat=None, wsi_feat=None):
        features = []
        if clinical_feat is not None and self.clinical_proj is not None:
            features.append(self.clinical_proj(clinical_feat))
        if mri_feat is not None and self.mri_proj is not None:
            features.append(self.mri_proj(mri_feat))
        if wsi_feat is not None and self.wsi_proj is not None:
            features.append(self.wsi_proj(wsi_feat))

        if not features:
            raise ValueError("No modalities provided.")

        fused = torch.cat(features, dim=1) if isinstance(self.fusion, nn.Identity) else self.fusion(features)

        # Dropout *after fusion*
        fused = F.dropout(fused, p=self.fusion_dropout, training=self.training)

        return self.head(fused)