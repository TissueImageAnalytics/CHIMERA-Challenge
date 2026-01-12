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
            nn.Dropout(PROJECTION_DROPOUT)
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

# === Multimodal Survival Model ===
class MultimodalSurvivalModel(nn.Module):
    def __init__(
        self,
        clin_dim=None,
        rna_dim=None,
        wsi_dim=None,
    ):
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

        # Fusion strategy
        if FUSION_TYPE == "concat":
            fusion_dim = (HIDDEN_DIM if (clin_dim and clin_dim > 0) else 0) \
                    + (HIDDEN_DIM if (rna_dim and rna_dim > 0) else 0) \
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

    def forward(self, clinical_feat=None, rna_feat=None, wsi_feat=None):
        features = []
        if clinical_feat is not None and self.clinical_proj is not None:
            features.append(self.clinical_proj(clinical_feat))
        if rna_feat is not None and self.rna_proj is not None:
            features.append(self.rna_proj(rna_feat))
        if wsi_feat is not None and self.wsi_proj is not None:
            features.append(self.wsi_proj(wsi_feat))

        if not features:
            raise ValueError("No modalities provided.")

        fused = torch.cat(features, dim=1) if isinstance(self.fusion, nn.Identity) else self.fusion(features)

        # Dropout *after fusion*
        fused = F.dropout(fused, p=self.fusion_dropout, training=self.training)

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
