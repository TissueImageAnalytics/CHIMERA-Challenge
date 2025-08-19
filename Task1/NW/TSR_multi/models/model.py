# (Full file — updated TransductiveSR with modality fusion)
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sksurv.metrics import concordance_index_censored
from config import *

USE_CUDA = torch.cuda.is_available()
from torch.autograd import Variable

def cuda(v, device=None):
    if USE_CUDA:
        if device is None:
            return v.cuda()
        else:
            return v.to(device)
    return v

def toTensor(v, dtype=torch.float, requires_grad=False, device=None):
    tensor = torch.tensor(v, dtype=dtype)
    var = Variable(tensor, requires_grad=requires_grad)
    return cuda(var, device=device)

def toNumpy(v):
    return v.detach().cpu().numpy()

def TransductiveLoss(z):
    return torch.exp(-3 * (z ** 2))

def sksurv_cox(y_pred: torch.Tensor,
                event:    torch.Tensor,
                time:     torch.Tensor) -> torch.Tensor:

    n_samples = event.shape[0]
    loss = 0

    for i in range(n_samples):
        at_risk = 0
        for j in range(n_samples):
            if time[j] >= time[i]:
                at_risk += torch.exp(y_pred[j])
        loss += event[i] * (y_pred[i] - torch.log(at_risk))

    return - loss

def cox_ph_loss(log_risk: torch.Tensor,
                event:    torch.Tensor,
                time:     torch.Tensor) -> torch.Tensor:
    order     = torch.argsort(time, descending=True)
    log_risk  = log_risk[order]
    event     = event[order]

    # cumulative log-sum-exp of exp(log_risk) over the *prefix* j>=i
    log_cumsum_hazard = torch.logcumsumexp(log_risk, dim=0)

    pll = (log_risk - log_cumsum_hazard) * event          # only uncensored contribute
    return -pll.sum() / event.sum()                       # mean over events

def _partial_likelihood_breslow(
    predictions: torch.Tensor,
    E: torch.Tensor,
    T: torch.Tensor

):
    """Calculate the partial log likelihood for the Cox proportional hazards model
    using Breslow's method to handle ties in event time.
    """

    sorted_indices = torch.argsort(T)

    # Step 2: Apply the sorting to T, predictions, and E
    time_sorted = T[sorted_indices]
    event_sorted = E[sorted_indices]
    log_hz_sorted = predictions[sorted_indices]

    N = len(time_sorted)

    R = [torch.where(time_sorted >= time_sorted[i])[0] for i in range(N)]
    log_denominator = torch.tensor(
        [torch.logsumexp(log_hz_sorted[R[i]], dim=0) for i in range(N)]
    )

    event_mask = event_sorted == 1  # Creates a boolean mask
    return torch.mean( (log_hz_sorted.squeeze() - log_denominator.cuda())[event_mask] )


# New model structure with latent layer
class SurvivalNet(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(SurvivalNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        score = self.head(z).squeeze()
        return score, z,0

# Head to reduce dimensionality
class DimReducer(nn.Module):
    def __init__(self, input_dim,dropout=0.0, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
# Head for survival ranking (structure is the same as the OG TSR)
class RankingHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, z):
        return self.activation(self.linear(z)).squeeze()

class RankModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32,dropout=0.0):
        super().__init__()
        self.encoder = DimReducer(input_dim, latent_dim=latent_dim,dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        self.head = RankingHead(latent_dim)

    def forward(self, x):
        z = self.encoder(x)        # Φ(x)
        x_recon = self.decoder(z)
        score = self.head(z)       # tanh(wᵀΦ(x))
        return score, z, x_recon
    
class SimpleSurv(nn.Module):
    """
    A very small Cox network that matches the structure you used before:
        Linear → (optional) Tanh → (optional) Dropout → 1-D log-risk score.
    Setting `bounded=False` removes the tanh, giving the classic linear Cox head.
    """
    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()

        self.input_layer = nn.Linear(in_features, 1, bias=True)
        layers = [self.input_layer]
        layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a 1-D tensor of log-risk scores with shape (batch,).
        """
        return self.net(x).squeeze(-1),0,0

class DeepSurvNet(nn.Module):
    """
    DeepSurv-style MLP
    ------------------------------------------------------------
    Input  ➜  ( Linear → Activation → Dropout → [BatchNorm] ) × L
           ➜  Linear(1)                    # log-risk score
    ------------------------------------------------------------
    Parameters
    ----------
    in_features : int
        Dimensionality of input vector (e.g. 768 for WSI embeddings).
    hidden      : list[int] | tuple[int]
        Sizes of hidden layers.  Example: [256, 128].
    act         : str
        'relu' (default) or 'selu'.  ReLU matches the original paper.
    p_dropout   : float
        Dropout probability applied *after* each activation.
    use_bn      : bool
        Insert BatchNorm1d after each dropout (optional).
    bounded     : bool
        If True, applies a final tanh to bound risk scores in [-1, 1]
        (not in the original DeepSurv, but matches your old code).
    """

    def __init__(
        self,
        in_features: int,
        hidden      = (256, 128),
        act: str    = "relu",
        p_dropout:  float = 0.3,
        use_bn:     bool  = False,
        bounded:    bool  = True
    ):
        super().__init__()

        act_layer = nn.ReLU if act.lower() == "relu" else nn.SELU
        layers    = []

        # ---------- first (input) layer -----------------------------------
        self.input_layer = nn.Linear(in_features, hidden[0])
        layers += [self.input_layer, act_layer(inplace=True), nn.Dropout(p_dropout)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden[0]))

        # ---------- additional hidden layers ------------------------------
        last_dim = hidden[0]
        for h in hidden[1:]:
            layers += [
                nn.Linear(last_dim, h),
                act_layer(inplace=True),
                nn.Dropout(p_dropout)
            ]
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            last_dim = h

        # ---------- output head -------------------------------------------
        self.output_layer = nn.Linear(last_dim, 1)   # linear log-risk
        layers.append(self.output_layer)

        if bounded:               # optional tanh (keeps your old behaviour)
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a 1-D tensor of log-risk scores with shape (batch,).
        """
        return self.net(x).squeeze(-1),0,0


# ----------------- Fusion utilities -------------
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
        # features: list of tensors [B, D]
        stacked = torch.stack(features, dim=1)  # [B, N_modalities, D]
        weights = torch.softmax(self.weights, dim=0)  # [N_modalities]
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
        return fused

## linear layer after concat
class LinearFusion(nn.Module):
    def __init__(self, input_dims, out_dim=None):
        super().__init__()
        total_dim = sum(input_dims)
        out_dim = out_dim or max(input_dims)
        self.fc = nn.Linear(total_dim, out_dim)

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

# --------------------------------------------------------------------------------------

class TransductiveSR:
    def __init__(self, model=None,structure='SurvivalNet', lambda_w=0.1, lambda_u=0.0, p=2,
                  lr=1e-2, Tmax=200,dropout=0.2, latent_dim=32, loss_type='ranking',
                  # new modality/fusion args:
                  clin_dim=0, mri_dim=0, wsi_dim=0, device='cpu'):
        self.lambda_w = lambda_w
        self.lambda_u = lambda_u
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        self.latent_dim = latent_dim
        self.model = model
        self.structure = structure
        self.dropout = dropout
        self.loss_type = loss_type
        self.patience = 300

        # modality info
        self.clin_in_dim = int(clin_dim)
        self.wsi_in_dim = int(wsi_dim)
        self.mri_in_dim = int(mri_dim)
        self.device = device

        # build projection heads & fusion module if any modality dims provided
        self.clinical_proj = ProjectionHead(self.clin_in_dim, HIDDEN_DIM) if self.clin_in_dim > 0 else None
        self.wsi_proj = ProjectionHead(self.wsi_in_dim, HIDDEN_DIM) if self.wsi_in_dim > 0 else None
        self.mri_proj = ProjectionHead(self.mri_in_dim, HIDDEN_DIM) if self.mri_in_dim > 0 else None

        input_dims = []
        if self.clinical_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.mri_proj is not None:
            input_dims.append(HIDDEN_DIM)
        if self.wsi_proj is not None:
            input_dims.append(HIDDEN_DIM)

        if FUSION_TYPE == 'simple':
            self.fusion = SimpleConcatFusion()
            self.fusion_dim = sum(input_dims)
        elif FUSION_TYPE == 'linear':
            self.fusion = LinearFusion(input_dims)
            self.fusion_dim = HIDDEN_DIM
        elif FUSION_TYPE == 'cross_att':
            if len(input_dims) < 2:
                # Fallback: no cross-att possible, just use identity or linear
                print("Error: CrossAttentionFusion requires >= 2 modalities. Please use linear or simple.")
                exit()
            else:
                self.fusion = CrossAttentionFusion(input_dims)
                self.fusion_dim = HIDDEN_DIM
        else:
            raise ValueError(f"Unknown fusion_type {FUSION_TYPE}")

    def _fuse_numpy(self, X_np):
        if self.fusion is None:
            return X_np

        # Split numpy input into parts
        idx = 0
        parts = {}
        if self.clin_in_dim > 0:
            parts['clin'] = X_np[:, idx: idx + self.clin_in_dim]
            idx += self.clin_in_dim
        else:
            parts['clin'] = None

        if self.wsi_in_dim > 0:
            parts['wsi'] = X_np[:, idx: idx + self.wsi_in_dim]
            idx += self.wsi_in_dim
        else:
            parts['wsi'] = None

        if self.mri_in_dim > 0:
            parts['mri'] = X_np[:, idx: idx + self.mri_in_dim]
            idx += self.mri_in_dim
        else:
            parts['mri'] = None

        features = []
        # Clinical
        if self.clinical_proj is not None:
            x_clin = torch.tensor(parts['clin'] if parts['clin'] is not None else
                                np.zeros((X_np.shape[0], self.clin_in_dim)),
                                dtype=torch.float32, device=self.device)
            features.append(self.clinical_proj(x_clin))
        # MRI
        if self.mri_proj is not None:
            x_mri = torch.tensor(parts['mri'] if parts['mri'] is not None else
                                np.zeros((X_np.shape[0], self.mri_in_dim)),
                                dtype=torch.float32, device=self.device)
            features.append(self.mri_proj(x_mri))
        # WSI
        if self.wsi_proj is not None:
            x_wsi = torch.tensor(parts['wsi'] if parts['wsi'] is not None else
                                np.zeros((X_np.shape[0], self.wsi_in_dim)),
                                dtype=torch.float32, device=self.device)
            features.append(self.wsi_proj(x_wsi))

        fused_t = self.fusion(features)
        return toNumpy(fused_t)


    def fit(self, X_train, T_train, E_train, X_test=None,plot_loss=False,val_frac=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Move projection heads and fusion module to device BEFORE fusion
        if self.clinical_proj is not None:
            self.clinical_proj.to(self.device)
        if self.mri_proj is not None:
            self.mri_proj.to(self.device)
        if self.wsi_proj is not None:
            self.wsi_proj.to(self.device)
        if self.fusion is not None:
            self.fusion.to(self.device)

        # Step 2: Fuse inputs (now device matches projection heads and fusion)
        if self.fusion is not None:
            X_train_fused = self._fuse_numpy(X_train)
            if X_test is not None:
                X_test_fused = self._fuse_numpy(X_test)
            else:
                X_test_fused = None
            x = toTensor(X_train_fused, device=self.device)
            if X_test_fused is not None:
                X_test = toTensor(X_test_fused, device=self.device)
        else:
            x = toTensor(X_train, device=self.device)
            if X_test is not None:
                X_test = toTensor(X_test, device=self.device)

        T = toTensor(T_train, device=self.device)
        E = toTensor(E_train, device=self.device)

        # ---------- 1. validation split ----------------------------------------
        N = len(x)
        rng = np.random.default_rng(seed=42)
        idx  = np.arange(N)
        # stratified on event flag (approx.)
        pos  = idx[E_train == 1]
        neg  = idx[E_train == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_val_pos = int(np.ceil(len(pos) * val_frac))
        n_val_neg = int(np.ceil(len(neg) * val_frac))
        val_idx   = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
        trn_idx   = np.setdiff1d(idx, val_idx)

        x_tr, T_tr, E_tr = x[trn_idx], T[trn_idx], E[trn_idx]
        x_val, T_val, E_val = x[val_idx], T[val_idx], E[val_idx]

        # ---------- 2. model ----------------------------------------------------
        D_in = x.shape[1]
        if self.model is None:
            if self.structure == 'RankModel':
                self.model = RankModel(D_in, self.latent_dim,self.dropout)
            if self.structure == 'DeepSurv':
                self.model = DeepSurvNet(in_features=D_in, p_dropout=self.dropout)
            elif self.structure == 'SurvivalNet':
                self.model = SurvivalNet(D_in, latent_dim=64)
            elif self.structure == 'SimpleSurv':
                self.model = SimpleSurv(D_in, dropout=self.dropout)
        model = cuda(self.model)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0)
        self.bias = 0
        # ---------- 3. bookkeeping ---------------------------------------------
        L_tr, L_val, C_val = [], [], []
        best_c, best_state = -np.inf, None

        if plot_loss:
            fig, (ax_loss, ax_cidx) = plt.subplots(
                2, 1, figsize=(15, 8), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]}
            )
            line_tr,   = ax_loss.plot([], [], label='Train loss')
            line_val,  = ax_loss.plot([], [], label='Val loss')
            line_cidx, = ax_cidx.plot([], [], color='green', label='Val C-index')

            ax_loss.set_ylabel('Loss')
            ax_cidx.set_ylabel('C-index')
            ax_cidx.set_xlabel('Epoch')

            ax_loss.legend(loc='upper right')
            ax_cidx.legend(loc='lower right')
            plt.tight_layout()

        patience   = self.patience          # epochs to wait after last improvement
        min_delta  = 1e-3        # smallest C-index gain to count
        wait       = 0           # epochs since last improvement
        best_epoch = -1

        # ---------- 4. training loop -------------------------------------------
        for epoch in range(self.Tmax):
            model.train()
            optimizer.zero_grad()

            # ---- forward ------------------------------------------------------
            y_hat_tr, _, _ = model(x_tr)

            # ---- supervised loss ---------------------------------------------
            if self.loss_type == 'ranking':
                dT = T_tr[:, None] - T_tr[None, :]
                dP = ((dT > 0) * E_tr).bool()
                dZ = (y_hat_tr.unsqueeze(1) - y_hat_tr)[dP]
                loss = torch.mean(torch.clamp(1.0 - dZ, min=0))
            else:  # Cox
                loss = _partial_likelihood_breslow(y_hat_tr, E_tr, T_tr)

            # ---- transductive penalty ----------------------------------------
            if X_test is not None and self.lambda_u > 0:
                y_hat_test, _,_ = model(X_test)
                loss_u = TransductiveLoss(y_hat_test).mean()
                loss += self.lambda_u * loss_u

            # ---- weight regularisation ---------------------------------------
            if self.structure in ['DeepSurv', 'SimpleSurv']:
                w = model.input_layer.weight.view(-1)
            elif self.structure == 'RankModel':
                w= model.head.linear.weight.view(-1)
            else:  # SurvivalNet
                w = model.encoder[0].weight.view(-1)
            loss += self.lambda_w * torch.norm(w, self.p) ** self.p

            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Train Loss: {loss}")

            # ---- back-prop ----------------------------------------------------
            loss.backward()
            optimizer.step()
            L_tr.append(loss.item())

            # ---------- 5. validation -----------------------------------------
            model.eval()
            with torch.no_grad():
                y_hat_val, _,_ = model(x_val)

                # validation loss (same definition)
                if self.loss_type == 'ranking':
                    dT = T_val[:, None] - T_val[None, :]
                    dP = ((dT > 0) * E_val).bool()
                    dZ = (y_hat_val.unsqueeze(1) - y_hat_val)[dP]
                    val_loss = torch.mean(torch.clamp(1.0 - dZ, min=0))
                else:
                    val_loss = _partial_likelihood_breslow(y_hat_val, E_val, T_val)
                
                if X_test is not None and self.lambda_u > 0:
                    y_hat_test, _,_ = model(X_test)
                    loss_u = TransductiveLoss(y_hat_test).mean()
                    loss += self.lambda_u * loss_u

                # ---- weight regularisation ---------------------------------------
                if self.structure in ['DeepSurv', 'SimpleSurv']:
                    w = model.input_layer.weight.view(-1)
                elif self.structure == 'RankModel':
                    w= model.head.linear.weight.view(-1)
                else:  # SurvivalNet
                    w = model.encoder[0].weight.view(-1)
                loss += self.lambda_w * torch.norm(w, self.p) ** self.p
                
                L_val.append(val_loss.item())

                # validation C-index
                event_indicator = toNumpy(E_val).astype(bool)
                cidx,_,_,_,_ = concordance_index_censored(event_indicator, toNumpy(T_val), -toNumpy(y_hat_val))
                C_val.append(cidx)

                if epoch % 50 == 0:
                    print(f"Epoch {epoch} | Val Loss: {loss} | Val C-index: {cidx}")

                # keep best model
                improved = cidx - best_c > min_delta
                if improved:
                    best_c      = cidx
                    best_state  = deepcopy(model.state_dict())
                    best_epoch  = epoch
                    wait        = 0          # reset patience
                else:
                    wait += 1

                # stop if no improvement for 'patience' epochs
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}. "
                          f"Best C-index {best_c:.3f} at epoch {best_epoch+1}.")
                    break

            # ---------- 6. live plot ------------------------------------------
            if plot_loss:
                epochs_x = range(1, len(L_tr) + 1)
                line_tr.set_data(epochs_x, L_tr)
                line_val.set_data(epochs_x, L_val)
                line_cidx.set_data(epochs_x, C_val)

                # rescale axes
                for ax in (ax_loss, ax_cidx):
                    ax.relim()
                    ax.autoscale_view()

                ax_loss.set_title(
                    f'Epoch {epoch+1}/{self.Tmax} — best C-index {best_c:.3f}'
                )
                plt.pause(0.01)

        # ---------- 7. restore best weights ------------------------------------
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---------- 8. stash & return ------------------------------------------
        self.model = model
        self.L_train = L_tr
        self.L_val   = L_val
        self.C_val   = C_val
        self.best_c  = best_c
        return model
    
    def decision_function(self, x, return_latent=False):
        # accepts same concatenated X layout as training (clin + mri + wsi),
        # will fuse if fusion module is set up.
        if self.fusion is not None:
            x_fused = self._fuse_numpy(x)
            x = toTensor(x_fused)
        else:
            x = toTensor(x)

        with torch.no_grad():
            score, z,_ = self.model(x)
        if return_latent:
            return toNumpy(score - self.bias).flatten(), toNumpy(z)
        return toNumpy(score - self.bias).flatten()

    def getW(self):
        return toNumpy(self.w / torch.linalg.norm(self.w, ord=1))

    def get_latent(self, X):
        # X given as concatenated modalities
        if self.fusion is not None:
            X_fused = self._fuse_numpy(X)
            x = toTensor(X_fused)
        else:
            x = toTensor(X)
        with torch.no_grad():
            _, z, _ = self.model(x)
        return toNumpy(z)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        # x is concatenated modalities
        if self.fusion is not None:
            x_fused = self._fuse_numpy(x)
            x = toTensor(x_fused)
        else:
            x = toTensor(x)

        self.model.train()  # Ensure dropout is active
        predictions = []
        for _ in range(n_samples):
            preds,_,_ = self.model(x)
            preds=preds.detach().cpu().numpy()
            predictions.append(preds)
        predictions = np.array(predictions)
        mean_preds = predictions.mean(axis=0)
        std_preds = predictions.std(axis=0)
        return mean_preds.flatten(), std_preds.flatten()