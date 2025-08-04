import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sksurv.metrics import concordance_index_censored
from config import *
import torch.nn.functional as F

# ===== Dataset for torch =====
class SurvivalDataset(Dataset):
    def __init__(self, clin_array, mri_array, wsi_array, durations, events):
        self.clin_array = clin_array
        self.mri_array = mri_array
        self.wsi_array = wsi_array
        self.durations = durations
        self.events = events

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, idx):
        clin = self.clin_array[idx] if self.clin_array is not None else np.zeros(0, dtype=np.float32)
        mri = self.mri_array[idx] if self.mri_array is not None else np.zeros(0, dtype=np.float32)
        wsi = self.wsi_array[idx] if self.wsi_array is not None else np.zeros(0, dtype=np.float32)
        duration = self.durations[idx]
        event = self.events[idx]
        return (
            torch.tensor(clin, dtype=torch.float32),
            torch.tensor(mri, dtype=torch.float32),
            torch.tensor(wsi, dtype=torch.float32),
            torch.tensor(duration, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
        )

# ===== Cox partial log-likelihood loss =====
def cox_loss(predictions, durations, events):
    hazards = predictions.squeeze()
    sorted_indices = torch.argsort(durations, descending=True)
    hazards = hazards[sorted_indices]
    events = events[sorted_indices]
    log_cum_sum_exp = torch.logcumsumexp(hazards, dim=0)
    losses = -hazards + log_cum_sum_exp
    losses = losses * events
    return losses.sum() / events.sum()

# ===== DeepHit loss =====
def deephit_loss_censored(pred, durations, events, alpha=0.5, time_bins=TIME_BINS, debug=False):
    """
    DeepHit loss combining likelihood loss and ranking loss.

    Args:
        pred: Tensor of shape [N, T], predicted PMFs over discrete time bins.
        durations: Tensor of shape [N], observed durations (event or censoring times).
        events: Tensor of shape [N], 1 if event observed, 0 if censored.
        alpha: float, weight between likelihood and ranking losses.
        time_bins: int, total number of discrete time bins.

    Returns:
        Scalar tensor with combined DeepHit loss.
    """
    device = pred.device
    N, T = pred.shape

    # Clamp durations to valid range of time bins
    durations = torch.clamp(durations.long(), max=time_bins - 1).to(device)
    events = events.bool().to(device)

    # Negative log likelihood part (cause-specific likelihood)
    idx = torch.arange(N, device=device)
    likelihood_loss = -torch.log(pred[idx, durations] + 1e-8)
    likelihood_loss = torch.mean(likelihood_loss[events])

    # Ranking loss part (to encourage correct ordering)
    # Construct pairwise comparisons for subjects with events
    event_idx = torch.where(events)[0]
    if len(event_idx) <= 1:
        # Not enough events to compute ranking loss, fallback to likelihood only
        return likelihood_loss

    # Compute risk scores as expected time
    time_range = torch.arange(T, device=device).float()
    risk_scores = (pred * time_range).sum(dim=1)

    rank_loss = 0.0
    count = 0

    for i in event_idx:
        for j in range(N):
            if durations[j] > durations[i]:  # j survived longer than i
                diff = risk_scores[j] - risk_scores[i]
                rank_loss += torch.exp(-diff)
                count += 1

    if count > 0:
        rank_loss = rank_loss / count
    else:
        rank_loss = 0.0

    # Combine losses
    total_loss = (1 - alpha) * likelihood_loss + alpha * rank_loss

    return total_loss

def safe_log(x, eps=1e-6):
    """Numerically stable log."""
    return torch.log(torch.clamp(x, min=eps))

def deephit_loss_uncensored(pred, durations, events, alpha=0.5, time_bins=TIME_BINS, debug=False):
    """
    Numerically stable DeepHit loss with both uncensored and censored components.

    Args:
        pred: Tensor of shape [N, T], predicted PMF over time bins.
        durations: Tensor of shape [N], observed durations (discrete indices).
        events: Tensor of shape [N], 1 if event occurred, 0 if censored.
        alpha: Float weight between likelihood and ranking loss.
        time_bins: Total number of discrete time bins.
        debug: If True, prints problematic values causing NaNs.

    Returns:
        Combined scalar loss.
    """
    device = pred.device
    N, T = pred.shape

    durations = torch.clamp(durations.long(), max=time_bins - 1).to(device)
    events = events.bool().to(device)
    idx = torch.arange(N, device=device)

    # ----------------------------------
    # Likelihood Loss (Stable)
    # ----------------------------------
    # Uncensored: log P(event at t)
    p_event = pred[idx, durations]
    if debug and torch.any(p_event <= 0):
        print("🔴 Invalid event probabilities:", p_event[p_event <= 0])
    uncensored_loss = -safe_log(p_event)
    uncensored_loss = uncensored_loss[events]

    # Censored: log(1 - F(t)) ≈ log(1 - CDF)
    cdf = torch.cumsum(pred, dim=1)
    survival_prob = torch.clamp(1.0 - cdf[idx, durations], min=1e-6)
    if debug and torch.any(survival_prob <= 0):
        print("🔴 Invalid survival probabilities:", survival_prob[survival_prob <= 0])
    censored_loss = -safe_log(survival_prob)
    censored_loss = censored_loss[~events]

    likelihood_loss = torch.cat([uncensored_loss, censored_loss], dim=0).mean()

    # ----------------------------------
    # Ranking Loss
    # ----------------------------------
    event_idx = torch.where(events)[0]
    if len(event_idx) <= 1:
        # Not enough to compute ranking loss
        return likelihood_loss

    time_range = torch.arange(T, device=device).float()
    risk_scores = (pred * time_range).sum(dim=1)  # expected time as proxy for risk

    rank_loss = 0.0
    count = 0

    for i in event_idx:
        for j in range(N):
            if durations[j] > durations[i]:
                diff = risk_scores[j] - risk_scores[i]
                rank_loss += torch.exp(-diff)
                count += 1

    rank_loss = rank_loss / count if count > 0 else torch.tensor(0.0, device=device)

    # ----------------------------------
    # Final Combined Loss
    # ----------------------------------
    total_loss = (1 - alpha) * likelihood_loss + alpha * rank_loss
    return total_loss

def multimodal_loss(predictions, durations, events, survival_model="cox"):
    """
    predictions: model output
    durations: observed durations (time to event or censoring)
    events: 1 if event occurred, 0 if censored
    """
    if survival_model == "cox":
        # Cox partial likelihood loss (negative)
        risk = predictions.view(-1)
        # Sort by descending time
        order = torch.argsort(durations, descending=True)
        risk_sorted = risk[order]
        events_sorted = events[order]

        # Compute log partial likelihood
        exp_risk = torch.exp(risk_sorted)
        log_cum_sum = torch.log(torch.cumsum(exp_risk, dim=0))
        log_likelihood = risk_sorted - log_cum_sum
        neg_partial_ll = -torch.sum(log_likelihood * events_sorted) / torch.sum(events_sorted)
        return neg_partial_ll

    elif survival_model == "nnet_survival":
        # Assume predictions are raw logits for time_bins
        # Use negative log-likelihood of survival probability
        probs = torch.sigmoid(predictions)
        eps = 1e-8
        loss = 0.0
        for i in range(len(durations)):
            t = int(durations[i].item())
            e = events[i].item()
            if e == 1:
                loss -= torch.log(probs[i, t] + eps)
            else:
                loss -= torch.log(1 - probs[i, t] + eps)
        return loss / len(durations)
    else:
        raise ValueError(f"Unsupported survival model: {survival_model}")

# ===== Inference for expected event time =====
def infer_time(model, clin, mri):
    with torch.no_grad():
        out = model(clinical_feat=clin, mri_feat=mri)
        time_points = torch.arange(out.size(1), device=out.device).float()
        expected_time = torch.sum(out * time_points, dim=1)
        return expected_time

# ===== Training Loop =====
def train_one_epoch(model, dataloader, optimizer, survival_model, device):
    model.train()
    total_loss = 0.0
    count = 0

    for clin, mri, wsi, duration, event in dataloader:
        clin, mri, wsi, duration, event = (
            clin.to(device),
            mri.to(device),
            wsi.to(device),
            duration.to(device),
            event.to(device),
        )

        optimizer.zero_grad()
        outputs = model(clinical_feat=clin, mri_feat=mri, wsi_feat=wsi)

        main_pred = outputs[0]

        if survival_model == 'cox':
            loss = cox_loss(main_pred, duration, event)
        elif survival_model == 'deephit' and DEEPHIT_LOSS == 'uncensored':
            loss = deephit_loss_uncensored(main_pred, duration, event, debug=True)
        elif survival_model == 'deephit' and DEEPHIT_LOSS == 'censored':
            loss = deephit_loss_censored(main_pred, duration, event, debug=True)
        else:
            raise ValueError(f"Unknown survival model: {survival_model}")

        if torch.isnan(loss):
            raise ValueError("NaN in loss function")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count if count > 0 else float("inf")
    return avg_loss

# ===== Evaluation Function =====
def evaluate(model, dataloader, survival_model, device):
    model.eval()
    all_preds = []
    all_events = []
    all_durations = []

    with torch.no_grad():
        for clin, mri, wsi, duration, event in dataloader:
            clin = clin.to(device)
            mri = mri.to(device)
            wsi = wsi.to(device)
            duration = duration.to(device)
            event = event.to(device)

            outputs = model(clinical_feat=clin, mri_feat=mri, wsi_feat=wsi)
            main_pred = outputs[0]

            if survival_model == 'cox':
                preds = main_pred.squeeze()
            elif survival_model == 'deephit':
                time_bins = main_pred.shape[1]
                time_range = torch.arange(time_bins).float().to(device)
                preds = (main_pred * time_range).sum(dim=1)
            else:
                raise ValueError(f"Unknown survival model: {survival_model}")

            all_preds.append(preds.cpu())
            all_events.append(event.cpu())
            all_durations.append(duration.cpu())


    all_preds = torch.cat(all_preds).numpy()
    all_durations = torch.cat(all_durations).numpy()
    all_events = torch.cat(all_events).numpy()

    c_index = concordance_index_censored(
        all_events.astype(bool),
        all_durations,
        -all_preds
    )
    return c_index

# ===== Save Model =====
def save_model(model, fold_idx, run_num):
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    model_path = os.path.join(GLOBAL_DIR, f"best_model_run{run_num}_fold{fold_idx}.pt")
    torch.save(model.state_dict(), model_path)
    if VERBOSE:
        print(f"Saved best model for fold {fold_idx} at {model_path}")
    return model_path

def load_model(model_class, fold_idx, run_num):
    model_path = os.path.join(GLOBAL_DIR, f"best_model_run{run_num}_fold{fold_idx}.pt")
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
