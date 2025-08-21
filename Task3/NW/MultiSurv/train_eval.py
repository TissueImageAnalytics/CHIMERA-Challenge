import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sksurv.metrics import concordance_index_censored
from config import *
import torch.nn.functional as F
import random

# ===== Dataset for torch =====
class SurvivalDataset(Dataset):
    def __init__(self, clin_array, rna_array, wsi_array, durations, events):
        self.clin_array = clin_array
        self.rna_array = rna_array
        self.wsi_array = wsi_array
        self.durations = durations
        self.events = events

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, idx):
        clin = self.clin_array[idx] if self.clin_array is not None else np.zeros(0, dtype=np.float32)
        rna = self.rna_array[idx] if self.rna_array is not None else np.zeros(0, dtype=np.float32)
        wsi = self.wsi_array[idx] if self.wsi_array is not None else np.zeros(0, dtype=np.float32)
        duration = self.durations[idx]
        event = self.events[idx]
        return (
            torch.tensor(clin, dtype=torch.float32),
            torch.tensor(rna, dtype=torch.float32),
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

    if debug:
        # Check sum of PMF per sample (should be close to 1)
        pmf_sums = pred.sum(dim=1)
        if torch.any(pmf_sums < 0.99) or torch.any(pmf_sums > 1.01):
            print("Warning: PMF sums not close to 1:", pmf_sums)

        # Check for very small event probabilities that cause large loss
        very_small_probs = p_event < 1e-6
        if torch.any(very_small_probs):
            print("Warning: very small predicted event probs causing large loss:", p_event[very_small_probs])

    if debug and torch.any(p_event <= 0):
        print("🔴 Invalid event probabilities:", p_event[p_event <= 0])
    uncensored_loss = -safe_log(p_event)
    uncensored_loss = uncensored_loss[events]

    # Censored: log(1 - F(t)) ≈ log(1 - CDF)
    cdf = torch.cumsum(pred, dim=1)
    survival_prob = torch.clamp(1.0 - cdf[idx, durations], min=1e-4)
    if debug and torch.any(survival_prob <= 0):
        print("🔴 Invalid survival probabilities:", survival_prob[survival_prob <= 0])
    censored_loss = -safe_log(survival_prob)
    censored_loss = censored_loss[~events]

    likelihood_loss = torch.cat([uncensored_loss, censored_loss], dim=0).mean()

    if debug:
        min_surv = survival_prob.min().item()
        max_surv = survival_prob.max().item()
        #print(f"Survival prob min/max: {min_surv:.8f} / {max_surv:.8f}")

    if torch.any(survival_prob == 1e-6):
        print(f"Some survival probs at clamp minimum")


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

    if debug:
        # Return all three losses for inspection
        return total_loss, likelihood_loss, rank_loss
    else:
        return total_loss

# ===== Inference for expected event time =====
def infer_time(model, clin, rna):
    with torch.no_grad():
        out = model(clinical_feat=clin, rna_feat=rna)
        time_points = torch.arange(out.size(1), device=out.device).float()
        expected_time = torch.sum(out * time_points, dim=1)
        return expected_time

def modality_dropout(clin_feat, rna_feat, wsi_feat, drop_probs=(CLINICAL_DROPOUT, RNA_DROPOUT, WSI_DROPOUT)):
    if clin_feat is not None and random.random() < drop_probs[0]:
        clin_feat = torch.zeros_like(clin_feat)
    if rna_feat is not None and random.random() < drop_probs[1]:
        rna_feat = torch.zeros_like(rna_feat)
    if wsi_feat is not None and random.random() < drop_probs[2]:
        wsi_feat = torch.zeros_like(wsi_feat)
    return clin_feat, rna_feat, wsi_feat

# ===== Training Loop =====
def train_one_epoch(model, dataloader, optimizer, survival_model, device):
    model.train()
    total_loss = 0.0
    count = 0

    for clin, rna, wsi, duration, event in dataloader:
        clin, rna, wsi, duration, event = (
            clin.to(device),
            rna.to(device),
            wsi.to(device),
            duration.to(device),
            event.to(device),
        )

        optimizer.zero_grad()
        clin, rna, wsi = modality_dropout(clin_feat=clin, rna_feat=rna, wsi_feat=wsi)
        outputs = model(clinical_feat=clin, rna_feat=rna, wsi_feat=wsi)

        for i in range(min(5, outputs.size(0))):
            #print(f"Sample {i}:")
            #print(f" Event time bin: {duration[i].item()}")
            #print(f" Event observed: {event[i].item()}")
            #print(f" Predicted PMF: {outputs[i].detach().cpu().numpy()}")
            pass

        pmf_sums = outputs.sum(dim=1)
        #print("PMF sums min/max:", pmf_sums.min().item(), pmf_sums.max().item())

        min_val = outputs.min().item()
        if min_val < 0:
            print("Warning: Negative probabilities in predictions!")
        if (pmf_sums < 0.99).any() or (pmf_sums > 1.01).any():
            print("Warning: PMF sums not close to 1")

        if survival_model == 'cox':
            loss = cox_loss(outputs, duration, event)
        elif survival_model == 'deephit' and DEEPHIT_LOSS == 'uncensored':
            result = deephit_loss_uncensored(outputs, duration, event, debug=True)
            if isinstance(result, tuple):
                loss, likelihood_loss, rank_loss = result
                #print(f"Loss: {loss.item():.4f} Likelihood: {likelihood_loss.item():.4f} Ranking: {rank_loss.item():.4f}")
                #print(f"Loss: {loss.item():.4f}")
            else:
                loss = result
        elif survival_model == 'deephit' and DEEPHIT_LOSS == 'censored':
            result = deephit_loss_censored(outputs, duration, event, debug=True)
            if isinstance(result, tuple):
                loss, likelihood_loss, rank_loss = result
                #print(f"Loss: {loss.item():.4f} Likelihood: {likelihood_loss.item():.4f} Ranking: {rank_loss.item():.4f}")
                #print(f"Loss: {loss.item():.4f}")
            else:
                loss = result
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
        for clin, rna, wsi, duration, event in dataloader:
            clin = clin.to(device)
            rna = rna.to(device)
            wsi = wsi.to(device)
            duration = duration.to(device)
            event = event.to(device)
            clin, rna, wsi = modality_dropout(clin_feat=clin, rna_feat=rna, wsi_feat=wsi)
            output = model(clinical_feat=clin, rna_feat=rna, wsi_feat=wsi) 

            if survival_model == 'cox':
                preds = output.squeeze()
            elif survival_model == 'deephit':
                time_bins = output.shape[1]
                time_range = torch.arange(time_bins).float().to(device)
                preds = (output * time_range).sum(dim=1)
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
