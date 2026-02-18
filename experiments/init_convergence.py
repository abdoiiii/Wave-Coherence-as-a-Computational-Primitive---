"""
Initialization Convergence Test — Does Random Init Find the Same Structure?

THE QUESTION:
Train 5 identical models from different random seeds. Same data, same
architecture, same hyperparameters. Compare internal organisation.

THREE POSSIBLE OUTCOMES:
1. Different every time → no natural structure. Random init leads to
   random filing. Harmonic init provides structure the model can't find.
2. Same every time → natural structure exists. The data demands it.
   Harmonic init just gets there faster (skipping the search).
3. Partially convergent → some aspects converge (structural), others
   vary (identity). Tells us exactly which parts of harmonic encoding
   are "natural" vs "gifted".

WHY IT MATTERS:
If models organise differently from different seeds, that's the same
mechanism that makes ChatGPT different from Claude. The filing system
is an accident of initialisation. Harmonic encoding would standardise it.

WHAT WE MEASURE (per run):
- Channel independence (% of independent pairs at final layer)
- Correlation matrix between harmonic channel energies
- Per-band spectral profile (which bands carry which info)
- Cross-run similarity: do the 5 models agree on channel structure?
"""

import math
import os
import time
import urllib.request
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

class Config:
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = 256
    dropout = 0.0
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 3000
    eval_interval = 500
    eval_iters = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 5
SEEDS = [42, 137, 256, 1337, 9999]
N_HARMONICS_CHECK = 64  # check all 64 harmonic channels


# =============================================================================
# Model (reused architecture)
# =============================================================================

class HarmonicEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, trainable=True):
        super().__init__()
        assert embedding_dim % 2 == 0
        n_harmonics = embedding_dim // 2
        angles = torch.arange(num_embeddings, dtype=torch.float32) * (2 * math.pi / num_embeddings)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        phase_matrix = angles.unsqueeze(1) * harmonics.unsqueeze(0)
        embedding = torch.zeros(num_embeddings, embedding_dim)
        embedding[:, 0::2] = torch.cos(phase_matrix)
        embedding[:, 1::2] = torch.sin(phase_matrix)
        embedding = embedding * (1.0 / math.sqrt(n_harmonics))
        if trainable:
            self.weight = nn.Parameter(embedding)
        else:
            self.register_buffer("weight", embedding)

    def forward(self, x):
        return F.embedding(x, self.weight)


class HarmonicPositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim, trainable=True):
        super().__init__()
        assert embedding_dim % 2 == 0
        n_harmonics = embedding_dim // 2
        positions = torch.arange(max_len, dtype=torch.float32)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        freq_scale = 1.0 / (10000.0 ** (2.0 * (harmonics - 1) / embedding_dim))
        phase_matrix = positions.unsqueeze(1) * freq_scale.unsqueeze(0)
        encoding = torch.zeros(max_len, embedding_dim)
        encoding[:, 0::2] = torch.cos(phase_matrix)
        encoding[:, 1::2] = torch.sin(phase_matrix)
        encoding = encoding * (1.0 / math.sqrt(n_harmonics))
        if trainable:
            self.weight = nn.Parameter(encoding)
        else:
            self.register_buffer("weight", encoding)

    def forward(self, seq_len):
        return self.weight[:seq_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class HarmonicGPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.wte = HarmonicEmbedding(vocab_size, config.n_embd, trainable=True)
        self.wpe = HarmonicPositionalEncoding(config.block_size, config.n_embd, trainable=True)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class BaselineGPT(nn.Module):
    """Same architecture but with nn.Embedding (random init, no harmonic structure)."""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# =============================================================================
# Data
# =============================================================================

def download_shakespeare():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "python", "data")
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        filepath = os.path.join(data_dir, "shakespeare.txt")
        if not os.path.exists(filepath):
            os.makedirs(data_dir, exist_ok=True)
            print("  Downloading Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r") as f:
        return f.read()


class Dataset:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = [self.stoi[c] for c in text]
        n = int(0.9 * len(data))
        self.train_data = torch.tensor(data[:n], dtype=torch.long)
        self.val_data = torch.tensor(data[n:], dtype=torch.long)

    def get_batch(self, split, config):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        return x.to(config.device), y.to(config.device)


# =============================================================================
# Analysis functions
# =============================================================================

def collect_activations(model, dataset, config, n_batches=10):
    """Collect final-layer activations for analysis."""
    model.eval()
    all_acts = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = dataset.get_batch("val", config)
            B, T = x.size()

            # Forward through model manually to get final layer output
            if hasattr(model, 'wte') and isinstance(model.wte, HarmonicEmbedding):
                tok_emb = model.wte(x)
                pos_emb = model.wpe(T)
            else:
                tok_emb = model.wte(x)
                pos_emb = model.wpe(torch.arange(T, device=x.device))
            h = tok_emb + pos_emb
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
            all_acts.append(h.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


def compute_channel_correlation_matrix(activations, n_check):
    """Compute correlation matrix between harmonic channel energies."""
    flat = activations.reshape(-1, activations.shape[2])  # [N, n_embd]
    n_harmonics = flat.shape[1] // 2
    n_check = min(n_check, n_harmonics)

    # Per-harmonic energy
    energy = np.zeros((flat.shape[0], n_check))
    for h in range(n_check):
        energy[:, h] = np.sqrt(flat[:, h*2]**2 + flat[:, h*2+1]**2)

    # Correlation matrix
    corr = np.zeros((n_check, n_check))
    for i in range(n_check):
        for j in range(n_check):
            a = energy[:, i] - energy[:, i].mean()
            b = energy[:, j] - energy[:, j].mean()
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom > 1e-10:
                corr[i, j] = np.dot(a, b) / denom
    return corr


def compute_independence_stats(corr_matrix):
    """From correlation matrix, compute independence statistics."""
    n = corr_matrix.shape[0]
    independent = 0
    cooperative = 0
    competitive = 0
    coupled = 0

    for i in range(n):
        for j in range(i+1, n):
            c = corr_matrix[i, j]
            if abs(c) < 0.2:
                independent += 1
            elif c > 0.4:
                cooperative += 1
            elif c < -0.4:
                competitive += 1
            else:
                coupled += 1

    total = independent + cooperative + competitive + coupled
    return {
        "independent": independent,
        "cooperative": cooperative,
        "competitive": competitive,
        "coupled": coupled,
        "total": total,
        "pct_independent": 100.0 * independent / total if total > 0 else 0,
    }


def compute_band_energy_profile(activations, n_check):
    """Per-band average energy — spectral profile of what the model uses."""
    flat = activations.reshape(-1, activations.shape[2])
    n_harmonics = flat.shape[1] // 2
    n_check = min(n_check, n_harmonics)

    profile = np.zeros(n_check)
    for h in range(n_check):
        profile[h] = np.mean(np.sqrt(flat[:, h*2]**2 + flat[:, h*2+1]**2))
    return profile


def eval_loss(model, dataset, config, n_batches=None):
    if n_batches is None:
        n_batches = config.eval_iters
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch("val", config)
        with torch.no_grad():
            _, loss = model(x, y)
        total += loss.item()
    return total / n_batches


# =============================================================================
# Training
# =============================================================================

def train_model(config, dataset, seed, model_type="baseline"):
    """Train a single model with given seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if model_type == "baseline":
        model = BaselineGPT(config, dataset.vocab_size).to(config.device)
    else:
        model = HarmonicGPT(config, dataset.vocab_size).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss(model, dataset, config)
            elapsed = time.time() - start
            print(f"    step {it:>5} | val {val_loss:.4f} | {elapsed:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = eval_loss(model, dataset, config)
    print(f"    Final val loss: {final_loss:.4f} ({time.time()-start:.1f}s)")
    return model, final_loss


# =============================================================================
# Cross-run comparison
# =============================================================================

def compare_correlation_matrices(matrices):
    """
    Compare correlation matrices across runs.
    Returns pairwise similarity (correlation of upper triangles).
    """
    n_runs = len(matrices)
    # Extract upper triangles
    triu_idx = np.triu_indices(matrices[0].shape[0], k=1)
    uppers = [m[triu_idx] for m in matrices]

    # Pairwise correlation between upper triangles
    sim_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            a = uppers[i] - uppers[i].mean()
            b = uppers[j] - uppers[j].mean()
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom > 1e-10:
                sim_matrix[i, j] = np.dot(a, b) / denom
            else:
                sim_matrix[i, j] = 0.0
    return sim_matrix


def compare_energy_profiles(profiles):
    """Compare spectral energy profiles across runs."""
    n_runs = len(profiles)
    sim_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            a = profiles[i] - profiles[i].mean()
            b = profiles[j] - profiles[j].mean()
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom > 1e-10:
                sim_matrix[i, j] = np.dot(a, b) / denom
            else:
                sim_matrix[i, j] = 0.0
    return sim_matrix


# =============================================================================
# Main experiment
# =============================================================================

def main():
    print("=" * 60)
    print("  INITIALIZATION CONVERGENCE TEST")
    print("  Does random init find the same structure every time?")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # =========================================================================
    # PART 1: Train 5 baseline models (random init, different seeds)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 1: BASELINE MODELS (random init × 5 seeds)")
    print("=" * 60)

    baseline_models = []
    baseline_losses = []
    baseline_corr_matrices = []
    baseline_independence = []
    baseline_energy_profiles = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  --- Baseline run {i+1}/5 (seed={seed}) ---")
        model, final_loss = train_model(config, dataset, seed, "baseline")
        baseline_models.append(model)
        baseline_losses.append(final_loss)

        # Collect activations and analyse
        acts = collect_activations(model, dataset, config)
        corr = compute_channel_correlation_matrix(acts, N_HARMONICS_CHECK)
        indep = compute_independence_stats(corr)
        profile = compute_band_energy_profile(acts, N_HARMONICS_CHECK)

        baseline_corr_matrices.append(corr)
        baseline_independence.append(indep)
        baseline_energy_profiles.append(profile)
        print(f"    Channel independence: {indep['pct_independent']:.1f}%")

    # =========================================================================
    # PART 2: Train 5 harmonic models (harmonic init × 5 seeds)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 2: HARMONIC MODELS (harmonic init × 5 seeds)")
    print("=" * 60)

    harmonic_models = []
    harmonic_losses = []
    harmonic_corr_matrices = []
    harmonic_independence = []
    harmonic_energy_profiles = []

    for i, seed in enumerate(SEEDS):
        print(f"\n  --- Harmonic run {i+1}/5 (seed={seed}) ---")
        model, final_loss = train_model(config, dataset, seed, "harmonic")
        harmonic_models.append(model)
        harmonic_losses.append(final_loss)

        acts = collect_activations(model, dataset, config)
        corr = compute_channel_correlation_matrix(acts, N_HARMONICS_CHECK)
        indep = compute_independence_stats(corr)
        profile = compute_band_energy_profile(acts, N_HARMONICS_CHECK)

        harmonic_corr_matrices.append(corr)
        harmonic_independence.append(indep)
        harmonic_energy_profiles.append(profile)
        print(f"    Channel independence: {indep['pct_independent']:.1f}%")

    # =========================================================================
    # ANALYSIS 1: Per-run summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: Per-Run Summary")
    print("=" * 60)

    print(f"\n  {'Model':<12} {'Seed':>6} {'Val Loss':>10} {'Independence':>14}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*14}")
    for i in range(N_RUNS):
        print(f"  {'Baseline':<12} {SEEDS[i]:>6} {baseline_losses[i]:>10.4f} {baseline_independence[i]['pct_independent']:>13.1f}%")
    print()
    for i in range(N_RUNS):
        print(f"  {'Harmonic':<12} {SEEDS[i]:>6} {harmonic_losses[i]:>10.4f} {harmonic_independence[i]['pct_independent']:>13.1f}%")

    bl_loss_mean = np.mean(baseline_losses)
    bl_loss_std = np.std(baseline_losses)
    hm_loss_mean = np.mean(harmonic_losses)
    hm_loss_std = np.std(harmonic_losses)
    bl_indep_mean = np.mean([x['pct_independent'] for x in baseline_independence])
    bl_indep_std = np.std([x['pct_independent'] for x in baseline_independence])
    hm_indep_mean = np.mean([x['pct_independent'] for x in harmonic_independence])
    hm_indep_std = np.std([x['pct_independent'] for x in harmonic_independence])

    print(f"\n  {'':>12} {'Mean Loss':>10} {'Std':>8} {'Mean Indep':>12} {'Std':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*12} {'-'*8}")
    print(f"  {'Baseline':<12} {bl_loss_mean:>10.4f} {bl_loss_std:>8.4f} {bl_indep_mean:>11.1f}% {bl_indep_std:>7.1f}%")
    print(f"  {'Harmonic':<12} {hm_loss_mean:>10.4f} {hm_loss_std:>8.4f} {hm_indep_mean:>11.1f}% {hm_indep_std:>7.1f}%")

    # =========================================================================
    # ANALYSIS 2: Cross-run structural consistency
    # =========================================================================
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Cross-Run Structural Consistency")
    print("  Do different seeds produce the same internal organisation?")
    print("=" * 60)

    # Correlation matrix similarity
    bl_corr_sim = compare_correlation_matrices(baseline_corr_matrices)
    hm_corr_sim = compare_correlation_matrices(harmonic_corr_matrices)

    # Extract off-diagonal (cross-run) similarities
    triu = np.triu_indices(N_RUNS, k=1)
    bl_cross = bl_corr_sim[triu]
    hm_cross = hm_corr_sim[triu]

    print(f"\n  Channel correlation structure (do runs agree on which channels interact?):")
    print(f"    Baseline cross-run similarity:  {np.mean(bl_cross):.4f} +/- {np.std(bl_cross):.4f}")
    print(f"    Harmonic cross-run similarity:  {np.mean(hm_cross):.4f} +/- {np.std(hm_cross):.4f}")

    if np.mean(hm_cross) > np.mean(bl_cross) + 0.05:
        print(f"    >> Harmonic models converge to MORE consistent structure")
    elif np.mean(bl_cross) > np.mean(hm_cross) + 0.05:
        print(f"    >> Baseline models converge to MORE consistent structure")
    else:
        print(f"    >> Both model types show similar structural consistency")

    # Energy profile similarity
    bl_prof_sim = compare_energy_profiles(baseline_energy_profiles)
    hm_prof_sim = compare_energy_profiles(harmonic_energy_profiles)

    bl_prof_cross = bl_prof_sim[triu]
    hm_prof_cross = hm_prof_sim[triu]

    print(f"\n  Energy profile (do runs agree on which bands carry energy?):")
    print(f"    Baseline cross-run similarity:  {np.mean(bl_prof_cross):.4f} +/- {np.std(bl_prof_cross):.4f}")
    print(f"    Harmonic cross-run similarity:  {np.mean(hm_prof_cross):.4f} +/- {np.std(hm_prof_cross):.4f}")

    if np.mean(hm_prof_cross) > np.mean(bl_prof_cross) + 0.05:
        print(f"    >> Harmonic models converge to MORE consistent energy distribution")
    elif np.mean(bl_prof_cross) > np.mean(hm_prof_cross) + 0.05:
        print(f"    >> Baseline models converge to MORE consistent energy distribution")
    else:
        print(f"    >> Both model types show similar energy consistency")

    # =========================================================================
    # ANALYSIS 3: Band-by-band convergence
    # =========================================================================
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: Band-by-Band Convergence")
    print("  Which bands converge across seeds? Which vary?")
    print("=" * 60)

    # Per-band variance across runs (lower = more convergent)
    bl_profiles = np.array(baseline_energy_profiles)  # [5, 64]
    hm_profiles = np.array(harmonic_energy_profiles)

    # Normalise per run so we compare shape not scale
    bl_normed = bl_profiles / (bl_profiles.sum(axis=1, keepdims=True) + 1e-10)
    hm_normed = hm_profiles / (hm_profiles.sum(axis=1, keepdims=True) + 1e-10)

    bl_band_std = np.std(bl_normed, axis=0)
    hm_band_std = np.std(hm_normed, axis=0)

    # Group into low/mid/high
    low_bands = slice(0, 16)
    mid_bands = slice(16, 40)
    high_bands = slice(40, 64)

    print(f"\n  Cross-run variance by band region (lower = more convergent):")
    print(f"    {'Region':<15} {'Baseline var':>14} {'Harmonic var':>14} {'More consistent':>18}")
    print(f"    {'-'*15} {'-'*14} {'-'*14} {'-'*18}")

    for name, sl in [("Low (1-16)", low_bands), ("Mid (17-40)", mid_bands), ("High (41-64)", high_bands)]:
        bl_v = np.mean(bl_band_std[sl])
        hm_v = np.mean(hm_band_std[sl])
        winner = "Harmonic" if hm_v < bl_v else "Baseline"
        print(f"    {name:<15} {bl_v:>14.6f} {hm_v:>14.6f} {winner:>18}")

    # Top 5 most convergent and most variable bands for each model type
    print(f"\n  Baseline — 5 most convergent bands (lowest cross-run variance):")
    bl_sorted = np.argsort(bl_band_std)
    for idx in bl_sorted[:5]:
        print(f"    Band n={idx+1:>2}: variance {bl_band_std[idx]:.6f}")

    print(f"\n  Baseline — 5 most variable bands (highest cross-run variance):")
    for idx in bl_sorted[-5:][::-1]:
        print(f"    Band n={idx+1:>2}: variance {bl_band_std[idx]:.6f}")

    print(f"\n  Harmonic — 5 most convergent bands:")
    hm_sorted = np.argsort(hm_band_std)
    for idx in hm_sorted[:5]:
        print(f"    Band n={idx+1:>2}: variance {hm_band_std[idx]:.6f}")

    print(f"\n  Harmonic — 5 most variable bands:")
    for idx in hm_sorted[-5:][::-1]:
        print(f"    Band n={idx+1:>2}: variance {hm_band_std[idx]:.6f}")

    # =========================================================================
    # ANALYSIS 4: Pairwise correlation matrix comparison (detailed)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  ANALYSIS 4: Cross-Run Correlation Detail")
    print("=" * 60)

    print(f"\n  Baseline cross-run correlation matrix similarity:")
    for i in range(N_RUNS):
        row = "    "
        for j in range(N_RUNS):
            if i == j:
                row += "  ---  "
            else:
                row += f" {bl_corr_sim[i,j]:>5.3f} "
        print(row + f"  (seed {SEEDS[i]})")

    print(f"\n  Harmonic cross-run correlation matrix similarity:")
    for i in range(N_RUNS):
        row = "    "
        for j in range(N_RUNS):
            if i == j:
                row += "  ---  "
            else:
                row += f" {hm_corr_sim[i,j]:>5.3f} "
        print(row + f"  (seed {SEEDS[i]})")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)

    bl_struct_consistency = np.mean(bl_cross)
    hm_struct_consistency = np.mean(hm_cross)
    bl_profile_consistency = np.mean(bl_prof_cross)
    hm_profile_consistency = np.mean(hm_prof_cross)

    print(f"\n  Structural consistency (channel interactions):")
    print(f"    Baseline: {bl_struct_consistency:.4f}")
    print(f"    Harmonic: {hm_struct_consistency:.4f}")

    print(f"\n  Energy profile consistency (band usage):")
    print(f"    Baseline: {bl_profile_consistency:.4f}")
    print(f"    Harmonic: {hm_profile_consistency:.4f}")

    print(f"\n  Independence spread:")
    print(f"    Baseline: {bl_indep_mean:.1f}% +/- {bl_indep_std:.1f}%")
    print(f"    Harmonic: {hm_indep_mean:.1f}% +/- {hm_indep_std:.1f}%")

    # Determine which outcome
    if bl_struct_consistency > 0.8:
        if hm_struct_consistency > 0.8:
            print(f"\n  OUTCOME 2: Both converge to similar structure.")
            print(f"  The data demands a specific organisation regardless of init.")
            if hm_indep_mean > bl_indep_mean + 5:
                print(f"  But harmonic init achieves {hm_indep_mean - bl_indep_mean:.1f}% more independence.")
                print(f"  Harmonic encoding overshoots the natural attractor.")
        else:
            print(f"\n  UNEXPECTED: Baseline converges but harmonic doesn't.")
    elif bl_struct_consistency < 0.5:
        if hm_struct_consistency > 0.8:
            print(f"\n  OUTCOME 1+3: Baseline organises DIFFERENTLY each time,")
            print(f"  but harmonic init converges to CONSISTENT structure.")
            print(f"  Random init leads to arbitrary filing — harmonic provides the drawers.")
        else:
            print(f"\n  OUTCOME 1: Both organise differently each time.")
            print(f"  No natural attractor. Every model invents its own filing system.")
    else:
        # Partially convergent
        print(f"\n  OUTCOME 3: PARTIAL CONVERGENCE.")
        if hm_struct_consistency > bl_struct_consistency + 0.1:
            print(f"  Harmonic models are MORE consistent ({hm_struct_consistency:.3f} vs {bl_struct_consistency:.3f}).")
            print(f"  Harmonic encoding provides structure beyond what gradient descent finds.")
        elif bl_struct_consistency > hm_struct_consistency + 0.1:
            print(f"  Baseline models are surprisingly MORE consistent ({bl_struct_consistency:.3f} vs {hm_struct_consistency:.3f}).")
        else:
            print(f"  Similar consistency levels ({bl_struct_consistency:.3f} vs {hm_struct_consistency:.3f}).")
        print(f"  Some aspects of organisation converge; others depend on the seed.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
