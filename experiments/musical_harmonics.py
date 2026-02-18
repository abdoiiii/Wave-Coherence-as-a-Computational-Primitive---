"""
Musical Harmonic Analysis of Phase Coherence Channels

THE INSIGHT:
The ratios between harmonic channels n=2, n=3, n=4... ARE musical intervals.
n=3 : n=2 = 3:2 = perfect fifth
n=4 : n=3 = 4:3 = perfect fourth
n=5 : n=4 = 5:4 = major third
n=6 : n=5 = 6:5 = minor third

Music theory provides centuries of knowledge about which combinations are
stable (consonant) versus unstable (dissonant). If this maps onto our
channel independence data from Phase 2, we get engineering guidance for free:
- Consonant channel pairs = stable, load-bearing, don't touch
- Dissonant channel pairs = transitional, safe to edit

THE TEST:
1. Compute musical intervals between all channel pairs
2. Score consonance using Tenney height (lower = more consonant)
3. Cross-reference with Phase 2 independence/entanglement data
4. Test: do consonant pairs correlate with independent channels?
5. Test: do dissonant pairs correlate with entangled channels?
6. Apply to injection results: did we accidentally edit consonant structure?
"""

import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fractions import Fraction


# =============================================================================
# Musical Theory Functions
# =============================================================================

# Named musical intervals and their ratios
MUSICAL_INTERVALS = {
    "unison":         Fraction(1, 1),
    "minor second":   Fraction(16, 15),
    "major second":   Fraction(9, 8),
    "minor third":    Fraction(6, 5),
    "major third":    Fraction(5, 4),
    "perfect fourth": Fraction(4, 3),
    "tritone":        Fraction(45, 32),
    "perfect fifth":  Fraction(3, 2),
    "minor sixth":    Fraction(8, 5),
    "major sixth":    Fraction(5, 3),
    "minor seventh":  Fraction(16, 9),
    "major seventh":  Fraction(15, 8),
    "octave":         Fraction(2, 1),
}

# Consonance ranking (lower = more consonant)
CONSONANCE_RANK = {
    "unison": 0, "octave": 1,
    "perfect fifth": 2, "perfect fourth": 3,
    "major third": 4, "minor third": 5,
    "major sixth": 6, "minor sixth": 7,
    "major second": 8, "minor seventh": 9,
    "major seventh": 10, "minor second": 11,
    "tritone": 12,
}


def tenney_height(p, q):
    """
    Tenney height of ratio p/q: log2(p * q) where p/q is in lowest terms.
    Lower = more consonant. Unison (1:1) = 0, Octave (2:1) = 1, Fifth (3:2) = 2.58
    """
    f = Fraction(p, q)
    return math.log2(f.numerator * f.denominator)


def identify_interval(ratio):
    """
    Find the closest named musical interval for a given ratio.
    Returns (name, distance_from_pure).
    """
    # Reduce ratio to within one octave (1.0 to 2.0)
    r = ratio
    while r > 2.0:
        r /= 2.0
    while r < 1.0:
        r *= 2.0

    best_name = "unknown"
    best_dist = float("inf")

    for name, frac in MUSICAL_INTERVALS.items():
        target = float(frac)
        # Try both the interval and its octave reduction
        dist = abs(r - target)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name, best_dist


def consonance_score(n_a, n_b):
    """
    Compute consonance score for the ratio between harmonic channels n_a and n_b.
    Returns (tenney_height, interval_name, consonance_rank).
    """
    if n_a == 0 or n_b == 0:
        return float("inf"), "undefined", 99

    # The ratio between channels
    hi, lo = max(n_a, n_b), min(n_a, n_b)
    ratio = hi / lo

    # Tenney height
    f = Fraction(hi, lo).limit_denominator(100)
    th = math.log2(f.numerator * f.denominator)

    # Named interval
    name, dist = identify_interval(ratio)
    rank = CONSONANCE_RANK.get(name, 13)

    return th, name, rank


# =============================================================================
# Model (same as all experiments)
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
    eval_iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, config, vocab_size, mode="harmonic"):
        super().__init__()
        self.config = config
        self.mode = mode
        trainable = (mode == "harmonic")
        self.wte = HarmonicEmbedding(vocab_size, config.n_embd, trainable=trainable)
        self.wpe = HarmonicPositionalEncoding(config.block_size, config.n_embd, trainable=trainable)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  {mode} model: {n_params:,} trainable parameters")

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


def train_model(config, dataset, mode="harmonic"):
    print(f"\n  Training {mode} model...")
    model = HarmonicGPT(config, dataset.vocab_size, mode=mode).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()
    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            model.eval()
            losses = {"train": 0.0, "val": 0.0}
            for split in ["train", "val"]:
                for _ in range(config.eval_iters):
                    x, y = dataset.get_batch(split, config)
                    with torch.no_grad():
                        _, loss = model(x, y)
                    losses[split] += loss.item()
                losses[split] /= config.eval_iters
            elapsed = time.time() - start
            print(f"  step {it:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")
            model.train()
        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# Activation capture (from Phase 2)
# =============================================================================

def forward_with_activations(model, x):
    """Capture activations at every layer for spectral analysis."""
    activations = {}
    B, T = x.size()
    tok_emb = model.wte(x)
    activations["embedding"] = tok_emb.detach()
    pos_emb = model.wpe(T)
    h = tok_emb + pos_emb
    activations["embed+pos"] = h.detach()

    for i, block in enumerate(model.blocks):
        ln1_out = block.ln_1(h)
        activations[f"layer{i}_pre_attn"] = ln1_out.detach()
        attn_out = block.attn(ln1_out)
        h = h + attn_out
        activations[f"layer{i}_post_attn"] = h.detach()
        ln2_out = block.ln_2(h)
        mlp_out = block.mlp(ln2_out)
        h = h + mlp_out
        activations[f"layer{i}_post_mlp"] = h.detach()

    h = model.ln_f(h)
    activations["final"] = h.detach()
    return activations


# =============================================================================
# Analysis 1: Musical interval map of channel pairs
# =============================================================================

def analyze_interval_map(n_harmonics):
    """
    Map all channel pair ratios to musical intervals.
    """
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 1: Musical Interval Map")
    print(f"  Ratios between all {n_harmonics} harmonic channels")
    print(f"{'='*60}")

    # Focus on adjacent and nearby channels
    print(f"\n  Adjacent channel intervals (n, n+1):")
    print(f"  {'Pair':>10} | {'Ratio':>8} | {'Interval':>16} | {'Tenney':>8} | {'Consonance':>11}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}-+-{'-'*11}")

    for n in range(1, min(17, n_harmonics)):
        m = n + 1
        ratio = m / n
        th, name, rank = consonance_score(n, m)
        cons_label = "consonant" if rank <= 5 else "mild" if rank <= 9 else "dissonant"
        print(f"  ({n:>2},{m:>2})   | {ratio:>8.4f} | {name:>16} | {th:>8.2f} | {cons_label:>11}")

    # Musical intervals within the first 16 channels
    print(f"\n  Key musical intervals found in channels 1-16:")
    print(f"  {'Interval':>16} | {'Ratio':>8} | {'Channel pairs':>30}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*30}")

    for interval_name, interval_ratio in sorted(MUSICAL_INTERVALS.items(),
                                                  key=lambda x: CONSONANCE_RANK.get(x[0], 99)):
        r = float(interval_ratio)
        pairs = []
        for n in range(1, 17):
            for m in range(n+1, 17):
                if abs(m/n - r) < 0.01:
                    pairs.append(f"({n},{m})")
        if pairs:
            rank = CONSONANCE_RANK[interval_name]
            cons = "***" if rank <= 3 else "**" if rank <= 7 else "*" if rank <= 9 else ""
            print(f"  {interval_name:>16} | {r:>8.4f} | {', '.join(pairs[:6]):>30} {cons}")


# =============================================================================
# Analysis 2: Cross-reference with channel independence
# =============================================================================

def analyze_consonance_vs_independence(model, dataset, config):
    """
    Compute channel independence (from Phase 2) and cross-reference with
    musical consonance scores.

    THE TEST: Do consonant channel pairs show more independence?
    """
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 2: Consonance vs Channel Independence")
    print(f"  Do consonant pairs stay independent? Do dissonant pairs entangle?")
    print(f"{'='*60}")

    model.eval()
    n_harmonics = config.n_embd // 2

    # Collect activations at the final layer
    n_batches = 30
    all_activations = []
    for _ in range(n_batches):
        x, _ = dataset.get_batch("val", config)
        with torch.no_grad():
            acts = forward_with_activations(model, x)
        all_activations.append(acts["final"].cpu())

    final_acts = torch.cat(all_activations, dim=0)  # [N, T, C]
    # Flatten to [N*T, C]
    flat = final_acts.reshape(-1, config.n_embd).numpy()

    # Compute correlation between harmonic band energies
    # Each band h has 2 dims: cos at 2h, sin at 2h+1
    band_energies = np.zeros((flat.shape[0], n_harmonics))
    for h in range(n_harmonics):
        ci, si = h * 2, h * 2 + 1
        band_energies[:, h] = flat[:, ci]**2 + flat[:, si]**2

    # Correlation matrix between band energies
    corr_matrix = np.corrcoef(band_energies.T)  # [n_harmonics, n_harmonics]

    # For each channel pair, compute:
    # 1. Musical consonance (Tenney height)
    # 2. Channel independence (1 - abs(correlation))
    pairs_data = []

    for i in range(n_harmonics):
        for j in range(i+1, n_harmonics):
            n_i = i + 1  # 1-indexed harmonic number
            n_j = j + 1

            th, interval_name, rank = consonance_score(n_i, n_j)
            independence = 1.0 - abs(corr_matrix[i, j])
            correlation = corr_matrix[i, j]

            pairs_data.append({
                "n_i": n_i, "n_j": n_j,
                "tenney": th, "interval": interval_name, "rank": rank,
                "independence": independence,
                "correlation": abs(correlation),
            })

    # Sort into consonance buckets
    consonant = [p for p in pairs_data if p["rank"] <= 5]   # unison through minor third
    mild = [p for p in pairs_data if 5 < p["rank"] <= 9]     # major sixth through minor seventh
    dissonant = [p for p in pairs_data if p["rank"] > 9]     # major seventh, minor second, tritone
    # Also bucket by tenney height
    low_tenney = [p for p in pairs_data if p["tenney"] < 4]
    mid_tenney = [p for p in pairs_data if 4 <= p["tenney"] < 7]
    high_tenney = [p for p in pairs_data if p["tenney"] >= 7]

    print(f"\n  Channel pair statistics by musical consonance:")
    print(f"  {'Category':>14} | {'Count':>6} | {'Avg independence':>18} | {'Avg correlation':>17}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*18}-+-{'-'*17}")

    for label, bucket in [("consonant", consonant), ("mild", mild), ("dissonant", dissonant)]:
        if bucket:
            avg_ind = np.mean([p["independence"] for p in bucket])
            avg_corr = np.mean([p["correlation"] for p in bucket])
            print(f"  {label:>14} | {len(bucket):>6} | {avg_ind:>16.4f}   | {avg_corr:>15.4f}")

    print(f"\n  Channel pair statistics by Tenney height:")
    print(f"  {'Tenney range':>14} | {'Count':>6} | {'Avg independence':>18} | {'Avg correlation':>17}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*18}-+-{'-'*17}")

    for label, bucket in [("low (<4)", low_tenney), ("mid (4-7)", mid_tenney), ("high (>7)", high_tenney)]:
        if bucket:
            avg_ind = np.mean([p["independence"] for p in bucket])
            avg_corr = np.mean([p["correlation"] for p in bucket])
            print(f"  {label:>14} | {len(bucket):>6} | {avg_ind:>16.4f}   | {avg_corr:>15.4f}")

    # Overall correlation between consonance and independence
    all_tenney = np.array([p["tenney"] for p in pairs_data])
    all_indep = np.array([p["independence"] for p in pairs_data])
    all_corr_vals = np.array([p["correlation"] for p in pairs_data])

    corr_tenney_indep = np.corrcoef(all_tenney, all_indep)[0, 1]
    corr_tenney_corr = np.corrcoef(all_tenney, all_corr_vals)[0, 1]

    print(f"\n  Correlation: Tenney height vs independence: {corr_tenney_indep:+.4f}")
    print(f"  Correlation: Tenney height vs correlation:  {corr_tenney_corr:+.4f}")
    print(f"  (Positive tenney-independence means: more dissonant pairs are MORE independent)")
    print(f"  (Negative tenney-correlation means: more dissonant pairs are LESS correlated)")

    # Show the most consonant pairs and their independence
    print(f"\n  Most consonant channel pairs (lowest Tenney height):")
    print(f"  {'Pair':>10} | {'Ratio':>8} | {'Interval':>16} | {'Tenney':>8} | {'Indep.':>8} | {'Corr.':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    sorted_pairs = sorted(pairs_data, key=lambda p: p["tenney"])
    for p in sorted_pairs[:20]:
        ratio = p["n_j"] / p["n_i"]
        print(f"  ({p['n_i']:>2},{p['n_j']:>2})   | {ratio:>8.4f} | {p['interval']:>16} | {p['tenney']:>8.2f} | {p['independence']:>8.4f} | {p['correlation']:>8.4f}")

    # Show the most entangled pairs and their musical interval
    print(f"\n  Most entangled pairs (highest correlation, lowest independence):")
    print(f"  {'Pair':>10} | {'Ratio':>8} | {'Interval':>16} | {'Tenney':>8} | {'Indep.':>8} | {'Corr.':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    entangled = sorted(pairs_data, key=lambda p: p["independence"])
    for p in entangled[:20]:
        ratio = p["n_j"] / p["n_i"]
        print(f"  ({p['n_i']:>2},{p['n_j']:>2})   | {ratio:>8.4f} | {p['interval']:>16} | {p['tenney']:>8.2f} | {p['independence']:>8.4f} | {p['correlation']:>8.4f}")

    return pairs_data, corr_matrix


# =============================================================================
# Analysis 3: Spectral chords — which channels fire together?
# =============================================================================

def analyze_spectral_chords(model, dataset, config):
    """
    For different character types (vowels, consonants, punctuation, uppercase),
    identify which harmonic channels are most active. Then compute the musical
    intervals between the active channels.

    This tells us: does 'e' form a major chord? Does 'z' form a dissonant cluster?
    """
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 3: Spectral Chords")
    print(f"  What musical chord does each character type form?")
    print(f"{'='*60}")

    model.eval()
    n_harmonics = config.n_embd // 2

    # Get the trained embedding table
    emb = model.wte.weight.data.cpu().numpy()

    # Character groups
    stoi = dataset.stoi
    groups = {
        "vowels": [c for c in "aeiou" if c in stoi],
        "common consonants": [c for c in "tnsrhl" if c in stoi],
        "rare consonants": [c for c in "zqxjk" if c in stoi],
        "punctuation": [c for c in ".,;:!?'-" if c in stoi],
        "uppercase": [c for c in "ABCDEFGHIJKLM" if c in stoi],
        "space/newline": [c for c in " \n" if c in stoi],
    }

    for group_name, chars in groups.items():
        if not chars:
            continue

        # Average band energy across characters in this group
        avg_energy = np.zeros(n_harmonics)
        for c in chars:
            idx = stoi[c]
            for h in range(n_harmonics):
                ci, si = h * 2, h * 2 + 1
                avg_energy[h] += emb[idx, ci]**2 + emb[idx, si]**2
        avg_energy /= len(chars)

        # Find top active bands
        top_bands = np.argsort(avg_energy)[::-1][:6]
        top_bands_1indexed = [b + 1 for b in top_bands]

        # Compute intervals between top bands
        intervals = []
        for i in range(len(top_bands_1indexed)):
            for j in range(i+1, len(top_bands_1indexed)):
                n_i, n_j = top_bands_1indexed[i], top_bands_1indexed[j]
                hi, lo = max(n_i, n_j), min(n_i, n_j)
                _, name, rank = consonance_score(lo, hi)
                intervals.append((lo, hi, name, rank))

        # Classify the chord
        avg_rank = np.mean([iv[3] for iv in intervals]) if intervals else 0
        if avg_rank <= 4:
            chord_type = "MAJOR (consonant)"
        elif avg_rank <= 7:
            chord_type = "MINOR (mild)"
        elif avg_rank <= 10:
            chord_type = "DIMINISHED (tense)"
        else:
            chord_type = "AUGMENTED (dissonant)"

        print(f"\n  {group_name} ({', '.join(repr(c) for c in chars[:5])}{'...' if len(chars) > 5 else ''}):")
        print(f"    Top bands: {top_bands_1indexed}")
        print(f"    Band energies: {[f'{avg_energy[b]:.4f}' for b in top_bands[:6]]}")

        # Show intervals
        consonant_count = sum(1 for iv in intervals if iv[3] <= 5)
        dissonant_count = sum(1 for iv in intervals if iv[3] > 9)
        print(f"    Intervals: {consonant_count} consonant, {len(intervals)-consonant_count-dissonant_count} mild, {dissonant_count} dissonant")
        print(f"    Avg consonance rank: {avg_rank:.1f} -> {chord_type}")

        # Show the specific intervals
        for lo, hi, name, rank in sorted(intervals, key=lambda x: x[3])[:4]:
            star = "***" if rank <= 3 else "**" if rank <= 7 else "*"
            print(f"      n={lo}:n={hi} = {hi/lo:.3f} = {name} {star}")


# =============================================================================
# Analysis 4: Injection safety scored by consonance
# =============================================================================

def analyze_injection_safety(model, dataset, config, pairs_data):
    """
    From Phase 3b we know character swaps work at ~80% rate.
    Score each potential swap by the consonance of the channels that would
    be affected. Prediction: swaps that disturb consonant pairs should
    cause more collateral damage.
    """
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 4: Injection Safety Scored by Consonance")
    print(f"  Do consonant channel disturbances cause more damage?")
    print(f"{'='*60}")

    model.eval()
    n_harmonics = config.n_embd // 2

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # For several character pairs, compute which bands differ most
    # and score those bands by consonance with other active bands
    test_pairs = [('e', 'a'), ('t', 's'), ('o', 'i'), ('h', 'n'), ('d', 'c'), ('r', 'l')]

    # Collect test contexts
    n_test = 300
    test_contexts = []
    step = max(1, (len(dataset.val_data) - config.block_size) // n_test)
    for i in range(0, len(dataset.val_data) - config.block_size - 1, step):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Baseline predictions
    baseline_preds = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits, _ = model(ctx)
            baseline_preds.append(torch.argmax(logits[:, -1, :], dim=-1).item())

    print(f"\n  {'Pair':>6} | {'Avg consonance':>16} | {'Swap rate':>10} | {'Preservation':>14} | {'Prediction':>11}")
    print(f"  {'-'*6}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*11}")

    pair_results = []

    for ca, cb in test_pairs:
        idx_a = dataset.stoi[ca]
        idx_b = dataset.stoi[cb]

        # Find which bands differ most between these two characters
        band_diffs = np.zeros(n_harmonics)
        for h in range(n_harmonics):
            ci, si = h * 2, h * 2 + 1
            band_diffs[h] = abs(orig_emb[idx_a, ci] - orig_emb[idx_b, ci]).item() + \
                           abs(orig_emb[idx_a, si] - orig_emb[idx_b, si]).item()

        # Top differing bands
        top_diff_bands = np.argsort(band_diffs)[::-1][:8]

        # Average consonance score of these bands with each other
        cons_scores = []
        for i in range(len(top_diff_bands)):
            for j in range(i+1, len(top_diff_bands)):
                n_i = top_diff_bands[i] + 1
                n_j = top_diff_bands[j] + 1
                th, _, rank = consonance_score(n_i, n_j)
                cons_scores.append(rank)
        avg_cons = np.mean(cons_scores) if cons_scores else 0

        # Do the actual swap and measure
        model.wte.weight.data[idx_a] = orig_emb[idx_b].clone()
        model.wte.weight.data[idx_b] = orig_emb[idx_a].clone()
        model.lm_head.weight.data[idx_a] = orig_head[idx_b].clone()
        model.lm_head.weight.data[idx_b] = orig_head[idx_a].clone()

        swapped_preds = []
        with torch.no_grad():
            for ctx in test_contexts:
                logits, _ = model(ctx)
                swapped_preds.append(torch.argmax(logits[:, -1, :], dim=-1).item())

        # Restore
        model.wte.weight.data = orig_emb.clone()
        model.lm_head.weight.data = orig_head.clone()

        # Measure
        a_to_b = sum(1 for bp, sp in zip(baseline_preds, swapped_preds) if bp == idx_a and sp == idx_b)
        b_to_a = sum(1 for bp, sp in zip(baseline_preds, swapped_preds) if bp == idx_b and sp == idx_a)
        total_a = sum(1 for bp in baseline_preds if bp == idx_a)
        total_b = sum(1 for bp in baseline_preds if bp == idx_b)
        total_other = sum(1 for bp in baseline_preds if bp != idx_a and bp != idx_b)
        unchanged = sum(1 for bp, sp in zip(baseline_preds, swapped_preds)
                        if bp != idx_a and bp != idx_b and bp == sp)

        swap_rate = ((a_to_b / total_a if total_a else 0) + (b_to_a / total_b if total_b else 0)) / 2 * 100
        preserve_rate = unchanged / total_other * 100 if total_other else 0

        # Higher consonance rank = more dissonant = predicted safer to edit
        safety = "SAFE" if avg_cons > 8 else "CAUTION" if avg_cons > 5 else "RISKY"

        print(f"  {ca}<->{cb} | {avg_cons:>14.1f}   | {swap_rate:>8.1f}% | {preserve_rate:>12.1f}% | {safety:>11}")

        pair_results.append({
            "pair": f"{ca}<->{cb}",
            "avg_consonance_rank": avg_cons,
            "swap_rate": swap_rate,
            "preserve_rate": preserve_rate,
        })

    # Correlation: does consonance predict preservation?
    cons_ranks = [r["avg_consonance_rank"] for r in pair_results]
    preserves = [r["preserve_rate"] for r in pair_results]
    swaps = [r["swap_rate"] for r in pair_results]

    if len(cons_ranks) > 2:
        corr_cons_pres = np.corrcoef(cons_ranks, preserves)[0, 1]
        corr_cons_swap = np.corrcoef(cons_ranks, swaps)[0, 1]
        print(f"\n  Correlation: consonance rank vs preservation: {corr_cons_pres:+.4f}")
        print(f"  Correlation: consonance rank vs swap rate:    {corr_cons_swap:+.4f}")
        print(f"  (Positive means: more dissonant pairs are easier to swap safely)")


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  MUSICAL HARMONIC ANALYSIS")
    print(f"  Do musical intervals predict channel behavior?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Analysis 1: Pure theory — the interval map
    n_harmonics = config.n_embd // 2
    analyze_interval_map(n_harmonics)

    # Train model for empirical tests
    model = train_model(config, dataset, mode="harmonic")

    # Analysis 2: Consonance vs independence
    pairs_data, corr_matrix = analyze_consonance_vs_independence(model, dataset, config)

    # Analysis 3: What chords do character types form?
    analyze_spectral_chords(model, dataset, config)

    # Analysis 4: Does consonance predict injection safety?
    analyze_injection_safety(model, dataset, config, pairs_data)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"""
  The mapping between musical intervals and harmonic channels:

  Channel ratio   Musical interval   Consonance
  n=2 : n=1       octave             perfect
  n=3 : n=2       perfect fifth      perfect
  n=4 : n=3       perfect fourth     perfect
  n=5 : n=4       major third        consonant
  n=6 : n=5       minor third        consonant
  n=8 : n=5       minor sixth        mild
  n=9 : n=8       major second       mild
  n=15 : n=8      major seventh      dissonant
  n=16 : n=15     minor second       dissonant

  The question: does this musical structure predict which channel
  pairs are load-bearing (consonant, don't touch) versus editable
  (dissonant, safe to modify)?
""")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
