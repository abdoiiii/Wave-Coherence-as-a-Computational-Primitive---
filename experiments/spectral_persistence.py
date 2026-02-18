"""
Spectral Persistence Test — Does harmonic structure survive through transformer layers?

THE QUESTION:
We built harmonic embeddings with known frequency structure. When those embeddings
flow through attention and MLP layers, does the harmonic organization persist or
does it get scrambled?

If it persists -> knowledge is addressable by frequency band through the entire network.
If it scrambles -> we know where structure breaks and can investigate why.

THE METHOD:
1. Train a harmonic transformer on Shakespeare (reuse existing code)
2. Feed input through the trained model, capturing activations at every layer
3. Compute FFT of activations at each layer
4. Measure how much of the original harmonic structure survives
5. Compare: harmonic model vs baseline (random init)

KEY METRICS:
- Spectral coherence between embedding layer and each subsequent layer
- Energy concentration: what % of spectral energy stays in the original harmonic bands
- Per-harmonic tracking: do individual frequency channels maintain identity through layers
"""

import math
import os
import sys
import time
import urllib.request

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
    dropout = 0.0       # No dropout for analysis — we want deterministic activations
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 3000     # Enough to learn patterns, fast enough to iterate
    eval_interval = 500
    eval_iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model (from harmonic_transformer.py, with activation capture hooks)
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
    def __init__(self, config, vocab_size, mode="harmonic"):
        super().__init__()
        self.config = config
        self.mode = mode

        if mode == "baseline":
            self.wte = nn.Embedding(vocab_size, config.n_embd)
            nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
            nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        elif mode in ("harmonic", "frozen"):
            trainable = (mode == "harmonic")
            self.wte = HarmonicEmbedding(vocab_size, config.n_embd, trainable=trainable)
            self.wpe = HarmonicPositionalEncoding(config.block_size, config.n_embd, trainable=trainable)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        if mode == "baseline":
            self.wte.weight = self.lm_head.weight

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
        if self.mode == "baseline":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.wpe(pos)
        else:
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

    def forward_with_activations(self, idx):
        """Forward pass that captures activations at every layer."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        if self.mode == "baseline":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.wpe(pos)
        else:
            pos_emb = self.wpe(T)

        activations = {}
        activations["embedding"] = tok_emb.detach().cpu()
        activations["embed+pos"] = (tok_emb + pos_emb).detach().cpu()

        x = tok_emb + pos_emb
        for i, block in enumerate(self.blocks):
            # Capture pre-attention (after layernorm)
            x_ln1 = block.ln_1(x)
            activations[f"layer{i}_pre_attn"] = x_ln1.detach().cpu()

            # Through attention
            x = x + block.attn(x_ln1)
            activations[f"layer{i}_post_attn"] = x.detach().cpu()

            # Through MLP
            x_ln2 = block.ln_2(x)
            x = x + block.mlp(x_ln2)
            activations[f"layer{i}_post_mlp"] = x.detach().cpu()

        x = self.ln_f(x)
        activations["final"] = x.detach().cpu()

        return activations


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
            print("  Done.")
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

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)


# =============================================================================
# Training
# =============================================================================

def train_model(config, dataset, mode):
    print(f"\n{'='*60}")
    print(f"  Training: {mode.upper()}")
    print(f"{'='*60}")

    model = HarmonicGPT(config, dataset.vocab_size, mode=mode).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    start = time.time()
    for iteration in range(config.max_iters):
        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
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
            print(f"  step {iteration:>5} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total = time.time() - start
    print(f"  Training complete in {total:.1f}s")
    return model


# =============================================================================
# Spectral Analysis
# =============================================================================

def compute_spectral_profile(activations_tensor):
    """
    Compute the FFT-based spectral profile of activations.

    Input: tensor of shape [batch, seq_len, n_embd]
    Output: spectral energy per frequency band, averaged across batch and positions
    """
    # Average across batch and positions to get a representative activation vector
    # Shape: [n_embd]
    mean_activation = activations_tensor.mean(dim=(0, 1)).numpy()

    # Compute FFT
    spectrum = np.fft.fft(mean_activation)
    magnitudes = np.abs(spectrum[:len(spectrum)//2])

    return magnitudes


def compute_spectral_coherence(spec_a, spec_b):
    """
    Compute coherence between two spectral profiles.
    Returns a single number: how similar the frequency distributions are.
    """
    # Normalize
    a = spec_a / (np.linalg.norm(spec_a) + 1e-10)
    b = spec_b / (np.linalg.norm(spec_b) + 1e-10)
    return float(np.dot(a, b))


def compute_energy_concentration(spectrum, n_harmonics_to_check=16):
    """
    What fraction of total spectral energy is concentrated in the first N harmonics?
    High concentration = structure preserved. Low = scrambled.
    """
    total = np.sum(spectrum ** 2)
    if total < 1e-10:
        return 0.0
    top_n = np.sum(np.sort(spectrum ** 2)[-n_harmonics_to_check:])
    return float(top_n / total)


def compute_per_channel_correlation(act_embedding, act_layer):
    """
    For each dimension (harmonic channel) in the embedding, compute
    the correlation with the same dimension at a later layer.

    High correlation = that harmonic channel maintained its identity.
    Low correlation = the channel was scrambled/mixed with others.

    Input: two tensors of shape [batch, seq_len, n_embd]
    Output: array of per-channel correlations [n_embd]
    """
    n_embd = act_embedding.shape[2]
    correlations = np.zeros(n_embd)

    for dim in range(n_embd):
        emb_vals = act_embedding[:, :, dim].flatten().numpy()
        layer_vals = act_layer[:, :, dim].flatten().numpy()

        # Pearson correlation
        emb_centered = emb_vals - emb_vals.mean()
        layer_centered = layer_vals - layer_vals.mean()
        numer = np.dot(emb_centered, layer_centered)
        denom = np.sqrt(np.dot(emb_centered, emb_centered) * np.dot(layer_centered, layer_centered))
        if denom > 1e-10:
            correlations[dim] = numer / denom
        else:
            correlations[dim] = 0.0

    return correlations


def analyze_spectral_persistence(model, dataset, config):
    """
    Main analysis: capture activations and measure spectral persistence.
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"  SPECTRAL PERSISTENCE ANALYSIS: {model.mode.upper()}")
    print(f"{'='*60}")

    # Get a batch of real data
    x, _ = dataset.get_batch("val", config)

    with torch.no_grad():
        activations = model.forward_with_activations(x)

    # ------------------------------------------------------------------
    # 1. Spectral profiles at each layer
    # ------------------------------------------------------------------
    print(f"\n  1. Spectral profiles across layers")
    print(f"  {'-'*50}")

    layer_names = [k for k in activations.keys()]
    spectra = {}
    for name in layer_names:
        spectra[name] = compute_spectral_profile(activations[name])

    # Coherence between embedding and each subsequent layer
    ref_spectrum = spectra["embedding"]
    print(f"\n  Spectral coherence with embedding layer:")
    print(f"  {'Layer':<25} {'Coherence':>10} {'Energy Conc.':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*14}")

    coherence_results = {}
    for name in layer_names:
        coh = compute_spectral_coherence(ref_spectrum, spectra[name])
        ec = compute_energy_concentration(spectra[name])
        coherence_results[name] = coh
        print(f"  {name:<25} {coh:>10.4f} {ec:>14.4f}")

    # ------------------------------------------------------------------
    # 2. Per-channel correlation tracking
    # ------------------------------------------------------------------
    print(f"\n  2. Per-channel correlation (embedding -> each layer)")
    print(f"  {'-'*50}")

    ref_act = activations["embedding"]
    print(f"\n  {'Layer':<25} {'Mean corr':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    channel_results = {}
    for name in layer_names:
        if name == "embedding":
            continue
        corr = compute_per_channel_correlation(ref_act, activations[name])
        channel_results[name] = corr
        print(f"  {name:<25} {corr.mean():>10.4f} {np.median(corr):>10.4f} "
              f"{corr.min():>10.4f} {corr.max():>10.4f} {corr.std():>10.4f}")

    # ------------------------------------------------------------------
    # 3. Cos/sin pair analysis — do harmonic pairs stay coupled?
    # ------------------------------------------------------------------
    print(f"\n  3. Harmonic pair coupling (cos/sin pairs)")
    print(f"  {'-'*50}")
    print(f"  Do cos(n*theta)/sin(n*theta) pairs maintain their relationship?")

    n_harmonics = config.n_embd // 2
    final_act = activations["final"]
    embed_act = activations["embedding"]

    pair_coherences = []
    for h in range(min(n_harmonics, 16)):  # Check first 16 harmonics
        cos_idx = h * 2
        sin_idx = h * 2 + 1

        # In embedding: cos and sin of same harmonic are 90° out of phase
        emb_cos = embed_act[:, :, cos_idx].flatten().numpy()
        emb_sin = embed_act[:, :, sin_idx].flatten().numpy()

        # In final layer: are they still coupled?
        fin_cos = final_act[:, :, cos_idx].flatten().numpy()
        fin_sin = final_act[:, :, sin_idx].flatten().numpy()

        # Cross-correlation between cos/sin in embedding vs final
        emb_pair_corr = np.corrcoef(emb_cos, emb_sin)[0, 1]
        fin_pair_corr = np.corrcoef(fin_cos, fin_sin)[0, 1]

        # How much did the cos channel maintain identity?
        cos_persist = np.corrcoef(emb_cos, fin_cos)[0, 1] if np.std(fin_cos) > 1e-10 else 0.0
        sin_persist = np.corrcoef(emb_sin, fin_sin)[0, 1] if np.std(fin_sin) > 1e-10 else 0.0

        pair_coherences.append((h+1, cos_persist, sin_persist, emb_pair_corr, fin_pair_corr))

    print(f"\n  {'Harmonic':>9} {'cos persist':>12} {'sin persist':>12} {'pair corr(emb)':>15} {'pair corr(final)':>17}")
    print(f"  {'-'*9} {'-'*12} {'-'*12} {'-'*15} {'-'*17}")
    for h, cp, sp, epc, fpc in pair_coherences:
        print(f"  n={h:<6} {cp:>12.4f} {sp:>12.4f} {epc:>15.4f} {fpc:>17.4f}")

    avg_cos_persist = np.mean([c[1] for c in pair_coherences])
    avg_sin_persist = np.mean([c[2] for c in pair_coherences])
    print(f"\n  Average cos persistence: {avg_cos_persist:.4f}")
    print(f"  Average sin persistence: {avg_sin_persist:.4f}")

    # ------------------------------------------------------------------
    # 4. Cross-channel leakage — does harmonic N bleed into harmonic M?
    # ------------------------------------------------------------------
    print(f"\n  4. Cross-channel leakage matrix (first 8 harmonics)")
    print(f"  {'-'*50}")
    print(f"  How much does each embedding harmonic correlate with each final-layer dimension?")

    n_check = min(8, n_harmonics)
    leakage = np.zeros((n_check, n_check))

    for h_emb in range(n_check):
        emb_channel = embed_act[:, :, h_emb * 2].flatten().numpy()  # cos channel
        for h_fin in range(n_check):
            fin_channel = final_act[:, :, h_fin * 2].flatten().numpy()
            if np.std(emb_channel) > 1e-10 and np.std(fin_channel) > 1e-10:
                leakage[h_emb, h_fin] = abs(np.corrcoef(emb_channel, fin_channel)[0, 1])

    print(f"\n  |corr| between embedding harmonic (row) and final-layer harmonic (col):")
    header = "  emb\\fin  " + "".join([f"  n={h+1:<4}" for h in range(n_check)])
    print(header)
    print(f"  {'-'*9}" + "-" * (8 * n_check))
    for h_emb in range(n_check):
        row = f"  n={h_emb+1:<6}"
        for h_fin in range(n_check):
            val = leakage[h_emb, h_fin]
            row += f"  {val:.4f}"
        print(row)

    # Diagonal vs off-diagonal
    diag_mean = np.mean(np.diag(leakage))
    off_diag = leakage[~np.eye(n_check, dtype=bool)]
    off_diag_mean = np.mean(off_diag)
    print(f"\n  Diagonal mean (same-channel persistence): {diag_mean:.4f}")
    print(f"  Off-diagonal mean (cross-channel leakage): {off_diag_mean:.4f}")
    print(f"  Ratio (higher = better separation):        {diag_mean / (off_diag_mean + 1e-10):.2f}x")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {model.mode.upper()}")
    print(f"{'='*60}")

    final_coherence = coherence_results.get("final", 0)
    print(f"\n  Spectral coherence (embedding -> final): {final_coherence:.4f}")
    print(f"  Average channel persistence (cos):       {avg_cos_persist:.4f}")
    print(f"  Average channel persistence (sin):       {avg_sin_persist:.4f}")
    print(f"  Same-channel correlation:                {diag_mean:.4f}")
    print(f"  Cross-channel leakage:                   {off_diag_mean:.4f}")
    print(f"  Channel separation ratio:                {diag_mean / (off_diag_mean + 1e-10):.2f}x")

    if diag_mean > 2 * off_diag_mean:
        print(f"\n  VERDICT: Harmonic structure PERSISTS through layers.")
        print(f"  Same-channel signal is {diag_mean / (off_diag_mean + 1e-10):.1f}x stronger than cross-channel leakage.")
        print(f"  Knowledge may be addressable by frequency band.")
    elif diag_mean > off_diag_mean:
        print(f"\n  VERDICT: Partial persistence — some structure survives but with significant mixing.")
        print(f"  Further investigation needed on which layers cause the most scrambling.")
    else:
        print(f"\n  VERDICT: Harmonic structure is SCRAMBLED by the network.")
        print(f"  Attention/MLP layers mix channels. Direct frequency-band editing may not work.")
        print(f"  Possible next step: harmonic attention (constrain attention to preserve frequency structure).")

    return {
        "spectral_coherence": final_coherence,
        "cos_persistence": avg_cos_persist,
        "sin_persistence": avg_sin_persist,
        "diag_mean": diag_mean,
        "off_diag_mean": off_diag_mean,
        "separation_ratio": diag_mean / (off_diag_mean + 1e-10),
        "channel_results": channel_results,
        "leakage_matrix": leakage,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  SPECTRAL PERSISTENCE TEST")
    print(f"  Does harmonic structure survive through transformer layers?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    # Load data
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train and analyze both models
    results = {}

    for mode in ["harmonic", "baseline"]:
        model = train_model(config, dataset, mode)
        results[mode] = analyze_spectral_persistence(model, dataset, config)

    # =================================================================
    # Comparative summary
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  COMPARATIVE RESULTS")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<35} {'Harmonic':>10} {'Baseline':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    for key in ["spectral_coherence", "cos_persistence", "sin_persistence",
                 "diag_mean", "off_diag_mean", "separation_ratio"]:
        h_val = results["harmonic"][key]
        b_val = results["baseline"][key]
        print(f"  {key:<35} {h_val:>10.4f} {b_val:>10.4f}")

    print(f"\n  If harmonic model shows significantly higher channel separation")
    print(f"  than baseline, then harmonic structure provides addressable")
    print(f"  frequency channels that survive through the network — enabling")
    print(f"  targeted knowledge editing by frequency band.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
