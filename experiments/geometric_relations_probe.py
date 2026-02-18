"""
Geometric Relations Probe -- What harmonic relationships exist inside the transformer?

THE QUESTION:
Before we attempt to edit anything, we need to understand what's in there.
At each layer of the trained transformer, what geometric relationships exist
between the harmonic channels? Do they overlap or occupy distinct spaces?

If they overlap -> editing one relationship risks disturbing another.
If they're distinct -> targeted editing is feasible.

THE METHOD:
1. Train a harmonic transformer on Shakespeare
2. Feed diverse inputs through the model, capturing activations at every layer
3. At each layer, compute:
   a. Cross-channel coherence matrix (how channels relate to each other)
   b. Harmonic sweep: for each pair of channels, what harmonic relationships exist
   c. Relationship clustering: do channels group into identifiable geometric families?
4. Track how relationships evolve layer by layer
5. Identify overlaps vs distinct separation

This is Test 21 (harmonic sweep) applied to the INSIDE of the transformer.
"""

import math
import os
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
    dropout = 0.0
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 3000
    eval_interval = 500
    eval_iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model (same as spectral_persistence.py)
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

    def get_layer_activations(self, idx):
        """Get activations at each stage of the network."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        if self.mode == "baseline":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.wpe(pos)
        else:
            pos_emb = self.wpe(T)

        acts = {}
        acts["embedding"] = tok_emb.detach().cpu()

        x = tok_emb + pos_emb
        acts["embed+pos"] = x.detach().cpu()

        for i, block in enumerate(self.blocks):
            x_attn = x + block.attn(block.ln_1(x))
            acts[f"layer{i}_post_attn"] = x_attn.detach().cpu()
            x = x_attn + block.mlp(block.ln_2(x_attn))
            acts[f"layer{i}_post_mlp"] = x.detach().cpu()

        x = self.ln_f(x)
        acts["final"] = x.detach().cpu()
        return acts


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
# Training
# =============================================================================

def train_model(config, dataset, mode):
    print(f"\n{'='*60}")
    print(f"  Training: {mode.upper()}")
    print(f"{'='*60}")
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
# Geometric Relation Analysis
# =============================================================================

def compute_harmonic_coherence(vec_a, vec_b, n_harmonics=12):
    """
    Compute coherence at each harmonic between two activation vectors.
    Treats pairs of dimensions as cos/sin of each harmonic.

    Returns array of coherence values for harmonics 1..n_harmonics.
    """
    n_embd = len(vec_a)
    n_pairs = n_embd // 2
    max_h = min(n_harmonics, n_pairs)

    coherences = np.zeros(max_h)
    for h in range(max_h):
        cos_idx = h * 2
        sin_idx = h * 2 + 1
        # Coherence at this harmonic: dot product of the (cos, sin) pairs
        dot = vec_a[cos_idx] * vec_b[cos_idx] + vec_a[sin_idx] * vec_b[sin_idx]
        mag_a = math.sqrt(vec_a[cos_idx]**2 + vec_a[sin_idx]**2)
        mag_b = math.sqrt(vec_b[cos_idx]**2 + vec_b[sin_idx]**2)
        if mag_a > 1e-10 and mag_b > 1e-10:
            coherences[h] = dot / (mag_a * mag_b)
    return coherences


def compute_channel_energy(activations, n_embd):
    """
    For each harmonic channel (cos/sin pair), compute total energy across
    all batch items and positions.

    Returns: array of energy per harmonic [n_harmonics]
    """
    n_harmonics = n_embd // 2
    # activations: [batch, seq_len, n_embd]
    flat = activations.reshape(-1, n_embd)  # [batch*seq, n_embd]
    energies = np.zeros(n_harmonics)
    for h in range(n_harmonics):
        cos_e = np.sum(flat[:, h*2] ** 2)
        sin_e = np.sum(flat[:, h*2+1] ** 2)
        energies[h] = cos_e + sin_e
    return energies


def compute_cross_harmonic_coherence_matrix(activations, n_check=16):
    """
    For each pair of harmonic channels, compute how much they co-activate.

    If channel A and channel B fire independently -> low cross-coherence.
    If they fire together -> high cross-coherence (overlap).

    Returns: [n_check, n_check] matrix of absolute correlation values.
    """
    n_embd = activations.shape[2]
    n_harmonics = n_embd // 2
    n_check = min(n_check, n_harmonics)

    # Compute energy per position for each harmonic
    # activations: [batch, seq, n_embd]
    flat = activations.reshape(-1, n_embd)  # [N, n_embd]

    # Per-harmonic energy at each position: sqrt(cos^2 + sin^2)
    harmonic_energy = np.zeros((flat.shape[0], n_check))
    for h in range(n_check):
        harmonic_energy[:, h] = np.sqrt(flat[:, h*2]**2 + flat[:, h*2+1]**2)

    # Correlation matrix between harmonic energies
    # If two harmonics have correlated energy patterns, they overlap
    corr_matrix = np.zeros((n_check, n_check))
    for i in range(n_check):
        for j in range(n_check):
            a = harmonic_energy[:, i]
            b = harmonic_energy[:, j]
            a_c = a - a.mean()
            b_c = b - b.mean()
            denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c))
            if denom > 1e-10:
                corr_matrix[i, j] = np.dot(a_c, b_c) / denom

    return corr_matrix


def compute_relationship_types(activations, n_check=16):
    """
    For each pair of harmonic channels, classify the geometric relationship:
    - Independent: low correlation (|corr| < 0.2)
    - Cooperative: positive correlation (corr > 0.4) -- they fire together
    - Competitive: negative correlation (corr < -0.4) -- one suppresses the other
    - Coupled: moderate correlation -- partial overlap

    Returns counts and details.
    """
    corr = compute_cross_harmonic_coherence_matrix(activations, n_check)
    n = corr.shape[0]

    independent = 0
    cooperative = 0
    competitive = 0
    coupled = 0
    pairs_detail = []

    for i in range(n):
        for j in range(i+1, n):
            c = corr[i, j]
            if abs(c) < 0.2:
                rtype = "independent"
                independent += 1
            elif c > 0.4:
                rtype = "cooperative"
                cooperative += 1
            elif c < -0.4:
                rtype = "competitive"
                competitive += 1
            else:
                rtype = "coupled"
                coupled += 1
            pairs_detail.append((i+1, j+1, c, rtype))

    total = independent + cooperative + competitive + coupled
    return {
        "independent": independent,
        "cooperative": cooperative,
        "competitive": competitive,
        "coupled": coupled,
        "total": total,
        "matrix": corr,
        "pairs": pairs_detail,
    }


def analyze_geometric_relations(model, dataset, config):
    """Main analysis: probe all geometric relations at each layer."""
    model.eval()

    print(f"\n{'='*60}")
    print(f"  GEOMETRIC RELATIONS PROBE: {model.mode.upper()}")
    print(f"{'='*60}")

    # Collect activations from multiple batches for robust statistics
    all_acts = {}
    n_batches = 5
    print(f"\n  Collecting activations from {n_batches} batches...")

    for b in range(n_batches):
        x, _ = dataset.get_batch("val", config)
        with torch.no_grad():
            acts = model.get_layer_activations(x)
        for name, tensor in acts.items():
            if name not in all_acts:
                all_acts[name] = []
            all_acts[name].append(tensor.numpy())

    # Concatenate batches
    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name], axis=0)

    n_check = 16  # First 16 harmonics
    layer_names = list(all_acts.keys())

    # ==================================================================
    # 1. Energy distribution across harmonics at each layer
    # ==================================================================
    print(f"\n  1. Energy distribution across harmonics")
    print(f"  {'='*50}")
    print(f"  Which harmonics carry the most energy at each layer?")

    for name in layer_names:
        energy = compute_channel_energy(all_acts[name], config.n_embd)
        total_e = energy.sum()
        top_indices = np.argsort(energy)[::-1][:5]

        print(f"\n  {name}:")
        print(f"    Top 5 harmonics by energy:")
        for rank, idx in enumerate(top_indices):
            pct = 100.0 * energy[idx] / total_e if total_e > 0 else 0
            print(f"      #{rank+1}: n={idx+1:>2} -- {pct:5.1f}% of total energy")

        # How concentrated is the energy?
        sorted_e = np.sort(energy)[::-1]
        cumsum = np.cumsum(sorted_e) / total_e if total_e > 0 else np.zeros_like(sorted_e)
        n_for_50 = np.searchsorted(cumsum, 0.5) + 1
        n_for_90 = np.searchsorted(cumsum, 0.9) + 1
        print(f"    50% of energy in top {n_for_50} harmonics (of {len(energy)})")
        print(f"    90% of energy in top {n_for_90} harmonics (of {len(energy)})")

    # ==================================================================
    # 2. Cross-harmonic relationship classification at each layer
    # ==================================================================
    print(f"\n\n  2. Cross-harmonic relationships at each layer")
    print(f"  {'='*50}")
    print(f"  Do harmonics operate independently or overlap?")
    print(f"  (checking first {n_check} harmonics = {n_check*(n_check-1)//2} pairs)")

    layer_relations = {}
    for name in layer_names:
        rels = compute_relationship_types(all_acts[name], n_check)
        layer_relations[name] = rels

        total = rels["total"]
        print(f"\n  {name}:")
        print(f"    Independent (|corr|<0.2): {rels['independent']:>3} ({100*rels['independent']/total:5.1f}%)")
        print(f"    Coupled (0.2-0.4):        {rels['coupled']:>3} ({100*rels['coupled']/total:5.1f}%)")
        print(f"    Cooperative (corr>0.4):    {rels['cooperative']:>3} ({100*rels['cooperative']/total:5.1f}%)")
        print(f"    Competitive (corr<-0.4):   {rels['competitive']:>3} ({100*rels['competitive']/total:5.1f}%)")

    # ==================================================================
    # 3. Relationship evolution: how do relations change layer by layer?
    # ==================================================================
    print(f"\n\n  3. Relationship evolution across layers")
    print(f"  {'='*50}")

    print(f"\n  {'Layer':<25} {'Indep%':>8} {'Coupled%':>9} {'Coop%':>8} {'Compet%':>8} {'MeanAbsCorr':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*12}")

    for name in layer_names:
        r = layer_relations[name]
        t = r["total"]
        mac = np.mean(np.abs(r["matrix"][np.triu_indices(n_check, k=1)]))
        print(f"  {name:<25} {100*r['independent']/t:>7.1f}% {100*r['coupled']/t:>8.1f}% "
              f"{100*r['cooperative']/t:>7.1f}% {100*r['competitive']/t:>7.1f}% {mac:>12.4f}")

    # ==================================================================
    # 4. Detailed overlap matrix at embedding and final layers
    # ==================================================================
    print(f"\n\n  4. Cross-harmonic correlation matrices")
    print(f"  {'='*50}")

    for layer_name in ["embedding", "final"]:
        matrix = layer_relations[layer_name]["matrix"]
        print(f"\n  {layer_name.upper()} layer (first 8 harmonics):")
        n_show = min(8, n_check)
        header = "         " + "".join([f"  n={h+1:<4}" for h in range(n_show)])
        print(f"  {header}")
        print(f"  {'-'*9}" + "-" * (8 * n_show))
        for i in range(n_show):
            row = f"  n={i+1:<6}"
            for j in range(n_show):
                val = matrix[i, j]
                if i == j:
                    row += f"  {'--':>6}"
                else:
                    row += f"  {val:>6.3f}"
            print(row)

    # ==================================================================
    # 5. Which specific pairs are most entangled?
    # ==================================================================
    print(f"\n\n  5. Most entangled harmonic pairs at final layer")
    print(f"  {'='*50}")

    final_pairs = layer_relations["final"]["pairs"]
    sorted_pairs = sorted(final_pairs, key=lambda p: abs(p[2]), reverse=True)

    print(f"\n  Top 10 most correlated pairs:")
    print(f"  {'Pair':<12} {'Correlation':>12} {'Type':<15}")
    print(f"  {'-'*12} {'-'*12} {'-'*15}")
    for i, (h1, h2, corr, rtype) in enumerate(sorted_pairs[:10]):
        print(f"  n={h1:<2} <-> n={h2:<2} {corr:>12.4f} {rtype:<15}")

    print(f"\n  Top 10 most independent pairs:")
    sorted_indep = sorted(final_pairs, key=lambda p: abs(p[2]))
    for i, (h1, h2, corr, rtype) in enumerate(sorted_indep[:10]):
        print(f"  n={h1:<2} <-> n={h2:<2} {corr:>12.4f} {rtype:<15}")

    # ==================================================================
    # 6. Overlap assessment for editability
    # ==================================================================
    print(f"\n\n  6. EDITABILITY ASSESSMENT")
    print(f"  {'='*50}")

    final_rels = layer_relations["final"]
    emb_rels = layer_relations["embedding"]

    final_indep_pct = 100 * final_rels["independent"] / final_rels["total"]
    emb_indep_pct = 100 * emb_rels["independent"] / emb_rels["total"]

    final_mac = np.mean(np.abs(final_rels["matrix"][np.triu_indices(n_check, k=1)]))
    emb_mac = np.mean(np.abs(emb_rels["matrix"][np.triu_indices(n_check, k=1)]))

    print(f"\n  Independent pairs at embedding layer: {emb_indep_pct:.1f}%")
    print(f"  Independent pairs at final layer:     {final_indep_pct:.1f}%")
    print(f"  Mean |correlation| at embedding:       {emb_mac:.4f}")
    print(f"  Mean |correlation| at final:           {final_mac:.4f}")

    # Find channels that are most independent from all others
    final_matrix = final_rels["matrix"]
    channel_isolation = np.zeros(n_check)
    for i in range(n_check):
        others = [abs(final_matrix[i, j]) for j in range(n_check) if i != j]
        channel_isolation[i] = 1.0 - np.mean(others)  # Higher = more isolated

    print(f"\n  Most isolated harmonics at final layer (best candidates for editing):")
    isolated_order = np.argsort(channel_isolation)[::-1]
    for rank, idx in enumerate(isolated_order[:8]):
        mean_overlap = 1.0 - channel_isolation[idx]
        print(f"    #{rank+1}: n={idx+1:>2} -- mean overlap with others: {mean_overlap:.4f}")

    print(f"\n  Most entangled harmonics (worst candidates for editing):")
    for rank, idx in enumerate(isolated_order[-4:]):
        mean_overlap = 1.0 - channel_isolation[idx]
        print(f"    #{rank+1}: n={idx+1:>2} -- mean overlap with others: {mean_overlap:.4f}")

    if final_indep_pct > 50:
        print(f"\n  VERDICT: Majority of harmonic pairs are independent ({final_indep_pct:.0f}%).")
        print(f"  Targeted editing by frequency band is FEASIBLE for isolated channels.")
    elif final_indep_pct > 30:
        print(f"\n  VERDICT: Significant independence ({final_indep_pct:.0f}%), with overlap in some pairs.")
        print(f"  Editing is feasible but requires careful channel selection.")
        print(f"  Use the isolated harmonics above as primary editing targets.")
    else:
        print(f"\n  VERDICT: High overlap ({100-final_indep_pct:.0f}% of pairs are correlated).")
        print(f"  Direct frequency-band editing risks disturbing adjacent channels.")
        print(f"  Consider: editing at the embedding level before MLP mixing occurs,")
        print(f"  or developing harmonic-preserving MLP architectures.")

    return {
        "layer_relations": layer_relations,
        "channel_isolation": channel_isolation,
        "final_indep_pct": final_indep_pct,
        "emb_indep_pct": emb_indep_pct,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  GEOMETRIC RELATIONS PROBE")
    print(f"  What harmonic relationships exist inside the transformer?")
    print(f"  Do they overlap or stay distinct?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    results = {}
    for mode in ["harmonic", "baseline"]:
        model = train_model(config, dataset, mode)
        results[mode] = analyze_geometric_relations(model, dataset, config)

    # ==================================================================
    # Comparative summary
    # ==================================================================
    print(f"\n\n{'='*60}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<40} {'Harmonic':>10} {'Baseline':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    print(f"  {'Independent pairs at embedding':<40} {results['harmonic']['emb_indep_pct']:>9.1f}% {results['baseline']['emb_indep_pct']:>9.1f}%")
    print(f"  {'Independent pairs at final layer':<40} {results['harmonic']['final_indep_pct']:>9.1f}% {results['baseline']['final_indep_pct']:>9.1f}%")

    h_iso = results["harmonic"]["channel_isolation"]
    b_iso = results["baseline"]["channel_isolation"]
    print(f"  {'Mean channel isolation (final)':<40} {np.mean(h_iso):>10.4f} {np.mean(b_iso):>10.4f}")
    print(f"  {'Max channel isolation (final)':<40} {np.max(h_iso):>10.4f} {np.max(b_iso):>10.4f}")

    print(f"\n  Key question: Does harmonic initialization produce more")
    print(f"  independent channels than random initialization?")

    h_final = results["harmonic"]["final_indep_pct"]
    b_final = results["baseline"]["final_indep_pct"]
    if h_final > b_final + 5:
        print(f"  -> YES: Harmonic has {h_final-b_final:.1f}% more independent pairs.")
        print(f"  Harmonic structure creates separable channels that survive training.")
    elif abs(h_final - b_final) < 5:
        print(f"  -> SIMILAR: Both models have comparable independence levels.")
        print(f"  The network's own training dynamics dominate channel structure.")
    else:
        print(f"  -> NO: Baseline has {b_final-h_final:.1f}% more independent pairs.")
        print(f"  Random initialization may allow more flexible channel allocation.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
