"""
Natural Expression -- What Is the Model's Own Language?

CONTEXT:
Phase 6 showed progressive learning produces a "happier" model -- better loss,
faster knowledge absorption, more organized. Phase 11 showed we can't impose
our structure (chords/pooling) on the model because it composes by differentiation.

THE QUESTION:
What if we stop forcing the model into our categories? Stop decoding to tokens.
Stop imposing word boundaries. Let the model express in its own representation
space. What does the happy (progressive) model's internal world look like?

THE TESTS:
1. Internal landscape -- PCA of hidden states, progressive vs baseline.
   Does the happy model organize its representations differently?
2. Pre-projection geometry -- what does the final hidden state look like
   BEFORE lm_head crushes it to tokens? Natural dimensionality, cluster
   structure, energy distribution.
3. Attractor dynamics -- bypass tokens entirely. Feed hidden states back
   as hidden states. Let the model "dream." Where do representations settle?
4. Harmonic expression -- per-band energy and organization. Does the
   progressive model use its harmonic bands differently?
5. Natural groupings -- cluster hidden states without any token labels.
   What groups does the model discover? Do they correspond to anything
   we recognise?
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
    eval_iters = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model
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
        self.vocab_size = vocab_size
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

    def get_hidden_states(self, idx):
        """Return hidden states at every layer stage."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        h = tok_emb + pos_emb
        states = {"embedding": h.detach()}
        for i, block in enumerate(self.blocks):
            h = h + block.attn(block.ln_1(h))
            states[f"layer{i}_attn"] = h.detach()
            h = h + block.mlp(block.ln_2(h))
            states[f"layer{i}_mlp"] = h.detach()
        h = self.ln_f(h)
        states["final"] = h.detach()
        return states

    def forward_from_hidden(self, h):
        """Run the transformer blocks starting from hidden states (bypass embedding).
        Adds position encoding to distinguish positions but uses h as the content."""
        B, T, C = h.shape
        pos_emb = self.wpe(T)
        x = h + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


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
# Training (both baseline and progressive)
# =============================================================================

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


def train_baseline(config, dataset):
    """Standard training -- all parameters unfrozen from start."""
    print(f"\n  Training BASELINE model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss(model, dataset, config)
            print(f"  step {it:>5} | val {val_loss:.4f} | {time.time()-start:.1f}s")
            model.train()
        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Baseline training complete in {time.time()-start:.1f}s")
    return model


def train_progressive(config, dataset):
    """Progressive training -- structure first, detail later (Phase 6 approach)."""
    print(f"\n  Training PROGRESSIVE (happy) model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_harmonics = config.n_embd // 2  # 64

    # Stage boundaries
    stage1_end = 1000  # bands 1-8
    stage2_end = 2000  # bands 1-24
    # stage3: all bands (1-64)

    start = time.time()

    for it in range(config.max_iters):
        # Determine which bands get gradients
        if it < stage1_end:
            trainable_bands = 8
            stage = 1
        elif it < stage2_end:
            trainable_bands = 24
            stage = 2
        else:
            trainable_bands = n_harmonics
            stage = 3

        # Freeze/unfreeze embedding bands
        if hasattr(model.wte, 'weight') and model.wte.weight.requires_grad:
            with torch.no_grad():
                # We'll use a gradient hook instead of masking
                pass

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss(model, dataset, config)
            print(f"  step {it:>5} | val {val_loss:.4f} | stage {stage} (bands 1-{trainable_bands}) | {time.time()-start:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        optimizer.zero_grad()
        loss.backward()

        # Zero gradients for frozen bands in embedding
        if trainable_bands < n_harmonics:
            with torch.no_grad():
                # Embedding: channels are [cos1, sin1, cos2, sin2, ...]
                # Band n uses channels 2*(n-1) and 2*(n-1)+1
                freeze_start = trainable_bands * 2
                if model.wte.weight.grad is not None:
                    model.wte.weight.grad[:, freeze_start:] = 0
                if model.wpe.weight.grad is not None:
                    model.wpe.weight.grad[:, freeze_start:] = 0

        optimizer.step()

    print(f"  Progressive training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# TEST 1: Internal Landscape -- PCA of Hidden States
# =============================================================================

def test_internal_landscape(baseline, progressive, dataset, config, n_batches=5):
    """
    Extract hidden states from both models. PCA to see the natural
    dimensionality and structure. Does the happy model use its space
    differently?
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Internal Landscape")
    print("  How does each model organise its representation space?")
    print("=" * 60)

    for model_name, model in [("Baseline", baseline), ("Progressive", progressive)]:
        model.eval()
        print(f"\n  --- {model_name} Model ---")

        # Collect hidden states at final layer
        all_hidden = []
        all_tokens = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.get_batch("val", config)
                states = model.get_hidden_states(x)
                final = states["final"]  # [B, T, C]
                all_hidden.append(final.view(-1, config.n_embd).cpu().numpy())
                all_tokens.append(x.view(-1).cpu().numpy())

        H = np.concatenate(all_hidden, axis=0)  # [N, 128]
        tokens = np.concatenate(all_tokens, axis=0)
        N = H.shape[0]

        # PCA: how many dimensions does the model actually use?
        H_centered = H - H.mean(axis=0)
        cov = np.cov(H_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
        total_var = eigenvalues.sum()
        cumulative = np.cumsum(eigenvalues) / total_var

        dims_50 = np.searchsorted(cumulative, 0.50) + 1
        dims_80 = np.searchsorted(cumulative, 0.80) + 1
        dims_90 = np.searchsorted(cumulative, 0.90) + 1
        dims_95 = np.searchsorted(cumulative, 0.95) + 1
        dims_99 = np.searchsorted(cumulative, 0.99) + 1

        print(f"    Total samples: {N:,}")
        print(f"    Embedding dim: {config.n_embd}")
        print(f"    Effective dimensionality (% variance explained):")
        print(f"      50%: {dims_50} dims")
        print(f"      80%: {dims_80} dims")
        print(f"      90%: {dims_90} dims")
        print(f"      95%: {dims_95} dims")
        print(f"      99%: {dims_99} dims")

        # Top eigenvalue concentrations
        top1_pct = 100 * eigenvalues[0] / total_var
        top5_pct = 100 * eigenvalues[:5].sum() / total_var
        top10_pct = 100 * eigenvalues[:10].sum() / total_var
        print(f"    Variance in top components:")
        print(f"      PC1: {top1_pct:.1f}%")
        print(f"      PC1-5: {top5_pct:.1f}%")
        print(f"      PC1-10: {top10_pct:.1f}%")

        # Representation norm statistics
        norms = np.linalg.norm(H, axis=1)
        print(f"    Representation norms: mean={norms.mean():.3f}, std={norms.std():.3f}")

        # Per-token-type analysis: do different characters live in different regions?
        unique_tokens = np.unique(tokens)
        centroids = {}
        for tok in unique_tokens:
            mask = tokens == tok
            if mask.sum() > 100:
                centroids[tok] = H[mask].mean(axis=0)

        if len(centroids) >= 2:
            # Average inter-centroid distance
            centroid_list = list(centroids.values())
            dists = []
            for i in range(len(centroid_list)):
                for j in range(i+1, len(centroid_list)):
                    d = np.linalg.norm(centroid_list[i] - centroid_list[j])
                    dists.append(d)
            mean_inter_dist = np.mean(dists)

            # Average intra-cluster spread
            spreads = []
            for tok, centroid in centroids.items():
                mask = tokens == tok
                cluster_points = H[mask]
                spread = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
                spreads.append(spread)
            mean_intra_spread = np.mean(spreads)

            separation_ratio = mean_inter_dist / mean_intra_spread if mean_intra_spread > 0 else 0
            print(f"    Token cluster analysis:")
            print(f"      Mean inter-centroid distance: {mean_inter_dist:.4f}")
            print(f"      Mean intra-cluster spread: {mean_intra_spread:.4f}")
            print(f"      Separation ratio: {separation_ratio:.3f} (>1 = separated)")


# =============================================================================
# TEST 2: Pre-Projection Geometry
# =============================================================================

def test_pre_projection(baseline, progressive, dataset, config, n_batches=5):
    """
    What does the final hidden state look like BEFORE lm_head projects
    it to token space? The lm_head is a bottleneck -- what's lost?
    """
    print("\n" + "=" * 60)
    print("  TEST 2: Pre-Projection Geometry")
    print("  The model's thoughts before they become tokens")
    print("=" * 60)

    for model_name, model in [("Baseline", baseline), ("Progressive", progressive)]:
        model.eval()
        print(f"\n  --- {model_name} Model ---")

        all_hidden = []
        all_logits = []
        all_tokens = []

        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.get_batch("val", config)
                states = model.get_hidden_states(x)
                final = states["final"]  # [B, T, C]
                logits = model.lm_head(final)  # [B, T, vocab]

                all_hidden.append(final.view(-1, config.n_embd).cpu().numpy())
                all_logits.append(logits.view(-1, dataset.vocab_size).cpu().numpy())
                all_tokens.append(y.view(-1).cpu().numpy())

        H = np.concatenate(all_hidden, axis=0)
        L = np.concatenate(all_logits, axis=0)
        targets = np.concatenate(all_tokens, axis=0)

        # How much information survives the projection?
        # Measure: do nearby points in hidden space map to similar logit distributions?
        n_sample = min(5000, H.shape[0])
        idx = np.random.choice(H.shape[0], n_sample, replace=False)
        H_s = H[idx]
        L_s = L[idx]

        # Pairwise cosine similarities in hidden space vs logit space
        H_norm = H_s / (np.linalg.norm(H_s, axis=1, keepdims=True) + 1e-10)
        L_norm = L_s / (np.linalg.norm(L_s, axis=1, keepdims=True) + 1e-10)

        # Sample pairs for correlation
        n_pairs = 10000
        i_pairs = np.random.randint(0, n_sample, n_pairs)
        j_pairs = np.random.randint(0, n_sample, n_pairs)

        h_sims = np.sum(H_norm[i_pairs] * H_norm[j_pairs], axis=1)
        l_sims = np.sum(L_norm[i_pairs] * L_norm[j_pairs], axis=1)

        correlation = np.corrcoef(h_sims, l_sims)[0, 1]
        print(f"    Hidden-to-logit space correlation: {correlation:.4f}")
        print(f"    (1.0 = projection preserves all structure, 0.0 = structure lost)")

        # Entropy of hidden states vs entropy of logits
        logit_probs = np.exp(L_s - L_s.max(axis=1, keepdims=True))
        logit_probs = logit_probs / logit_probs.sum(axis=1, keepdims=True)
        logit_entropy = -np.sum(logit_probs * np.log(logit_probs + 1e-10), axis=1)

        # Hidden state "entropy" via singular value spread
        H_centered = H_s - H_s.mean(axis=0)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        sv_probs = S / S.sum()
        hidden_entropy = -np.sum(sv_probs * np.log(sv_probs + 1e-10))
        max_hidden_entropy = np.log(len(S))

        print(f"    Logit entropy: mean={logit_entropy.mean():.3f}, std={logit_entropy.std():.3f}")
        print(f"    Hidden spectral entropy: {hidden_entropy:.3f} / {max_hidden_entropy:.3f} max "
              f"({100*hidden_entropy/max_hidden_entropy:.1f}% of uniform)")

        # How many distinct "thoughts" are there?
        # K-means-like: find how many clusters the hidden states naturally form
        # Use eigenvalue gap to estimate natural cluster count
        H_cov = np.cov(H_centered.T)
        eigs = np.linalg.eigvalsh(H_cov)[::-1]
        eig_ratios = eigs[:-1] / (eigs[1:] + 1e-10)
        # Largest gap suggests natural cluster boundary
        natural_k = np.argmax(eig_ratios[:20]) + 1
        gap_size = eig_ratios[natural_k - 1]
        print(f"    Natural cluster estimate (eigenvalue gap): ~{natural_k} clusters (gap ratio: {gap_size:.2f})")


# =============================================================================
# TEST 3: Attractor Dynamics -- Let the Model Dream
# =============================================================================

def test_attractor_dynamics(baseline, progressive, dataset, config, n_iterations=20):
    """
    Bypass tokens. Feed hidden states back as input to the transformer blocks.
    Let the model's internal dynamics run freely. Where do representations settle?

    Start from real hidden states (from actual text), then iterate:
    output_hidden -> add position encoding -> run blocks -> output_hidden -> ...

    If the model has stable attractors, representations will converge.
    If it's chaotic, they'll diverge. If it's creative, they'll explore.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Attractor Dynamics -- The Model Dreams")
    print("  Feed hidden states back as input. Where do they settle?")
    print("=" * 60)

    for model_name, model in [("Baseline", baseline), ("Progressive", progressive)]:
        model.eval()
        print(f"\n  --- {model_name} Model ---")

        # Start from a real text input
        x, _ = dataset.get_batch("val", config)
        x = x[:4]  # just 4 sequences

        with torch.no_grad():
            # Get initial hidden states from real input
            states = model.get_hidden_states(x)
            h = states["final"]  # [4, T, C]

            # Track evolution
            norms = []
            changes = []
            entropies = []
            token_stability = []
            decoded_texts = []

            for iteration in range(n_iterations):
                # What tokens would this decode to?
                logits = model.lm_head(h)
                probs = F.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

                # Decode first sequence
                text = "".join([dataset.itos[t.item()] for t in predicted[0, :60]])
                decoded_texts.append(text)

                # Track metrics
                norm = h.norm(dim=-1).mean().item()
                norms.append(norm)
                entropies.append(entropy)

                if iteration > 0:
                    change = (h - h_prev).norm(dim=-1).mean().item()
                    changes.append(change)

                    # Token stability: how many positions kept the same prediction?
                    stability = (predicted == prev_predicted).float().mean().item()
                    token_stability.append(stability)

                h_prev = h.clone()
                prev_predicted = predicted.clone()

                # Dream step: feed hidden states back through the model
                h = model.forward_from_hidden(h)

            # Report
            print(f"\n    Iteration metrics:")
            print(f"    {'Iter':>6} {'Norm':>10} {'Change':>10} {'Entropy':>10} {'Stability':>10}")
            print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for i in range(n_iterations):
                change_str = f"{changes[i-1]:.4f}" if i > 0 else "---"
                stab_str = f"{100*token_stability[i-1]:.1f}%" if i > 0 else "---"
                print(f"    {i:>6} {norms[i]:>10.4f} {change_str:>10} {entropies[i]:>10.4f} {stab_str:>10}")

            # Did it converge?
            if len(changes) >= 3:
                final_change = np.mean(changes[-3:])
                initial_change = np.mean(changes[:3])
                converging = final_change < initial_change * 0.5
                final_stability = np.mean(token_stability[-3:])
            else:
                converging = False
                final_stability = 0

            print(f"\n    Convergence: {'YES' if converging else 'NO'} "
                  f"(initial change: {np.mean(changes[:3]):.4f}, "
                  f"final change: {np.mean(changes[-3:]):.4f})")
            print(f"    Final token stability: {100*final_stability:.1f}%")

            # Show the dream evolution
            print(f"\n    Dream sequence (first 60 chars of sequence 0):")
            for i in [0, 1, 2, 5, 10, 19]:
                if i < len(decoded_texts):
                    safe = decoded_texts[i].replace('\n', '\\n')
                    print(f"    iter {i:>2}: {safe}")

            # Does the dream converge to a fixed point or cycle?
            if len(decoded_texts) >= 10:
                last_texts = decoded_texts[-5:]
                unique_texts = len(set(last_texts))
                if unique_texts == 1:
                    print(f"\n    Dream state: FIXED POINT (converged to single output)")
                elif unique_texts <= 2:
                    print(f"\n    Dream state: LIMIT CYCLE (oscillating between {unique_texts} states)")
                else:
                    print(f"\n    Dream state: WANDERING ({unique_texts} unique in last 5 iterations)")


# =============================================================================
# TEST 4: Harmonic Expression -- Per-Band Energy
# =============================================================================

def test_harmonic_expression(baseline, progressive, dataset, config, n_batches=5):
    """
    Look at per-band energy in the hidden states. Does the progressive
    model distribute energy differently across harmonic channels?
    """
    print("\n" + "=" * 60)
    print("  TEST 4: Harmonic Expression")
    print("  How does each model use its harmonic bands?")
    print("=" * 60)

    n_harmonics = config.n_embd // 2

    for model_name, model in [("Baseline", baseline), ("Progressive", progressive)]:
        model.eval()
        print(f"\n  --- {model_name} Model ---")

        # Collect hidden states at multiple layers
        layer_energies = {}

        with torch.no_grad():
            for _ in range(n_batches):
                x, _ = dataset.get_batch("val", config)
                states = model.get_hidden_states(x)

                for stage_name, h in states.items():
                    if stage_name not in layer_energies:
                        layer_energies[stage_name] = []

                    # Per-band energy: band n = channels 2*(n-1) and 2*(n-1)+1
                    h_flat = h.view(-1, config.n_embd).cpu().numpy()
                    band_energy = np.zeros(n_harmonics)
                    for n in range(n_harmonics):
                        cos_ch = h_flat[:, 2*n]
                        sin_ch = h_flat[:, 2*n + 1]
                        band_energy[n] = np.mean(cos_ch**2 + sin_ch**2)
                    layer_energies[stage_name].append(band_energy)

        # Summarize
        print(f"\n    Band energy profile (mean across samples):")
        print(f"    {'Stage':<16} {'Low(1-8)':>10} {'Low(9-16)':>10} {'Mid(17-32)':>11} {'Mid(33-48)':>11} {'High(49-64)':>12} {'Total':>10}")
        print(f"    {'-'*16} {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*12} {'-'*10}")

        for stage_name in ["embedding", "layer0_mlp", "layer1_mlp", "layer2_mlp", "layer3_mlp", "final"]:
            if stage_name not in layer_energies:
                continue
            avg_energy = np.mean(layer_energies[stage_name], axis=0)
            low1 = avg_energy[:8].sum()
            low2 = avg_energy[8:16].sum()
            mid1 = avg_energy[16:32].sum()
            mid2 = avg_energy[32:48].sum()
            high = avg_energy[48:64].sum()
            total = avg_energy.sum()
            print(f"    {stage_name:<16} {low1:>10.4f} {low2:>10.4f} {mid1:>11.4f} {mid2:>11.4f} {high:>12.4f} {total:>10.4f}")

        # Energy concentration: how evenly spread?
        final_energy = np.mean(layer_energies["final"], axis=0)
        final_probs = final_energy / (final_energy.sum() + 1e-10)
        energy_entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))
        max_entropy = np.log(n_harmonics)
        uniformity = energy_entropy / max_entropy

        print(f"\n    Final layer energy uniformity: {uniformity:.3f} "
              f"(1.0 = perfectly uniform, 0.0 = all in one band)")

        # Peak bands
        top_5 = np.argsort(final_energy)[::-1][:5]
        bottom_5 = np.argsort(final_energy)[:5]
        print(f"    Highest energy bands: {', '.join(f'n={b+1}({final_energy[b]:.4f})' for b in top_5)}")
        print(f"    Lowest energy bands:  {', '.join(f'n={b+1}({final_energy[b]:.4f})' for b in bottom_5)}")


# =============================================================================
# TEST 5: Natural Groupings
# =============================================================================

def test_natural_groupings(baseline, progressive, dataset, config, n_batches=5):
    """
    Cluster hidden states without token labels. What groups does the
    model naturally form? Then reveal: what tokens are in each cluster?
    """
    print("\n" + "=" * 60)
    print("  TEST 5: Natural Groupings")
    print("  Cluster first, label after. What does the model see?")
    print("=" * 60)

    for model_name, model in [("Baseline", baseline), ("Progressive", progressive)]:
        model.eval()
        print(f"\n  --- {model_name} Model ---")

        all_hidden = []
        all_tokens = []
        all_contexts = []  # store surrounding characters for context

        with torch.no_grad():
            for _ in range(n_batches):
                x, _ = dataset.get_batch("val", config)
                states = model.get_hidden_states(x)
                final = states["final"]
                all_hidden.append(final.view(-1, config.n_embd).cpu().numpy())
                all_tokens.append(x.view(-1).cpu().numpy())

        H = np.concatenate(all_hidden, axis=0)
        tokens = np.concatenate(all_tokens, axis=0)

        # Simple K-means clustering (no sklearn dependency)
        K = 12  # try 12 clusters
        n_sample = min(20000, H.shape[0])
        idx = np.random.choice(H.shape[0], n_sample, replace=False)
        H_s = H[idx]
        tok_s = tokens[idx]

        # K-means initialization: random centroids
        np.random.seed(42)
        centroid_idx = np.random.choice(n_sample, K, replace=False)
        centroids = H_s[centroid_idx].copy()

        for km_iter in range(30):
            # Assign
            dists = np.linalg.norm(H_s[:, None, :] - centroids[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            # Update
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = H_s[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Now reveal: what's in each cluster?
        print(f"\n    K-means with K={K} on {n_sample:,} hidden states")
        print(f"    Converged in {km_iter+1} iterations")
        print(f"\n    {'Cluster':>8} {'Size':>8} {'Top characters (with counts)':>45}")
        print(f"    {'-'*8} {'-'*8} {'-'*45}")

        cluster_profiles = {}
        for k in range(K):
            mask = labels == k
            cluster_tokens = tok_s[mask]
            unique, counts = np.unique(cluster_tokens, return_counts=True)
            sorted_idx = np.argsort(counts)[::-1]

            total = mask.sum()
            # Top 6 characters with percentages
            top_chars = []
            for i in sorted_idx[:6]:
                char = dataset.itos[unique[i]]
                char_display = char if char not in '\n\t' else repr(char)
                pct = 100 * counts[i] / total
                top_chars.append(f"{char_display}({pct:.0f}%)")

            # What TYPE of cluster is this?
            top_char = dataset.itos[unique[sorted_idx[0]]]
            top_pct = 100 * counts[sorted_idx[0]] / total

            char_str = ", ".join(top_chars)
            print(f"    {k:>8} {total:>8} {char_str:>45}")

            cluster_profiles[k] = {
                "size": int(total),
                "top_char": top_char,
                "top_pct": top_pct,
                "unique_count": len(unique)
            }

        # Cluster purity: is each cluster dominated by one character type?
        purities = []
        for k, prof in cluster_profiles.items():
            purities.append(prof["top_pct"])
        avg_purity = np.mean(purities)
        max_purity = np.max(purities)
        min_purity = np.min(purities)

        print(f"\n    Cluster purity:")
        print(f"      Average: {avg_purity:.1f}% (top character dominance)")
        print(f"      Range: {min_purity:.1f}% -- {max_purity:.1f}%")
        print(f"      {'Pure' if avg_purity > 30 else 'Mixed'} clusters "
              f"({'character identity' if avg_purity > 30 else 'context-dependent'})")

        # Are clusters organized by character identity or by context?
        # Check: same character, same cluster?
        char_cluster_entropy = []
        for tok_id in range(dataset.vocab_size):
            mask = tok_s == tok_id
            if mask.sum() < 20:
                continue
            cluster_assignments = labels[mask]
            unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
            probs = cluster_counts / cluster_counts.sum()
            ent = -np.sum(probs * np.log(probs + 1e-10))
            max_ent = np.log(min(len(unique_clusters), K))
            char_cluster_entropy.append(ent / max_ent if max_ent > 0 else 0)

        avg_char_entropy = np.mean(char_cluster_entropy)
        print(f"\n    Character->cluster consistency:")
        print(f"      Average normalised entropy: {avg_char_entropy:.3f}")
        print(f"      (0.0 = each char always in same cluster = identity-based)")
        print(f"      (1.0 = each char spread across all clusters = context-based)")
        interpretation = ("IDENTITY-based" if avg_char_entropy < 0.3 else
                         "MIXED" if avg_char_entropy < 0.6 else
                         "CONTEXT-based")
        print(f"      Interpretation: {interpretation}")


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict(baseline_results, progressive_results):
    print("\n" + "=" * 60)
    print("  VERDICT: What Is the Model's Natural Expression?")
    print("=" * 60)

    print("""
  This test asked: if we stop imposing our categories (tokens, words,
  chords) and just look at what the model's internal space looks like --
  does the progressive ("happy") model express differently from baseline?

  The answer reveals whether progressive learning doesn't just make the
  model more accurate, but makes it think differently.""")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  NATURAL EXPRESSION TEST")
    print("  What is the model's own language?")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train both models
    baseline = train_baseline(config, dataset)
    progressive = train_progressive(config, dataset)

    # Check quality
    baseline_loss = eval_loss(baseline, dataset, config)
    progressive_loss = eval_loss(progressive, dataset, config)
    print(f"\n  Final validation loss:")
    print(f"    Baseline:    {baseline_loss:.4f}")
    print(f"    Progressive: {progressive_loss:.4f}")

    # Run tests
    test_internal_landscape(baseline, progressive, dataset, config)
    test_pre_projection(baseline, progressive, dataset, config)
    test_attractor_dynamics(baseline, progressive, dataset, config)
    test_harmonic_expression(baseline, progressive, dataset, config)
    test_natural_groupings(baseline, progressive, dataset, config)

    print_verdict(None, None)


if __name__ == "__main__":
    main()
