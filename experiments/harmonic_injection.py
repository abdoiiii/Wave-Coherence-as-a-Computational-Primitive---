"""
Harmonic Knowledge Injection — Nutrition, Not Surgery

THE INSIGHT (from Phase 2):
The transformer took harmonic structure from 35.8% independent to 92.5% independent.
It didn't need to be told. It WANTED to organize harmonic geometry.
The frozen result proved: geometry carries signal without learning.
n=14 has 3.6% overlap — practically a clean register.

THE PARADIGM SHIFT:
Don't edit weights (ROME surgery). Change the input geometry and let the
model's existing processing pipeline handle it. The model learned to PROCESS
harmonic structure, not to MEMORIZE specific values. So changing the geometry
changes what it computes — like updating a config file vs patching binaries.

THE TEST:
1. Train harmonic model on Shakespeare (learns character patterns)
2. After training, swap two characters' harmonic embeddings
3. Feed sequences through the UNCHANGED network
4. Measure: does the model now treat character A as if it were character B?
5. Measure: are all other characters unaffected? (92.5% independence predicts yes)
6. Test at specific band counts: how many bands must you swap?
7. Test the most isolated bands (n=14, n=5) vs entangled bands (n=9, n=11)
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
    eval_iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model (same architecture as all experiments)
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

    @torch.no_grad()
    def predict_next(self, idx):
        logits, _ = self(idx)
        logits = logits[:, -1, :]
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def get_logits(self, idx):
        logits, _ = self(idx)
        return logits[:, -1, :]


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

    def encode(self, text):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens):
        return "".join([self.itos[t.item() if hasattr(t, 'item') else t] for t in tokens])


# =============================================================================
# Training
# =============================================================================

def train_model(config, dataset):
    print(f"\n{'='*60}")
    print(f"  Training harmonic model on Shakespeare")
    print(f"{'='*60}")
    model = HarmonicGPT(config, dataset.vocab_size, mode="harmonic").to(config.device)
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
# Test 1: Full Character Identity Swap
# =============================================================================

def test_character_swap(model, dataset, config, char_a, char_b):
    """
    Swap the ENTIRE harmonic embedding of two characters.
    If the model processes geometry (not memorized values), then:
    - Wherever it would predict char_a, it should now predict char_b
    - Wherever it would predict char_b, it should now predict char_a
    - All other predictions should be unchanged
    """
    print(f"\n  Full embedding swap: '{char_a}' <-> '{char_b}'")

    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]

    # Collect test contexts from validation data
    model.eval()
    n_test = 200
    test_contexts = []
    for i in range(0, len(dataset.val_data) - config.block_size - 1, len(dataset.val_data) // n_test):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Get baseline predictions for all contexts
    baseline_preds = []
    baseline_probs = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            baseline_preds.append(pred)
            baseline_probs.append(probs.cpu())

    # Swap embeddings
    original_weight = model.wte.weight.data.clone()
    model.wte.weight.data[idx_a] = original_weight[idx_b].clone()
    model.wte.weight.data[idx_b] = original_weight[idx_a].clone()

    # Get swapped predictions
    swapped_preds = []
    swapped_probs = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            swapped_preds.append(pred)
            swapped_probs.append(probs.cpu())

    # Restore
    model.wte.weight.data = original_weight

    # Analysis
    a_to_b = 0  # Predictions that flipped from A to B
    b_to_a = 0  # Predictions that flipped from B to A
    unchanged = 0  # Non-A, non-B predictions that stayed the same
    other_changed = 0  # Non-A, non-B predictions that changed (collateral)
    total_a_pred = 0
    total_b_pred = 0

    for bp, sp in zip(baseline_preds, swapped_preds):
        if bp == idx_a:
            total_a_pred += 1
            if sp == idx_b:
                a_to_b += 1
        elif bp == idx_b:
            total_b_pred += 1
            if sp == idx_a:
                b_to_a += 1
        else:
            if sp == bp:
                unchanged += 1
            else:
                other_changed += 1

    total_other = len(baseline_preds) - total_a_pred - total_b_pred

    # Probability-level analysis: average KL divergence for non-swapped tokens
    kl_others = []
    for bp, sp, b_prob, s_prob in zip(baseline_preds, swapped_preds, baseline_probs, swapped_probs):
        if bp != idx_a and bp != idx_b:
            kl = F.kl_div(s_prob.log(), b_prob, reduction='sum').item()
            kl_others.append(kl)

    avg_kl_others = np.mean(kl_others) if kl_others else 0.0

    swap_rate_a = a_to_b / total_a_pred * 100 if total_a_pred > 0 else 0
    swap_rate_b = b_to_a / total_b_pred * 100 if total_b_pred > 0 else 0
    preserve_rate = unchanged / total_other * 100 if total_other > 0 else 0

    print(f"    '{char_a}' predictions ({total_a_pred} total): {a_to_b} flipped to '{char_b}' ({swap_rate_a:.1f}%)")
    print(f"    '{char_b}' predictions ({total_b_pred} total): {b_to_a} flipped to '{char_a}' ({swap_rate_b:.1f}%)")
    print(f"    Other predictions ({total_other} total): {unchanged} unchanged ({preserve_rate:.1f}%), {other_changed} changed")
    print(f"    Collateral damage (avg KL on others): {avg_kl_others:.6f}")

    return {
        "swap_rate_a": swap_rate_a,
        "swap_rate_b": swap_rate_b,
        "preserve_rate": preserve_rate,
        "collateral_kl": avg_kl_others,
        "total_a": total_a_pred,
        "total_b": total_b_pred,
    }


# =============================================================================
# Test 2: Band-by-Band Injection
# =============================================================================

def test_band_injection(model, dataset, config, char_source, char_target):
    """
    Progressively replace char_source's embedding with char_target's,
    one harmonic band at a time. Find the minimum bands needed for
    the model to treat source as target.

    This is the 'injection' test: we inject new identity into a character
    by replacing its harmonic geometry band by band.
    """
    print(f"\n  Band-by-band injection: '{char_source}' -> '{char_target}'")
    print(f"  (Replacing '{char_source}' embedding with '{char_target}' embedding, band by band)")

    idx_source = dataset.stoi[char_source]
    idx_target = dataset.stoi[char_target]
    n_harmonics = config.n_embd // 2

    model.eval()

    # Collect contexts where the model predicts char_target
    # (these are the contexts where, after injection, source should also trigger target-like behavior)
    n_test = 300
    test_contexts = []
    for i in range(0, len(dataset.val_data) - config.block_size - 1, len(dataset.val_data) // n_test):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Get baseline: what does the model predict for each context?
    with torch.no_grad():
        baseline_preds = []
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            pred = torch.argmax(logits, dim=-1).item()
            baseline_preds.append(pred)

    # Compute per-band importance: which bands carry the most identity difference?
    original_weight = model.wte.weight.data.clone()
    source_emb = original_weight[idx_source].clone()
    target_emb = original_weight[idx_target].clone()

    # Band importance = magnitude of difference between source and target at each band
    band_diff = torch.zeros(n_harmonics)
    for h in range(n_harmonics):
        cos_idx = h * 2
        sin_idx = h * 2 + 1
        diff = ((source_emb[cos_idx] - target_emb[cos_idx])**2 +
                (source_emb[sin_idx] - target_emb[sin_idx])**2).sqrt()
        band_diff[h] = diff

    # Sort bands by difference magnitude (biggest difference first = most identity)
    band_order = torch.argsort(band_diff, descending=True).numpy()

    print(f"\n    Top 5 bands by identity difference ('{char_source}' vs '{char_target}'):")
    for i in range(5):
        h = band_order[i]
        print(f"      n={h+1:>3}  diff={band_diff[h]:.4f}")

    # Progressive injection
    print(f"\n    {'Bands':>6} | {'Source pred':>12} | {'Match target':>13} | {'Others OK':>10} | {'Val loss':>9}")
    print(f"    {'-'*6}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*9}")

    results = []
    for n_bands in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
        if n_bands > n_harmonics:
            continue

        # Restore and apply injection
        model.wte.weight.data = original_weight.clone()
        bands = band_order[:n_bands]
        for h in bands:
            cos_idx = h * 2
            sin_idx = h * 2 + 1
            model.wte.weight.data[idx_source, cos_idx] = target_emb[cos_idx]
            model.wte.weight.data[idx_source, sin_idx] = target_emb[sin_idx]

        # Measure predictions after injection
        with torch.no_grad():
            injected_preds = []
            for ctx in test_contexts:
                logits = model.get_logits(ctx)
                pred = torch.argmax(logits, dim=-1).item()
                injected_preds.append(pred)

        # Count: how many source-predictions became target-predictions?
        source_became_target = 0
        total_source = 0
        others_unchanged = 0
        total_others = 0

        for bp, ip in zip(baseline_preds, injected_preds):
            if bp == idx_source:
                total_source += 1
                if ip == idx_target:
                    source_became_target += 1
            elif bp != idx_target:
                total_others += 1
                if ip == bp:
                    others_unchanged += 1

        match_rate = source_became_target / total_source * 100 if total_source > 0 else 0
        preserve_rate = others_unchanged / total_others * 100 if total_others > 0 else 0

        # Quick val loss check
        val_loss = 0.0
        for _ in range(5):
            x, y = dataset.get_batch("val", config)
            with torch.no_grad():
                _, loss = model(x, y)
            val_loss += loss.item()
        val_loss /= 5

        print(f"    {n_bands:>6} | {match_rate:>10.1f}% | {source_became_target:>5}/{total_source:<5}   | {preserve_rate:>8.1f}% | {val_loss:>9.4f}")

        results.append({
            "n_bands": n_bands,
            "match_rate": match_rate,
            "preserve_rate": preserve_rate,
            "val_loss": val_loss,
        })

    # Restore
    model.wte.weight.data = original_weight

    return results


# =============================================================================
# Test 3: Isolated vs Entangled Bands
# =============================================================================

def test_isolated_vs_entangled(model, dataset, config, char_source, char_target):
    """
    Compare injection using the MOST isolated bands (from Phase 2)
    vs the MOST entangled bands. The prediction:

    - Isolated bands (n=14, n=5, n=1): clean swap, minimal collateral
    - Entangled bands (n=9, n=11): messy swap, more collateral

    This directly tests whether Phase 2's independence map predicts
    editing safety.
    """
    print(f"\n  Isolated vs entangled bands: '{char_source}' -> '{char_target}'")

    idx_source = dataset.stoi[char_source]
    idx_target = dataset.stoi[char_target]

    # From Phase 2 results (0-indexed):
    isolated_bands = [13, 4, 0, 5, 12, 2]   # n=14,5,1,6,13,3 — least overlap
    entangled_bands = [8, 10, 15, 7, 9, 3]   # n=9,11,16,8,10,4 — most overlap

    model.eval()
    original_weight = model.wte.weight.data.clone()
    target_emb = original_weight[idx_target].clone()

    # Collect test contexts
    n_test = 300
    test_contexts = []
    for i in range(0, len(dataset.val_data) - config.block_size - 1, len(dataset.val_data) // n_test):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Baseline
    with torch.no_grad():
        baseline_preds = []
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            pred = torch.argmax(logits, dim=-1).item()
            baseline_preds.append(pred)

    print(f"\n    {'Band type':>12} | {'Swap rate':>10} | {'Others OK':>10} | {'KL (others)':>12} | {'Val loss':>9}")
    print(f"    {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*9}")

    for band_label, band_list in [("isolated", isolated_bands), ("entangled", entangled_bands)]:
        for n_bands in [2, 4, 6]:
            bands = band_list[:n_bands]
            model.wte.weight.data = original_weight.clone()

            for h in bands:
                cos_idx = h * 2
                sin_idx = h * 2 + 1
                model.wte.weight.data[idx_source, cos_idx] = target_emb[cos_idx]
                model.wte.weight.data[idx_source, sin_idx] = target_emb[sin_idx]

            with torch.no_grad():
                injected_preds = []
                injected_probs = []
                for ctx in test_contexts:
                    logits = model.get_logits(ctx)
                    probs = F.softmax(logits, dim=-1)
                    pred = torch.argmax(logits, dim=-1).item()
                    injected_preds.append(pred)
                    injected_probs.append(probs.cpu())

            # Get baseline probs for KL
            baseline_probs_list = []
            model.wte.weight.data = original_weight.clone()
            with torch.no_grad():
                for ctx in test_contexts:
                    logits = model.get_logits(ctx)
                    probs = F.softmax(logits, dim=-1)
                    baseline_probs_list.append(probs.cpu())

            # Restore for next iteration
            model.wte.weight.data = original_weight.clone()

            # Measure
            source_to_target = 0
            total_source = 0
            others_unchanged = 0
            total_others = 0
            kl_others = []

            for i, (bp, ip) in enumerate(zip(baseline_preds, injected_preds)):
                if bp == idx_source:
                    total_source += 1
                    if ip == idx_target:
                        source_to_target += 1
                elif bp != idx_target:
                    total_others += 1
                    if ip == bp:
                        others_unchanged += 1
                    kl = F.kl_div(
                        injected_probs[i].log(), baseline_probs_list[i],
                        reduction='sum'
                    ).item()
                    kl_others.append(kl)

            swap_rate = source_to_target / total_source * 100 if total_source > 0 else 0
            preserve_rate = others_unchanged / total_others * 100 if total_others > 0 else 0
            avg_kl = np.mean(kl_others) if kl_others else 0.0

            # Val loss
            model.wte.weight.data = original_weight.clone()
            for h in bands:
                cos_idx = h * 2
                sin_idx = h * 2 + 1
                model.wte.weight.data[idx_source, cos_idx] = target_emb[cos_idx]
                model.wte.weight.data[idx_source, sin_idx] = target_emb[sin_idx]
            val_loss = 0.0
            for _ in range(5):
                x, y = dataset.get_batch("val", config)
                with torch.no_grad():
                    _, loss = model(x, y)
                val_loss += loss.item()
            val_loss /= 5

            label = f"{band_label[:4]}({n_bands})"
            print(f"    {label:>12} | {swap_rate:>8.1f}% | {preserve_rate:>8.1f}% | {avg_kl:>12.6f} | {val_loss:>9.4f}")

    model.wte.weight.data = original_weight


# =============================================================================
# Test 4: Generation Quality After Injection
# =============================================================================

def test_generation_after_injection(model, dataset, config, char_source, char_target, n_bands_to_use):
    """
    The taste test: generate Shakespeare after injecting new identity.
    Does the text still read like Shakespeare, with only the swapped
    character changed?
    """
    print(f"\n  Generation after injecting '{char_source}' -> '{char_target}' ({n_bands_to_use} bands)")

    idx_source = dataset.stoi[char_source]
    idx_target = dataset.stoi[char_target]
    n_harmonics = config.n_embd // 2

    model.eval()
    original_weight = model.wte.weight.data.clone()
    source_emb = original_weight[idx_source].clone()
    target_emb = original_weight[idx_target].clone()

    # Use largest-difference bands
    band_diff = torch.zeros(n_harmonics)
    for h in range(n_harmonics):
        cos_idx = h * 2
        sin_idx = h * 2 + 1
        diff = ((source_emb[cos_idx] - target_emb[cos_idx])**2 +
                (source_emb[sin_idx] - target_emb[sin_idx])**2).sqrt()
        band_diff[h] = diff
    band_order = torch.argsort(band_diff, descending=True).numpy()

    # Generate BEFORE injection
    prompt = dataset.encode("\nKING HENRY:\nI shall ").unsqueeze(0).to(config.device)

    with torch.no_grad():
        generated_before = model.get_logits(prompt)
        # Full generation
        tokens = prompt.clone()
        for _ in range(150):
            logits = model.get_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
        text_before = dataset.decode(tokens[0].tolist())

    # Apply injection
    bands = band_order[:n_bands_to_use]
    for h in bands:
        cos_idx = h * 2
        sin_idx = h * 2 + 1
        model.wte.weight.data[idx_source, cos_idx] = target_emb[cos_idx]
        model.wte.weight.data[idx_source, sin_idx] = target_emb[sin_idx]

    # Generate AFTER injection
    with torch.no_grad():
        tokens = prompt.clone()
        for _ in range(150):
            logits = model.get_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
        text_after = dataset.decode(tokens[0].tolist())

    # Restore
    model.wte.weight.data = original_weight

    print(f"\n    BEFORE injection:")
    print(f"    {repr(text_before[:200])}")
    print(f"\n    AFTER injection ('{char_source}' -> '{char_target}'):")
    print(f"    {repr(text_after[:200])}")

    # Count character frequencies in generated text
    before_source = text_before.count(char_source)
    before_target = text_before.count(char_target)
    after_source = text_after.count(char_source)
    after_target = text_after.count(char_target)

    print(f"\n    Character frequencies in generated text:")
    print(f"      '{char_source}': {before_source} (before) -> {after_source} (after)")
    print(f"      '{char_target}': {before_target} (before) -> {after_target} (after)")


# =============================================================================
# Test 5: The Intune Test — Swap at Inference, No Retraining
# =============================================================================

def test_inference_time_swap(model, dataset, config):
    """
    The definitive test. Take specific prompts where the model confidently
    predicts a known character. Swap that character's embedding with another.
    Does the prediction follow the geometry?

    This is the Intune analogy: push a policy change, each machine applies
    it according to its own state.
    """
    print(f"\n  Inference-time swap test")
    print(f"  (Does prediction follow geometry, not memorized weights?)")

    model.eval()
    original_weight = model.wte.weight.data.clone()

    # Find high-confidence predictions
    # Get many contexts and their predictions
    test_size = 500
    contexts = []
    preds = []
    confs = []

    for i in range(0, len(dataset.val_data) - config.block_size - 1,
                    max(1, (len(dataset.val_data) - config.block_size) // test_size)):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        with torch.no_grad():
            logits = model.get_logits(ctx)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()

        contexts.append(ctx)
        preds.append(pred)
        confs.append(conf)

        if len(contexts) >= test_size:
            break

    # Find the most-predicted character (has enough test cases)
    from collections import Counter
    pred_counts = Counter(preds)
    top_chars = pred_counts.most_common(5)

    print(f"\n    Most frequently predicted characters:")
    for char_idx, count in top_chars:
        char = dataset.itos[char_idx]
        avg_conf = np.mean([c for p, c in zip(preds, confs) if p == char_idx])
        print(f"      '{char}' (idx {char_idx}): {count} predictions, avg confidence {avg_conf:.3f}")

    # Pick the top-predicted character and a swap partner
    primary_idx = top_chars[0][0]
    primary_char = dataset.itos[primary_idx]
    # Pick a less common character as swap target
    swap_target_idx = top_chars[-1][0]
    swap_target_char = dataset.itos[swap_target_idx]

    print(f"\n    Swapping: '{primary_char}' (idx {primary_idx}) <-> '{swap_target_char}' (idx {swap_target_idx})")

    # Get contexts where model predicts the primary character with high confidence
    primary_contexts = [(ctx, conf) for ctx, pred, conf in zip(contexts, preds, confs)
                        if pred == primary_idx and conf > 0.3]

    if len(primary_contexts) < 5:
        print(f"    Not enough high-confidence predictions for '{primary_char}'. Lowering threshold.")
        primary_contexts = [(ctx, conf) for ctx, pred, conf in zip(contexts, preds, confs)
                            if pred == primary_idx]

    print(f"    Testing {len(primary_contexts)} contexts where model predicts '{primary_char}'")

    # SWAP embedding
    model.wte.weight.data[primary_idx] = original_weight[swap_target_idx].clone()
    model.wte.weight.data[swap_target_idx] = original_weight[primary_idx].clone()

    # Check: do those contexts now predict the swap target?
    followed_geometry = 0
    stayed_same = 0
    went_other = 0

    for ctx, conf in primary_contexts:
        with torch.no_grad():
            logits = model.get_logits(ctx)
            new_pred = torch.argmax(logits, dim=-1).item()

        if new_pred == swap_target_idx:
            followed_geometry += 1
        elif new_pred == primary_idx:
            stayed_same += 1
        else:
            went_other += 1

    total = len(primary_contexts)
    print(f"\n    Results after swap:")
    print(f"      Followed geometry (now predicts '{swap_target_char}'): {followed_geometry}/{total} ({followed_geometry/total*100:.1f}%)")
    print(f"      Stayed on old prediction ('{primary_char}'):           {stayed_same}/{total} ({stayed_same/total*100:.1f}%)")
    print(f"      Went to other character:                       {went_other}/{total} ({went_other/total*100:.1f}%)")

    if followed_geometry / total > 0.5:
        print(f"\n    >> GEOMETRY WINS: The model follows the embedding, not memorized weights.")
    elif stayed_same / total > 0.5:
        print(f"\n    >> WEIGHTS WIN: The model ignores the embedding change.")
    else:
        print(f"\n    >> MIXED: Model partially follows geometry.")

    # Restore
    model.wte.weight.data = original_weight

    return followed_geometry / total if total > 0 else 0


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  HARMONIC KNOWLEDGE INJECTION")
    print(f"  Nutrition, not surgery")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    # Load Shakespeare
    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train
    model = train_model(config, dataset)

    # Baseline quality
    model.eval()
    baseline_loss = 0.0
    for _ in range(20):
        x, y = dataset.get_batch("val", config)
        with torch.no_grad():
            _, loss = model(x, y)
        baseline_loss += loss.item()
    baseline_loss /= 20
    print(f"\n  Baseline val loss: {baseline_loss:.4f}")

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TEST 1: Full Character Identity Swap")
    print(f"{'='*60}")
    # =========================================================================

    # Test with common characters
    print(f"\n  --- Common vowels ---")
    test_character_swap(model, dataset, config, 'e', 'a')
    test_character_swap(model, dataset, config, 'o', 'i')

    # Test with consonants
    print(f"\n  --- Common consonants ---")
    test_character_swap(model, dataset, config, 't', 's')

    # Test with rare vs common
    print(f"\n  --- Rare vs common ---")
    test_character_swap(model, dataset, config, 'z', 'e')

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TEST 2: Band-by-Band Injection")
    print(f"{'='*60}")
    # =========================================================================

    test_band_injection(model, dataset, config, 'e', 'a')

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TEST 3: Isolated vs Entangled Bands")
    print(f"{'='*60}")
    # =========================================================================

    test_isolated_vs_entangled(model, dataset, config, 'e', 'a')

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TEST 4: Generation Quality After Injection")
    print(f"{'='*60}")
    # =========================================================================

    test_generation_after_injection(model, dataset, config, 'e', 'a', n_bands_to_use=32)

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  TEST 5: The Intune Test")
    print(f"  Does prediction follow geometry or memorized weights?")
    print(f"{'='*60}")
    # =========================================================================

    geometry_rate = test_inference_time_swap(model, dataset, config)

    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  FINAL VERDICT")
    print(f"{'='*60}")
    # =========================================================================

    print(f"\n  The question: does the model process geometry or memorized values?")
    print(f"  Geometry following rate: {geometry_rate*100:.1f}%")

    if geometry_rate > 0.7:
        print(f"\n  CONFIRMED: The transformer processes harmonic geometry.")
        print(f"  Knowledge can be injected by changing input vectors —")
        print(f"  no weight editing, no retraining, no surgery.")
        print(f"  The model's MLP pipeline handles the rest.")
    elif geometry_rate > 0.4:
        print(f"\n  PARTIAL: The transformer partially follows geometry.")
        print(f"  Some knowledge is in the geometry, some in the weights.")
        print(f"  Hybrid approach may be needed.")
    else:
        print(f"\n  NEGATIVE: The transformer relies on memorized weights.")
        print(f"  Embedding swap alone does not redirect predictions.")
        print(f"  Knowledge lives in MLP weights, not input geometry.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
