"""
Harmonic Knowledge Injection v2 — Fixed Methodology

WHAT v1 SHOWED:
- Generation test proved the model FOLLOWS geometry ("thenk", "whet", "thet")
- Isolated vs entangled bands confirmed Phase 2 predictions (2x collateral)
- But swap metrics were low because we only swapped INPUT embeddings

THE FIX:
The output projection (lm_head) is a separate linear layer. It maps final
hidden states to output characters. Swapping input embeddings changes how
context is PROCESSED, but not how output is DECODED. For a full identity
swap, we need to swap BOTH sides:
  1. Input embedding (how the model reads the character)
  2. Output projection row (how the model writes the character)

ALSO TESTING:
- Frozen model: embeddings never co-adapted with weights, so the model
  learned GENERIC harmonic processing. This should show cleaner swaps.
- Trainable model: embeddings co-adapted, so the model may rely on
  specific values. This tests the harder case.
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


def eval_loss(model, dataset, config, n_batches=20):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch("val", config)
        with torch.no_grad():
            _, loss = model(x, y)
        total += loss.item()
    return total / n_batches


# =============================================================================
# Core test: Full pipeline swap (embedding + lm_head)
# =============================================================================

def test_full_pipeline_swap(model, dataset, config, char_a, char_b, label=""):
    """
    Swap BOTH the input embedding AND the output projection row.
    This is the correct test: the model reads A as B and writes B as A.
    If the model processes geometry, predictions should flip cleanly.
    """
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]

    model.eval()

    # Collect test contexts
    n_test = 500
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
            logits = model.get_logits(ctx)
            pred = torch.argmax(logits, dim=-1).item()
            baseline_preds.append(pred)

    # Save originals
    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # --- Swap input embeddings ---
    model.wte.weight.data[idx_a] = orig_emb[idx_b].clone()
    model.wte.weight.data[idx_b] = orig_emb[idx_a].clone()

    # --- Swap output projection rows ---
    model.lm_head.weight.data[idx_a] = orig_head[idx_b].clone()
    model.lm_head.weight.data[idx_b] = orig_head[idx_a].clone()

    # Swapped predictions
    swapped_preds = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            pred = torch.argmax(logits, dim=-1).item()
            swapped_preds.append(pred)

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    # Analysis
    a_to_b = 0
    b_to_a = 0
    total_a = 0
    total_b = 0
    unchanged = 0
    other_changed = 0
    total_other = 0

    for bp, sp in zip(baseline_preds, swapped_preds):
        if bp == idx_a:
            total_a += 1
            if sp == idx_b:
                a_to_b += 1
        elif bp == idx_b:
            total_b += 1
            if sp == idx_a:
                b_to_a += 1
        else:
            total_other += 1
            if sp == bp:
                unchanged += 1
            else:
                other_changed += 1

    swap_a = a_to_b / total_a * 100 if total_a > 0 else 0
    swap_b = b_to_a / total_b * 100 if total_b > 0 else 0
    preserve = unchanged / total_other * 100 if total_other > 0 else 0

    print(f"    '{char_a}' -> '{char_b}': {a_to_b}/{total_a} ({swap_a:.1f}%)")
    print(f"    '{char_b}' -> '{char_a}': {b_to_a}/{total_b} ({swap_b:.1f}%)")
    print(f"    Others unchanged: {unchanged}/{total_other} ({preserve:.1f}%)")

    return swap_a, swap_b, preserve


# =============================================================================
# Test: Embedding-only vs Full-pipeline vs lm_head-only
# =============================================================================

def test_swap_modes(model, dataset, config, char_a, char_b):
    """
    Compare three swap strategies to isolate where identity lives:
    1. Embedding only (input side)
    2. lm_head only (output side)
    3. Both (full pipeline)
    """
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]
    model.eval()

    # Collect contexts
    n_test = 500
    test_contexts = []
    step = max(1, (len(dataset.val_data) - config.block_size) // n_test)
    for i in range(0, len(dataset.val_data) - config.block_size - 1, step):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Baseline
    baseline_preds = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            baseline_preds.append(torch.argmax(logits, dim=-1).item())

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    results = {}

    for mode_name, swap_emb, swap_head in [
        ("emb_only", True, False),
        ("head_only", False, True),
        ("full_pipe", True, True),
    ]:
        # Apply swaps
        model.wte.weight.data = orig_emb.clone()
        model.lm_head.weight.data = orig_head.clone()

        if swap_emb:
            model.wte.weight.data[idx_a] = orig_emb[idx_b].clone()
            model.wte.weight.data[idx_b] = orig_emb[idx_a].clone()
        if swap_head:
            model.lm_head.weight.data[idx_a] = orig_head[idx_b].clone()
            model.lm_head.weight.data[idx_b] = orig_head[idx_a].clone()

        # Predict
        swapped_preds = []
        with torch.no_grad():
            for ctx in test_contexts:
                logits = model.get_logits(ctx)
                swapped_preds.append(torch.argmax(logits, dim=-1).item())

        # Measure
        a_to_b = sum(1 for bp, sp in zip(baseline_preds, swapped_preds) if bp == idx_a and sp == idx_b)
        b_to_a = sum(1 for bp, sp in zip(baseline_preds, swapped_preds) if bp == idx_b and sp == idx_a)
        total_a = sum(1 for bp in baseline_preds if bp == idx_a)
        total_b = sum(1 for bp in baseline_preds if bp == idx_b)
        total_other = sum(1 for bp in baseline_preds if bp != idx_a and bp != idx_b)
        unchanged = sum(1 for bp, sp in zip(baseline_preds, swapped_preds)
                        if bp != idx_a and bp != idx_b and bp == sp)

        swap_a = a_to_b / total_a * 100 if total_a > 0 else 0
        swap_b = b_to_a / total_b * 100 if total_b > 0 else 0
        preserve = unchanged / total_other * 100 if total_other > 0 else 0

        results[mode_name] = (swap_a, swap_b, preserve)

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    return results


# =============================================================================
# Test: Generation comparison
# =============================================================================

def test_generation(model, dataset, config, char_a, char_b, mode="full_pipe"):
    """Generate text before and after swap to see the effect qualitatively."""
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]
    model.eval()

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # Deterministic seed for comparison
    torch.manual_seed(42)
    prompt = dataset.encode("\nKING HENRY:\nI shall ").unsqueeze(0).to(config.device)

    # Before
    torch.manual_seed(42)
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(150):
            logits = model.get_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
    text_before = dataset.decode(tokens[0].tolist())

    # Swap
    if mode in ("full_pipe", "emb_only"):
        model.wte.weight.data[idx_a] = orig_emb[idx_b].clone()
        model.wte.weight.data[idx_b] = orig_emb[idx_a].clone()
    if mode in ("full_pipe", "head_only"):
        model.lm_head.weight.data[idx_a] = orig_head[idx_b].clone()
        model.lm_head.weight.data[idx_b] = orig_head[idx_a].clone()

    # After
    torch.manual_seed(42)
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(150):
            logits = model.get_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
    text_after = dataset.decode(tokens[0].tolist())

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    return text_before, text_after


# =============================================================================
# Test: Progressive band injection with lm_head
# =============================================================================

def test_progressive_injection(model, dataset, config, char_source, char_target):
    """
    Progressive injection: replace source's geometry with target's,
    band by band, in BOTH embedding AND lm_head.
    """
    idx_s = dataset.stoi[char_source]
    idx_t = dataset.stoi[char_target]
    n_harmonics = config.n_embd // 2
    model.eval()

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # Band importance by embedding difference magnitude
    emb_diff = torch.zeros(n_harmonics)
    for h in range(n_harmonics):
        ci, si = h * 2, h * 2 + 1
        emb_diff[h] = ((orig_emb[idx_s, ci] - orig_emb[idx_t, ci])**2 +
                        (orig_emb[idx_s, si] - orig_emb[idx_t, si])**2).sqrt()
    band_order = torch.argsort(emb_diff, descending=True).numpy()

    # Collect test contexts
    n_test = 500
    test_contexts = []
    step = max(1, (len(dataset.val_data) - config.block_size) // n_test)
    for i in range(0, len(dataset.val_data) - config.block_size - 1, step):
        ctx = dataset.val_data[i:i+config.block_size].unsqueeze(0).to(config.device)
        test_contexts.append(ctx)
        if len(test_contexts) >= n_test:
            break

    # Baseline
    baseline_preds = []
    with torch.no_grad():
        for ctx in test_contexts:
            logits = model.get_logits(ctx)
            baseline_preds.append(torch.argmax(logits, dim=-1).item())

    print(f"\n    {'Bands':>6} | {'S->T rate':>10} | {'Others OK':>10} | {'Val loss':>9}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")

    for n_bands in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
        if n_bands > n_harmonics:
            continue

        model.wte.weight.data = orig_emb.clone()
        model.lm_head.weight.data = orig_head.clone()

        bands = band_order[:n_bands]
        for h in bands:
            ci, si = h * 2, h * 2 + 1
            # Input: source gets target's embedding values
            model.wte.weight.data[idx_s, ci] = orig_emb[idx_t, ci]
            model.wte.weight.data[idx_s, si] = orig_emb[idx_t, si]
            # Output: source gets target's lm_head projection values
            model.lm_head.weight.data[idx_s, ci] = orig_head[idx_t, ci]
            model.lm_head.weight.data[idx_s, si] = orig_head[idx_t, si]

        # Predict
        injected_preds = []
        with torch.no_grad():
            for ctx in test_contexts:
                logits = model.get_logits(ctx)
                injected_preds.append(torch.argmax(logits, dim=-1).item())

        s_to_t = sum(1 for bp, ip in zip(baseline_preds, injected_preds)
                     if bp == idx_s and ip == idx_t)
        total_s = sum(1 for bp in baseline_preds if bp == idx_s)
        total_other = sum(1 for bp in baseline_preds if bp != idx_s and bp != idx_t)
        unchanged = sum(1 for bp, ip in zip(baseline_preds, injected_preds)
                        if bp != idx_s and bp != idx_t and bp == ip)

        match_rate = s_to_t / total_s * 100 if total_s > 0 else 0
        preserve = unchanged / total_other * 100 if total_other > 0 else 0

        val_loss = eval_loss(model, dataset, config, n_batches=5)

        print(f"    {n_bands:>6} | {match_rate:>8.1f}% | {preserve:>8.1f}% | {val_loss:>9.4f}")

    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  HARMONIC INJECTION v2 — Fixed Methodology")
    print(f"  Swap both embedding AND output projection")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # =====================================================================
    # Train BOTH models
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TRAINING")
    print(f"{'='*60}")

    harmonic_model = train_model(config, dataset, mode="harmonic")
    frozen_model = train_model(config, dataset, mode="frozen")

    h_loss = eval_loss(harmonic_model, dataset, config)
    f_loss = eval_loss(frozen_model, dataset, config)
    print(f"\n  Harmonic baseline val loss: {h_loss:.4f}")
    print(f"  Frozen baseline val loss:   {f_loss:.4f}")

    # =====================================================================
    # TEST 1: Where does identity live? (embedding vs lm_head vs both)
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TEST 1: Where does character identity live?")
    print(f"  Comparing: embedding-only / lm_head-only / full pipeline swap")
    print(f"{'='*60}")

    char_pairs = [('e', 'a'), ('t', 's'), ('o', 'i'), ('h', 'n')]

    for model_obj, model_name in [(harmonic_model, "HARMONIC"), (frozen_model, "FROZEN")]:
        print(f"\n  --- {model_name} model ---")
        print(f"  {'Pair':>6} | {'Mode':>10} | {'A->B':>8} | {'B->A':>8} | {'Others':>8}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        for ca, cb in char_pairs:
            results = test_swap_modes(model_obj, dataset, config, ca, cb)
            for mode_name, (sa, sb, pres) in results.items():
                print(f"  {ca}<->{cb} | {mode_name:>10} | {sa:>6.1f}% | {sb:>6.1f}% | {pres:>6.1f}%")

    # =====================================================================
    # TEST 2: Full pipeline swap — the clean test
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TEST 2: Full pipeline swap — clean identity transfer")
    print(f"{'='*60}")

    for model_obj, model_name in [(harmonic_model, "HARMONIC"), (frozen_model, "FROZEN")]:
        print(f"\n  --- {model_name} model ---")
        total_swap = 0
        total_preserve = 0
        n_pairs = 0

        for ca, cb in char_pairs:
            print(f"\n  Swap '{ca}' <-> '{cb}':")
            sa, sb, pres = test_full_pipeline_swap(model_obj, dataset, config, ca, cb)
            total_swap += (sa + sb) / 2
            total_preserve += pres
            n_pairs += 1

        avg_swap = total_swap / n_pairs
        avg_preserve = total_preserve / n_pairs
        print(f"\n  Average swap rate: {avg_swap:.1f}%")
        print(f"  Average preservation: {avg_preserve:.1f}%")

    # =====================================================================
    # TEST 3: Generation quality — the taste test
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TEST 3: Generation after full pipeline swap")
    print(f"{'='*60}")

    for model_obj, model_name in [(harmonic_model, "HARMONIC"), (frozen_model, "FROZEN")]:
        print(f"\n  --- {model_name} model: 'e' <-> 'a' ---")

        text_before, text_after = test_generation(model_obj, dataset, config, 'e', 'a', mode="full_pipe")
        print(f"\n    BEFORE:")
        print(f"    {repr(text_before[:200])}")
        print(f"\n    AFTER (e<->a full pipeline swap):")
        print(f"    {repr(text_after[:200])}")

        # Count systematic replacements
        e_before = text_before.count('e')
        a_before = text_before.count('a')
        e_after = text_after.count('e')
        a_after = text_after.count('a')
        print(f"\n    Frequencies: 'e' {e_before}->{e_after}  'a' {a_before}->{a_after}")

    # =====================================================================
    # TEST 4: Progressive band injection (with lm_head)
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TEST 4: Progressive band injection (embedding + lm_head)")
    print(f"{'='*60}")

    for model_obj, model_name in [(harmonic_model, "HARMONIC"), (frozen_model, "FROZEN")]:
        print(f"\n  --- {model_name} model: 'e' -> 'a' ---")
        test_progressive_injection(model_obj, dataset, config, 'e', 'a')

    # =====================================================================
    # TEST 5: Harmonic vs Frozen — which follows geometry better?
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TEST 5: Head-to-head — Harmonic vs Frozen geometry following")
    print(f"{'='*60}")

    all_pairs = [('e', 'a'), ('t', 's'), ('o', 'i'), ('h', 'n'),
                 ('r', 'l'), ('d', 'c'), ('u', 'y')]

    for model_obj, model_name in [(harmonic_model, "HARMONIC"), (frozen_model, "FROZEN")]:
        print(f"\n  --- {model_name} ---")
        swap_rates = []
        preserve_rates = []

        for ca, cb in all_pairs:
            sa, sb, pres = test_full_pipeline_swap(model_obj, dataset, config, ca, cb)
            avg_swap = (sa + sb) / 2
            swap_rates.append(avg_swap)
            preserve_rates.append(pres)

        print(f"  Across {len(all_pairs)} character pairs:")
        print(f"    Avg swap rate:     {np.mean(swap_rates):.1f}% +/- {np.std(swap_rates):.1f}%")
        print(f"    Avg preservation:  {np.mean(preserve_rates):.1f}% +/- {np.std(preserve_rates):.1f}%")
        print(f"    Best swap:         {max(swap_rates):.1f}%")
        print(f"    Worst collateral:  {min(preserve_rates):.1f}%")

    # =====================================================================
    # VERDICT
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")

    # Re-run the definitive comparison
    h_swaps = []
    f_swaps = []
    h_preserves = []
    f_preserves = []

    for ca, cb in all_pairs:
        sa, sb, pres = test_full_pipeline_swap(harmonic_model, dataset, config, ca, cb)
        h_swaps.append((sa + sb) / 2)
        h_preserves.append(pres)

        sa, sb, pres = test_full_pipeline_swap(frozen_model, dataset, config, ca, cb)
        f_swaps.append((sa + sb) / 2)
        f_preserves.append(pres)

    h_avg_swap = np.mean(h_swaps)
    f_avg_swap = np.mean(f_swaps)
    h_avg_pres = np.mean(h_preserves)
    f_avg_pres = np.mean(f_preserves)

    print(f"\n  Full pipeline swap (embedding + lm_head) across {len(all_pairs)} pairs:")
    print(f"")
    print(f"  {'Model':>12} | {'Avg swap rate':>14} | {'Avg preservation':>18}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*18}")
    print(f"  {'Harmonic':>12} | {h_avg_swap:>12.1f}% | {h_avg_pres:>16.1f}%")
    print(f"  {'Frozen':>12} | {f_avg_swap:>12.1f}% | {f_avg_pres:>16.1f}%")

    winner = "FROZEN" if f_avg_swap > h_avg_swap else "HARMONIC"
    cleaner = "FROZEN" if f_avg_pres > h_avg_pres else "HARMONIC"

    print(f"\n  Higher swap rate: {winner}")
    print(f"  Cleaner preservation: {cleaner}")

    overall = max(h_avg_swap, f_avg_swap)
    if overall > 70:
        print(f"\n  CONFIRMED: Identity follows geometry ({overall:.1f}% swap rate).")
        print(f"  Knowledge can be injected via harmonic vectors without retraining.")
    elif overall > 40:
        print(f"\n  PARTIAL: Identity partially follows geometry ({overall:.1f}% swap rate).")
        print(f"  Geometry carries signal but MLP weights add context-dependent processing.")
    elif overall > 15:
        print(f"\n  WEAK: Geometry has measurable but limited effect ({overall:.1f}% swap rate).")
        print(f"  Most identity lives in the MLP weights, not input geometry alone.")
    else:
        print(f"\n  MINIMAL: Geometry swap has little effect ({overall:.1f}% swap rate).")
        print(f"  Identity is distributed across all weights, not concentrated in embeddings.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
