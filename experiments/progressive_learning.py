"""
Progressive Harmonic Learning — Build the Drawers in Order

THE INSIGHT:
Humans learn language in stages:
1. Sound/rhythm (babies) — structural patterns, cadence
2. Vowels (a, e, i, o, u) — the backbone of words
3. Full alphabet — fine distinctions

Our spectral chord analysis showed this maps to harmonic bands:
- Low bands (1-17): structural info (space, newline, rhythm)
- Mid bands: vowel/word shape patterns
- High bands (44-64): character identity, fine distinctions

We've been training with all 64 bands active simultaneously.
What if we build the foundation first, then layer detail?

THE TEST:
1. BASELINE: train with all bands active from step 0
2. PROGRESSIVE: train in stages — low bands first, then mid, then all
3. COMPARE: learning speed, final quality, knowledge organization
4. KEY TEST: after training, introduce NEW data. Which model absorbs it
   faster with less catastrophic forgetting? The one with organized
   drawers should file new knowledge more efficiently.
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
    eval_interval = 200
    eval_iters = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Model with band masking capability
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
        # Band mask: 1.0 = active, 0.0 = masked
        self.register_buffer("band_mask", torch.ones(embedding_dim))

    def set_active_bands(self, max_band):
        """Activate bands 1..max_band, mask the rest."""
        n_harmonics = self.weight.shape[1] // 2
        mask = torch.zeros(self.weight.shape[1])
        for h in range(min(max_band, n_harmonics)):
            mask[h * 2] = 1.0      # cos
            mask[h * 2 + 1] = 1.0  # sin
        self.band_mask = mask.to(self.weight.device)

    def set_all_active(self):
        self.band_mask = torch.ones(self.weight.shape[1], device=self.weight.device)

    def forward(self, x):
        masked_weight = self.weight * self.band_mask.unsqueeze(0)
        return F.embedding(x, masked_weight)


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
# Evaluation
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


# =============================================================================
# Training strategies
# =============================================================================

def train_baseline(config, dataset):
    """Standard training: all bands active from step 0."""
    print(f"\n  BASELINE: All 64 bands active from start")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses_log = []
    start = time.time()

    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss(model, dataset, config)
            elapsed = time.time() - start
            losses_log.append((it, val_loss))
            print(f"  step {it:>5} | val {val_loss:.4f} | {elapsed:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Training complete in {time.time()-start:.1f}s")
    return model, losses_log


def train_progressive(config, dataset):
    """
    Progressive UNFREEZING: all bands visible, but only some trainable.

    All acoustic information present throughout (all bands active in forward
    pass). The difference is which embedding bands receive gradients:

    Stage 1 (steps 0-999):    bands 1-8  trainable (structure learns first)
    Stage 2 (steps 1000-1999): bands 1-24 trainable (add word-level patterns)
    Stage 3 (steps 2000-2999): all bands trainable (full detail)

    This is the human analogy: babies hear the FULL word but learn
    structure first, then detail. The signal is always complete —
    the understanding builds progressively.
    """
    print(f"\n  PROGRESSIVE UNFREEZING: All bands visible, gradients build up")

    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

    n_harmonics = config.n_embd // 2

    # All bands always active in forward pass
    model.wte.set_all_active()

    stages = [
        (0,    1000, 8,  "structure (bands 1-8 trainable)"),
        (1000, 2000, 24, "word patterns (bands 1-24 trainable)"),
        (2000, 3000, 64, "full detail (all bands trainable)"),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses_log = []
    start = time.time()

    for stage_start, stage_end, max_band, stage_name in stages:
        # Freeze/unfreeze embedding bands via gradient masking
        # We'll apply a gradient hook to mask frozen bands
        print(f"\n  --- Stage: {stage_name} ---")

        for it in range(stage_start, stage_end):
            if it % config.eval_interval == 0 or it == config.max_iters - 1:
                model.eval()
                val_loss = eval_loss(model, dataset, config)
                elapsed = time.time() - start
                losses_log.append((it, val_loss))
                print(f"  step {it:>5} | val {val_loss:.4f} | unfreeze 1-{max_band} | {elapsed:.1f}s")
                model.train()

            x, y = dataset.get_batch("train", config)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()

            # Zero out gradients for frozen bands in embedding
            if max_band < n_harmonics and model.wte.weight.grad is not None:
                for h in range(max_band, n_harmonics):
                    model.wte.weight.grad[:, h * 2] = 0.0
                    model.wte.weight.grad[:, h * 2 + 1] = 0.0

            optimizer.step()

    # Final eval
    model.eval()
    val_loss = eval_loss(model, dataset, config)
    losses_log.append((config.max_iters - 1, val_loss))
    print(f"  step {config.max_iters-1:>5} | val {val_loss:.4f} | all bands | {time.time()-start:.1f}s")
    print(f"  Training complete in {time.time()-start:.1f}s")

    return model, losses_log


# =============================================================================
# Test: New knowledge absorption (catastrophic forgetting test)
# =============================================================================

def test_new_knowledge(model, dataset, config, label):
    """
    After training on Shakespeare, fine-tune briefly on a NEW pattern.
    Measure:
    - How fast does the model learn the new pattern?
    - How much Shakespeare knowledge does it forget?

    We use a distinctive pattern the model hasn't seen:
    Repeated sequences of "XYZXYZ" that the model should learn quickly.
    We create a small dataset with the pattern interleaved.
    """
    print(f"\n  NEW KNOWLEDGE TEST: {label}")
    print(f"  Can the model absorb new data without forgetting old?")

    model.eval()

    # Measure baseline Shakespeare quality
    baseline_loss = eval_loss(model, dataset, config, n_batches=30)
    print(f"    Shakespeare baseline: {baseline_loss:.4f}")

    # Create new pattern data: character-name completions
    # We'll fine-tune on "KING HENRY" and "DUKE YORK" patterns
    new_text = ""
    for _ in range(500):
        new_text += "The good king is Henry and the brave duke is York.\n"
        new_text += "King Henry rules the land while Duke York guards the north.\n"

    # Build a dataset from the new text (using same vocab)
    new_data = []
    for c in new_text:
        if c in dataset.stoi:
            new_data.append(dataset.stoi[c])
    new_data = torch.tensor(new_data, dtype=torch.long)

    # Save model state for restoration
    original_state = copy.deepcopy(model.state_dict())

    # Fine-tune with very few steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # lower LR for fine-tuning

    finetune_steps = [5, 10, 20, 50, 100]
    results = []

    for target_steps in finetune_steps:
        # Restore to pre-fine-tune state
        model.load_state_dict(copy.deepcopy(original_state))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(target_steps):
            # Random batch from new data
            ix = torch.randint(len(new_data) - config.block_size, (config.batch_size,))
            x = torch.stack([new_data[i:i+config.block_size] for i in ix]).to(config.device)
            y = torch.stack([new_data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        # Measure new pattern learning
        new_loss = 0.0
        for _ in range(10):
            ix = torch.randint(len(new_data) - config.block_size, (min(16, config.batch_size),))
            x = torch.stack([new_data[i:i+config.block_size] for i in ix]).to(config.device)
            y = torch.stack([new_data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
            with torch.no_grad():
                _, loss = model(x, y)
            new_loss += loss.item()
        new_loss /= 10

        # Measure Shakespeare forgetting
        shakespeare_loss = eval_loss(model, dataset, config, n_batches=30)
        forgetting = shakespeare_loss - baseline_loss

        results.append({
            "steps": target_steps,
            "new_loss": new_loss,
            "shakespeare_loss": shakespeare_loss,
            "forgetting": forgetting,
        })

        print(f"    {target_steps:>3} steps: new={new_loss:.4f}  shakespeare={shakespeare_loss:.4f}  forgetting={forgetting:+.4f}")

    # Restore original
    model.load_state_dict(original_state)

    return results, baseline_loss


# =============================================================================
# Test: Channel independence after training
# =============================================================================

def measure_independence(model, dataset, config, label):
    """Measure channel independence (from Phase 2 methodology)."""
    model.eval()
    n_harmonics = config.n_embd // 2

    all_acts = []
    for _ in range(20):
        x, _ = dataset.get_batch("val", config)
        with torch.no_grad():
            B, T = x.size()
            tok_emb = model.wte(x)
            pos_emb = model.wpe(T)
            h = tok_emb + pos_emb
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
        all_acts.append(h.cpu())

    flat = torch.cat(all_acts, dim=0).reshape(-1, config.n_embd).numpy()

    # Band energies
    band_energies = np.zeros((flat.shape[0], n_harmonics))
    for h in range(n_harmonics):
        ci, si = h * 2, h * 2 + 1
        band_energies[:, h] = flat[:, ci]**2 + flat[:, si]**2

    # Correlation matrix
    corr = np.corrcoef(band_energies.T)

    # Independence: fraction of pairs with |correlation| < 0.1
    n_pairs = 0
    n_independent = 0
    for i in range(n_harmonics):
        for j in range(i+1, n_harmonics):
            n_pairs += 1
            if abs(corr[i, j]) < 0.1:
                n_independent += 1

    independence = n_independent / n_pairs * 100
    print(f"    {label}: {independence:.1f}% independent pairs")
    return independence


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  PROGRESSIVE HARMONIC LEARNING")
    print(f"  Build the drawers in order — structure first, detail later")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # =====================================================================
    # Train both models
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPARISON")
    print(f"{'='*60}")

    baseline_model, baseline_log = train_baseline(config, dataset)
    progressive_model, progressive_log = train_progressive(config, dataset)

    # =====================================================================
    # Compare final quality
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  FINAL QUALITY COMPARISON")
    print(f"{'='*60}")

    b_loss = eval_loss(baseline_model, dataset, config, n_batches=50)
    p_loss = eval_loss(progressive_model, dataset, config, n_batches=50)

    print(f"\n  Baseline final val loss:     {b_loss:.4f}")
    print(f"  Progressive final val loss:  {p_loss:.4f}")
    print(f"  Difference:                  {p_loss - b_loss:+.4f} ({'progressive better' if p_loss < b_loss else 'baseline better'})")

    # =====================================================================
    # Compare learning curves
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  LEARNING CURVE COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'Step':>6} | {'Baseline':>10} | {'Progressive':>12} | {'Diff':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

    # Align on common steps
    b_dict = {step: loss for step, loss in baseline_log}
    p_dict = {step: loss for step, loss in progressive_log}
    common_steps = sorted(set(b_dict.keys()) & set(p_dict.keys()))

    for step in common_steps:
        bl = b_dict[step]
        pl = p_dict[step]
        diff = pl - bl
        print(f"  {step:>6} | {bl:>10.4f} | {pl:>12.4f} | {diff:>+8.4f}")

    # =====================================================================
    # Channel independence
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  CHANNEL INDEPENDENCE (Phase 2 metric)")
    print(f"{'='*60}")

    b_indep = measure_independence(baseline_model, dataset, config, "Baseline")
    p_indep = measure_independence(progressive_model, dataset, config, "Progressive")

    print(f"\n  Baseline independence:     {b_indep:.1f}%")
    print(f"  Progressive independence:  {p_indep:.1f}%")
    print(f"  Difference:                {p_indep - b_indep:+.1f} points")

    # =====================================================================
    # New knowledge absorption
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  NEW KNOWLEDGE ABSORPTION")
    print(f"  Fine-tune briefly on new data, measure forgetting")
    print(f"{'='*60}")

    b_results, b_baseline_loss = test_new_knowledge(baseline_model, dataset, config, "Baseline")
    p_results, p_baseline_loss = test_new_knowledge(progressive_model, dataset, config, "Progressive")

    # Compare
    print(f"\n  Comparison at each fine-tuning step:")
    print(f"  {'Steps':>6} | {'B new':>8} | {'P new':>8} | {'B forget':>10} | {'P forget':>10} | {'Winner':>10}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    p_wins_learn = 0
    p_wins_forget = 0

    for br, pr in zip(b_results, p_results):
        new_winner = "prog" if pr["new_loss"] < br["new_loss"] else "base"
        forget_winner = "prog" if abs(pr["forgetting"]) < abs(br["forgetting"]) else "base"

        if pr["new_loss"] < br["new_loss"]:
            p_wins_learn += 1
        if abs(pr["forgetting"]) < abs(br["forgetting"]):
            p_wins_forget += 1

        winner = "PROG" if (new_winner == "prog" and forget_winner == "prog") else \
                 "BASE" if (new_winner == "base" and forget_winner == "base") else "mixed"

        print(f"  {br['steps']:>6} | {br['new_loss']:>8.4f} | {pr['new_loss']:>8.4f} | {br['forgetting']:>+10.4f} | {pr['forgetting']:>+10.4f} | {winner:>10}")

    # =====================================================================
    # VERDICT
    # =====================================================================

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")

    print(f"\n  Final quality:    {'Progressive' if p_loss < b_loss else 'Baseline'} wins ({min(p_loss, b_loss):.4f} vs {max(p_loss, b_loss):.4f})")
    print(f"  Independence:     {'Progressive' if p_indep > b_indep else 'Baseline'} wins ({max(p_indep, b_indep):.1f}% vs {min(p_indep, b_indep):.1f}%)")
    print(f"  Learns new faster: Progressive {p_wins_learn}/{len(b_results)} times")
    print(f"  Forgets less:      Progressive {p_wins_forget}/{len(b_results)} times")

    total_score = 0
    if p_loss < b_loss:
        total_score += 1
    if p_indep > b_indep:
        total_score += 1
    if p_wins_learn > len(b_results) / 2:
        total_score += 1
    if p_wins_forget > len(b_results) / 2:
        total_score += 1

    if total_score >= 3:
        print(f"\n  CONFIRMED: Progressive harmonic learning outperforms baseline.")
        print(f"  Building the drawers in order — structure first, detail later —")
        print(f"  produces a model that learns better and forgets less.")
    elif total_score >= 2:
        print(f"\n  PARTIAL: Progressive learning shows advantages in some metrics.")
        print(f"  The ordering helps but doesn't dominate.")
    elif total_score >= 1:
        print(f"\n  WEAK: Progressive learning shows minor advantages.")
        print(f"  The foundation-first approach has limited benefit at this scale.")
    else:
        print(f"\n  NEGATIVE: Progressive learning does not outperform baseline.")
        print(f"  At this model size, the training order may not matter enough.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
