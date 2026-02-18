"""
Early Exit Test -- Can the Model Skip Layers for Easy Tokens?

PREREQUISITE: Phase 9 (Commitment Point) showed:
- Layer3_mlp delivers nearly half the total accuracy in one step
- Vowels and rare consonants commit at layer2_mlp (one layer early)
- ~30% of tokens may not need the full pipeline

THE TEST:
If the model's entropy at layer 2 is below a threshold, output the
prediction and skip layers 3-4. Measure:
1. What % of tokens exit early at each threshold?
2. What accuracy do early-exit tokens achieve?
3. What's the combined accuracy (early + full pipeline)?
4. What's the effective compute saving?

THE CHILD ANALOGY:
A kid doesn't parse "the" with the same effort as "nevertheless."
Easy things are fast. Hard things require the full pipeline.
The model should do the same.
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

    def forward_early_exit(self, idx, exit_layer, entropy_threshold):
        """
        Forward pass with early exit.
        At exit_layer, compute entropy per token position.
        Tokens below threshold exit early. Others continue to the end.
        Returns combined logits and per-token exit decisions.
        """
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        h = tok_emb + pos_emb

        # Run layers up to exit point
        for i in range(exit_layer + 1):
            h = self.blocks[i](h)

        # Compute early logits and entropy
        h_early = self.ln_f(h)
        logits_early = self.lm_head(h_early)
        probs_early = F.softmax(logits_early, dim=-1)
        entropy_early = -(probs_early * torch.log(probs_early + 1e-10)).sum(dim=-1)  # [B, T]

        # Decide which tokens exit
        exit_mask = entropy_early < entropy_threshold  # [B, T] True = exit early

        # Continue remaining layers for tokens that didn't exit
        # (In a real implementation you'd skip computation for exited tokens.
        #  Here we compute everything but combine results.)
        h_full = h.clone()
        for i in range(exit_layer + 1, len(self.blocks)):
            h_full = self.blocks[i](h_full)
        h_full = self.ln_f(h_full)
        logits_full = self.lm_head(h_full)

        # Combine: early logits where exited, full logits where continued
        combined_logits = torch.where(
            exit_mask.unsqueeze(-1).expand_as(logits_full),
            logits_early,
            logits_full
        )

        return combined_logits, logits_early, logits_full, exit_mask, entropy_early


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


def train_model(config, dataset):
    print(f"\n  Training harmonic model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

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

    print(f"  Training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# TEST 1: Early Exit Sweep (varying thresholds)
# =============================================================================

def test_early_exit_sweep(model, dataset, config, n_batches=30):
    """Try different entropy thresholds for early exit at each layer."""
    print("\n" + "=" * 60)
    print("  TEST 1: Early Exit Sweep")
    print("  At each exit layer, what threshold gives the best tradeoff?")
    print("=" * 60)

    model.eval()

    # First get full-model baseline
    baseline_correct = 0
    baseline_total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            logits, _ = model(x, y)
            preds = logits.argmax(dim=-1)
            baseline_correct += (preds == y).sum().item()
            baseline_total += y.numel()
    baseline_acc = baseline_correct / baseline_total
    print(f"\n  Full model baseline accuracy: {baseline_acc:.4f}")

    # Try exit at each layer with various thresholds
    exit_layers = [1, 2]  # after layer1_mlp (block index 1) or layer2_mlp (block index 2)
    thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    for exit_layer in exit_layers:
        print(f"\n  --- Exit after layer {exit_layer} MLP ---")
        print(f"  {'Threshold':>10} {'% exited':>10} {'Early acc':>10} {'Full acc':>10} {'Combined':>10} {'vs base':>10} {'Layers saved':>13}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*13}")

        for thresh in thresholds:
            total_tokens = 0
            exited_tokens = 0
            early_correct = 0
            full_correct = 0
            combined_correct = 0

            with torch.no_grad():
                for _ in range(n_batches):
                    x, y = dataset.get_batch("val", config)
                    combined_logits, logits_early, logits_full, exit_mask, entropy = \
                        model.forward_early_exit(x, exit_layer, thresh)

                    B, T = y.shape
                    total_tokens += B * T

                    # Count exits
                    n_exited = exit_mask.sum().item()
                    exited_tokens += n_exited

                    # Early accuracy (only on exited tokens)
                    early_preds = logits_early.argmax(dim=-1)
                    if n_exited > 0:
                        early_correct += (early_preds[exit_mask] == y[exit_mask]).sum().item()

                    # Full accuracy (only on non-exited tokens)
                    full_preds = logits_full.argmax(dim=-1)
                    n_full = B * T - n_exited
                    if n_full > 0:
                        full_correct += (full_preds[~exit_mask] == y[~exit_mask]).sum().item()

                    # Combined accuracy
                    combined_preds = combined_logits.argmax(dim=-1)
                    combined_correct += (combined_preds == y).sum().item()

            pct_exited = 100 * exited_tokens / total_tokens
            early_acc = early_correct / exited_tokens if exited_tokens > 0 else 0
            full_acc = full_correct / (total_tokens - exited_tokens) if (total_tokens - exited_tokens) > 0 else 0
            combined_acc = combined_correct / total_tokens
            delta = combined_acc - baseline_acc

            # Layers saved: exited tokens skip (n_layers - exit_layer - 1) layers
            layers_skipped = config.n_layer - exit_layer - 1
            effective_saving = pct_exited * layers_skipped / config.n_layer

            print(f"  {thresh:>10.1f} {pct_exited:>9.1f}% {early_acc:>10.4f} {full_acc:>10.4f} {combined_acc:>10.4f} {delta:>+10.4f} {effective_saving:>12.1f}%")

    return baseline_acc


# =============================================================================
# TEST 2: Per-Category Early Exit Analysis
# =============================================================================

def test_category_exit(model, dataset, config, n_batches=30):
    """Which character types benefit most from early exit?"""
    print("\n" + "=" * 60)
    print("  TEST 2: Per-Category Early Exit Analysis")
    print("  Which token types exit early most often?")
    print("=" * 60)

    model.eval()

    # Categorise characters
    char_categories = {}
    for c, idx in dataset.stoi.items():
        if c == ' ':
            char_categories[idx] = "space"
        elif c == '\n':
            char_categories[idx] = "newline"
        elif c in 'aeiou':
            char_categories[idx] = "vowel"
        elif c in 'tsnrhldcm':
            char_categories[idx] = "common_cons"
        elif c.isalpha() and c.islower():
            char_categories[idx] = "rare_cons"
        elif c.isupper():
            char_categories[idx] = "uppercase"
        elif c in '.,;:!?\'-':
            char_categories[idx] = "punctuation"
        else:
            char_categories[idx] = "other"

    # Use a reasonable threshold and exit at layer 2
    exit_layer = 2
    threshold = 1.5  # middle ground

    category_stats = {}
    for cat in set(char_categories.values()):
        category_stats[cat] = {"total": 0, "exited": 0, "early_correct": 0, "full_correct": 0}

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            combined_logits, logits_early, logits_full, exit_mask, entropy = \
                model.forward_early_exit(x, exit_layer, threshold)

            y_flat = y.view(-1).cpu().numpy()
            exit_flat = exit_mask.view(-1).cpu().numpy()
            early_preds = logits_early.argmax(dim=-1).view(-1).cpu().numpy()
            full_preds = logits_full.argmax(dim=-1).view(-1).cpu().numpy()

            for i, target in enumerate(y_flat):
                if target in char_categories:
                    cat = char_categories[target]
                    category_stats[cat]["total"] += 1
                    if exit_flat[i]:
                        category_stats[cat]["exited"] += 1
                        if early_preds[i] == target:
                            category_stats[cat]["early_correct"] += 1
                    else:
                        if full_preds[i] == target:
                            category_stats[cat]["full_correct"] += 1

    print(f"\n  Exit layer: {exit_layer}, entropy threshold: {threshold}")
    print(f"\n  {'Category':<15} {'Total':>8} {'% exited':>10} {'Early acc':>10} {'Full acc':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    sorted_cats = sorted(category_stats.keys(), key=lambda c: -category_stats[c]["total"])
    for cat in sorted_cats:
        s = category_stats[cat]
        if s["total"] < 100:
            continue
        pct_exit = 100 * s["exited"] / s["total"] if s["total"] > 0 else 0
        early_acc = s["early_correct"] / s["exited"] if s["exited"] > 0 else 0
        full_acc = s["full_correct"] / (s["total"] - s["exited"]) if (s["total"] - s["exited"]) > 0 else 0
        print(f"  {cat:<15} {s['total']:>8} {pct_exit:>9.1f}% {early_acc:>10.4f} {full_acc:>10.4f}")

    return category_stats


# =============================================================================
# TEST 3: Optimal Strategy
# =============================================================================

def test_optimal_strategy(model, dataset, config, n_batches=30):
    """Find the best exit strategy: which layer + threshold maximises
    the tradeoff between accuracy retention and compute saving."""
    print("\n" + "=" * 60)
    print("  TEST 3: Optimal Early Exit Strategy")
    print("  Best tradeoff between accuracy and compute saving")
    print("=" * 60)

    model.eval()

    # Get baseline
    baseline_correct = 0
    baseline_total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            logits, _ = model(x, y)
            preds = logits.argmax(dim=-1)
            baseline_correct += (preds == y).sum().item()
            baseline_total += y.numel()
    baseline_acc = baseline_correct / baseline_total

    # Fine grid search
    best_score = 0
    best_config = None
    results = []

    for exit_layer in [1, 2]:
        for thresh in np.arange(0.5, 3.1, 0.25):
            total_tokens = 0
            exited_tokens = 0
            combined_correct = 0

            with torch.no_grad():
                for _ in range(n_batches):
                    x, y = dataset.get_batch("val", config)
                    combined_logits, _, _, exit_mask, _ = \
                        model.forward_early_exit(x, exit_layer, thresh)
                    total_tokens += y.numel()
                    exited_tokens += exit_mask.sum().item()
                    combined_preds = combined_logits.argmax(dim=-1)
                    combined_correct += (combined_preds == y).sum().item()

            pct_exited = 100 * exited_tokens / total_tokens
            combined_acc = combined_correct / total_tokens
            acc_retention = combined_acc / baseline_acc
            layers_skipped = config.n_layer - exit_layer - 1
            compute_saving = pct_exited * layers_skipped / config.n_layer

            # Score: maximize compute saving while keeping accuracy above 99% of baseline
            if acc_retention >= 0.99:
                score = compute_saving
            elif acc_retention >= 0.95:
                score = compute_saving * acc_retention
            else:
                score = 0

            results.append({
                "exit_layer": exit_layer, "threshold": thresh,
                "pct_exited": pct_exited, "combined_acc": combined_acc,
                "acc_retention": acc_retention, "compute_saving": compute_saving,
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_config = results[-1]

    # Show top 5 strategies
    results.sort(key=lambda r: -r["score"])
    print(f"\n  Baseline accuracy: {baseline_acc:.4f}")
    print(f"\n  Top 5 strategies (ranked by compute saving at >=99% accuracy):")
    print(f"  {'Exit layer':>11} {'Threshold':>10} {'% exited':>10} {'Accuracy':>10} {'Retention':>10} {'Compute saved':>14}")
    print(f"  {'-'*11} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for r in results[:5]:
        print(f"  layer{r['exit_layer']}_mlp  {r['threshold']:>10.2f} {r['pct_exited']:>9.1f}% {r['combined_acc']:>10.4f} {100*r['acc_retention']:>9.1f}% {r['compute_saving']:>13.1f}%")

    if best_config:
        print(f"\n  BEST STRATEGY:")
        print(f"    Exit at: layer{best_config['exit_layer']}_mlp")
        print(f"    Entropy threshold: {best_config['threshold']:.2f}")
        print(f"    Tokens that exit early: {best_config['pct_exited']:.1f}%")
        print(f"    Combined accuracy: {best_config['combined_acc']:.4f} ({100*best_config['acc_retention']:.1f}% of baseline)")
        print(f"    Effective compute saving: {best_config['compute_saving']:.1f}%")
    else:
        print(f"\n  No strategy achieved >=95% accuracy retention.")

    return best_config, results


# =============================================================================
# TEST 4: Generation with Early Exit
# =============================================================================

def test_generation(model, dataset, config):
    """Generate text with early exit enabled vs disabled. Do they differ?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Generation Quality Comparison")
    print("  Full pipeline vs early exit -- can you tell the difference?")
    print("=" * 60)

    model.eval()
    seed_text = "KING RICHARD:\nMy lord"

    for mode_name, use_early_exit in [("Full pipeline", False), ("Early exit (layer2, thresh=1.5)", True)]:
        tokens = dataset.stoi.copy()
        input_ids = torch.tensor([dataset.stoi[c] for c in seed_text if c in dataset.stoi],
                                 dtype=torch.long, device=config.device).unsqueeze(0)

        generated = list(input_ids[0].cpu().numpy())
        n_early = 0
        n_total = 0

        with torch.no_grad():
            for _ in range(200):
                idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)

                if use_early_exit:
                    combined_logits, _, _, exit_mask, _ = model.forward_early_exit(idx, 2, 1.5)
                    logits = combined_logits[0, -1, :]
                    if exit_mask[0, -1].item():
                        n_early += 1
                    n_total += 1
                else:
                    logits_out, _ = model(idx)
                    logits = logits_out[0, -1, :]

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)

        text = "".join([dataset.itos[t] for t in generated])
        print(f"\n  {mode_name}:")
        print(f"  {'-' * 50}")
        # Print safely
        safe_text = text[:300].encode('ascii', errors='replace').decode('ascii')
        for line in safe_text.split('\n')[:8]:
            print(f"    {line}")
        if use_early_exit:
            print(f"  [{n_early}/{n_total} tokens exited early ({100*n_early/n_total:.1f}%)]")


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict(baseline_acc, best_config, category_stats):
    print("\n" + "=" * 60)
    print("  VERDICT: Is Early Exit Viable?")
    print("=" * 60)

    if best_config and best_config["acc_retention"] >= 0.99:
        print(f"\n  YES. Early exit is viable.")
        print(f"\n  Best configuration:")
        print(f"    Exit layer:        layer{best_config['exit_layer']}_mlp")
        print(f"    Entropy threshold: {best_config['threshold']:.2f}")
        print(f"    Tokens exited:     {best_config['pct_exited']:.1f}%")
        print(f"    Accuracy retained: {100*best_config['acc_retention']:.1f}%")
        print(f"    Compute saved:     {best_config['compute_saving']:.1f}%")
        print(f"\n  At scale (billions of tokens), {best_config['compute_saving']:.1f}% compute")
        print(f"  saving with {100*best_config['acc_retention']:.1f}% accuracy is significant.")
    elif best_config and best_config["acc_retention"] >= 0.95:
        print(f"\n  PARTIALLY. Early exit works but with a quality tradeoff.")
        print(f"    Best accuracy retention: {100*best_config['acc_retention']:.1f}%")
        print(f"    Compute saved: {best_config['compute_saving']:.1f}%")
    else:
        print(f"\n  NOT YET. No configuration preserves >=95% accuracy.")
        print(f"  The model is too small -- all 4 layers are needed.")
        print(f"  This may change with larger models that have more redundant layers.")

    # Category insight
    print(f"\n  Token categories most suitable for early exit:")
    sorted_cats = sorted(category_stats.keys(),
                        key=lambda c: -(category_stats[c]["exited"] / category_stats[c]["total"]
                                        if category_stats[c]["total"] > 100 else 0))
    for cat in sorted_cats[:5]:
        s = category_stats[cat]
        if s["total"] < 100:
            continue
        pct = 100 * s["exited"] / s["total"]
        early_acc = s["early_correct"] / s["exited"] if s["exited"] > 0 else 0
        print(f"    {cat:<15} {pct:>5.1f}% exit early, {early_acc:.3f} accuracy when they do")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  EARLY EXIT TEST")
    print("  Can the model skip layers for easy tokens?")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    model = train_model(config, dataset)

    baseline_acc = test_early_exit_sweep(model, dataset, config)
    category_stats = test_category_exit(model, dataset, config)
    best_config, all_results = test_optimal_strategy(model, dataset, config)
    test_generation(model, dataset, config)

    print_verdict(baseline_acc, best_config, category_stats)


if __name__ == "__main__":
    main()
