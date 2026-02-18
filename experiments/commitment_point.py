"""
Commitment Point — Where Does the Model Make Its Decision?

THE QUESTION:
At which layer does the model commit to a prediction? Is there a point
where it stops being uncertain and starts executing? And which harmonic
bands does it need to reach that commitment?

This must be answered BEFORE selective loading (Idea 2), because you
can't design partial loading until you know what must always be loaded.

THE OS KERNEL ANALOGY:
- Layers 0-2 might be the kernel (always in memory)
- Layer 3+ and high bands might be applications (loadable on demand)
- Easy tokens might exit at layer 2, hard tokens need all layers

WHAT WE MEASURE:
1. Per-layer prediction entropy — run lm_head on hidden state at each
   layer. Where does entropy drop? That's the commitment point.
2. Band contribution to commitment — zero out band groups, measure
   entropy change. Which bands are needed for confidence?
3. Token-dependent depth — do easy tokens (space, common letters) commit
   earlier than hard tokens (rare chars, ambiguous context)?
4. Early exit quality — if we actually predict from layer 2, how good
   is it compared to the full model?
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
# Model (with per-layer logit extraction)
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

    def forward_per_layer(self, idx):
        """Run forward pass, returning logits at EVERY layer stage."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        h = tok_emb + pos_emb

        layer_logits = {}

        # Embedding-level prediction
        emb_logits = self.lm_head(self.ln_f(h))
        layer_logits["embedding"] = emb_logits.detach()

        for i, block in enumerate(self.blocks):
            # Post-attention
            ln1_out = block.ln_1(h)
            attn_out = block.attn(ln1_out)
            h = h + attn_out
            logits_attn = self.lm_head(self.ln_f(h))
            layer_logits[f"layer{i}_attn"] = logits_attn.detach()

            # Post-MLP
            ln2_out = block.ln_2(h)
            mlp_out = block.mlp(ln2_out)
            h = h + mlp_out
            logits_mlp = self.lm_head(self.ln_f(h))
            layer_logits[f"layer{i}_mlp"] = logits_mlp.detach()

        # Final (same as layer3_mlp + ln_f, but explicit)
        layer_logits["final"] = layer_logits[f"layer{self.config.n_layer - 1}_mlp"]

        return layer_logits, h

    def forward_with_band_ablation(self, idx, band_mask):
        """Forward pass with specific bands zeroed out at embedding level."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        # Apply band mask to token embeddings
        tok_emb = tok_emb * band_mask.unsqueeze(0).unsqueeze(0)
        pos_emb = self.wpe(T)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


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
# TEST 1: Per-Layer Prediction Entropy
# =============================================================================

def test_per_layer_entropy(model, dataset, config, n_batches=20):
    """At which layer does the model become confident about its prediction?"""
    print("\n" + "=" * 60)
    print("  TEST 1: Per-Layer Prediction Entropy")
    print("  At which layer does the model commit?")
    print("=" * 60)

    model.eval()
    layer_entropies = {}
    layer_accuracies = {}
    layer_top1_probs = {}

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            per_layer_logits, _ = model.forward_per_layer(x)

            for name, logits in per_layer_logits.items():
                probs = F.softmax(logits, dim=-1)
                # Entropy: -sum(p * log(p))
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1).mean().item()

                # Accuracy: does this layer predict the right next token?
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()

                # Top-1 probability (confidence)
                top1_prob = probs.max(dim=-1).values.mean().item()

                if name not in layer_entropies:
                    layer_entropies[name] = []
                    layer_accuracies[name] = []
                    layer_top1_probs[name] = []
                layer_entropies[name].append(entropy)
                layer_accuracies[name].append(acc)
                layer_top1_probs[name].append(top1_prob)

    # Report
    max_entropy = math.log(dataset.vocab_size)  # uniform distribution
    print(f"\n  Max possible entropy (uniform over {dataset.vocab_size} chars): {max_entropy:.4f}")
    print(f"\n  {'Layer':<20} {'Entropy':>10} {'% of max':>10} {'Accuracy':>10} {'Top-1 prob':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    layer_names = list(layer_entropies.keys())
    results = {}
    prev_entropy = max_entropy
    biggest_drop_name = None
    biggest_drop_val = 0

    for name in layer_names:
        ent = np.mean(layer_entropies[name])
        acc = np.mean(layer_accuracies[name])
        top1 = np.mean(layer_top1_probs[name])
        pct = 100 * ent / max_entropy
        drop = prev_entropy - ent

        if drop > biggest_drop_val and name != "embedding":
            biggest_drop_val = drop
            biggest_drop_name = name

        results[name] = {"entropy": ent, "accuracy": acc, "top1_prob": top1, "drop": drop}
        print(f"  {name:<20} {ent:>10.4f} {pct:>9.1f}% {acc:>10.4f} {top1:>10.4f}")
        prev_entropy = ent

    print(f"\n  Biggest entropy drop: {biggest_drop_name} (delta = {biggest_drop_val:.4f})")
    print(f"  >> This is the COMMITMENT POINT — where the model gains the most confidence.")

    return results


# =============================================================================
# TEST 2: Band Contribution to Commitment
# =============================================================================

def test_band_contribution(model, dataset, config, n_batches=20):
    """Which harmonic bands are needed for the model to be confident?"""
    print("\n" + "=" * 60)
    print("  TEST 2: Band Contribution to Commitment")
    print("  Which bands does the model need to make decisions?")
    print("=" * 60)

    model.eval()
    n_harmonics = config.n_embd // 2

    # First get baseline entropy (all bands)
    baseline_entropy = 0
    baseline_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            logits, loss = model(x, y)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            baseline_entropy += entropy
            baseline_loss += loss.item()
    baseline_entropy /= n_batches
    baseline_loss /= n_batches

    print(f"\n  Baseline (all bands): entropy={baseline_entropy:.4f}, loss={baseline_loss:.4f}")

    # Test band groups
    band_groups = {
        "Low only (1-16)": (0, 16),
        "Mid only (17-40)": (16, 40),
        "High only (41-64)": (40, 64),
        "Low+Mid (1-40)": (0, 40),
        "Mid+High (17-64)": (16, 64),
        "Low+High (1-16, 41-64)": "low_high",
    }

    print(f"\n  {'Band group':<25} {'Entropy':>10} {'Loss':>10} {'Ent increase':>14} {'Loss increase':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*14} {'-'*14}")

    group_results = {}
    for group_name, band_range in band_groups.items():
        # Build mask
        mask = torch.zeros(config.n_embd, device=config.device)
        if band_range == "low_high":
            for h in range(16):
                mask[h * 2] = 1.0
                mask[h * 2 + 1] = 1.0
            for h in range(40, 64):
                mask[h * 2] = 1.0
                mask[h * 2 + 1] = 1.0
        else:
            start, end = band_range
            for h in range(start, end):
                mask[h * 2] = 1.0
                mask[h * 2 + 1] = 1.0

        group_entropy = 0
        group_loss = 0
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.get_batch("val", config)
                logits = model.forward_with_band_ablation(x, mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                group_entropy += entropy
                group_loss += loss.item()
        group_entropy /= n_batches
        group_loss /= n_batches

        ent_delta = group_entropy - baseline_entropy
        loss_delta = group_loss - baseline_loss
        group_results[group_name] = {
            "entropy": group_entropy, "loss": group_loss,
            "ent_delta": ent_delta, "loss_delta": loss_delta
        }
        print(f"  {group_name:<25} {group_entropy:>10.4f} {group_loss:>10.4f} {ent_delta:>+14.4f} {loss_delta:>+14.4f}")

    # Find minimum viable set
    print(f"\n  Analysis:")
    best_partial = min(group_results.items(), key=lambda x: x[1]["loss"])
    print(f"    Best partial set: {best_partial[0]} (loss={best_partial[1]['loss']:.4f}, vs baseline {baseline_loss:.4f})")
    print(f"    Loss increase: {best_partial[1]['loss_delta']:+.4f} ({100*best_partial[1]['loss_delta']/baseline_loss:+.1f}%)")

    return baseline_entropy, group_results


# =============================================================================
# TEST 3: Token-Dependent Commitment Depth
# =============================================================================

def test_token_dependent_depth(model, dataset, config, n_batches=20):
    """Do easy tokens commit earlier than hard tokens?"""
    print("\n" + "=" * 60)
    print("  TEST 3: Token-Dependent Commitment Depth")
    print("  Do easy tokens commit earlier than hard tokens?")
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
            char_categories[idx] = "common_consonant"
        elif c.isalpha() and c.islower():
            char_categories[idx] = "rare_consonant"
        elif c.isupper():
            char_categories[idx] = "uppercase"
        elif c in '.,;:!?\'-':
            char_categories[idx] = "punctuation"
        else:
            char_categories[idx] = "other"

    # Per-category, per-layer: when does the model get the prediction right?
    category_layer_acc = {cat: {} for cat in set(char_categories.values())}
    category_layer_entropy = {cat: {} for cat in set(char_categories.values())}
    category_counts = {cat: 0 for cat in set(char_categories.values())}

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            per_layer_logits, _ = model.forward_per_layer(x)

            # For each target token, check which category it belongs to
            y_flat = y.view(-1).cpu().numpy()

            for name, logits in per_layer_logits.items():
                preds = logits.view(-1, logits.size(-1)).argmax(dim=-1).cpu().numpy()
                probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu().numpy()

                for i, (target, pred) in enumerate(zip(y_flat, preds)):
                    if target in char_categories:
                        cat = char_categories[target]
                        if name not in category_layer_acc[cat]:
                            category_layer_acc[cat][name] = []
                            category_layer_entropy[cat][name] = []
                        category_layer_acc[cat][name].append(1.0 if pred == target else 0.0)
                        category_layer_entropy[cat][name].append(ent[i])
                        if name == "embedding":
                            category_counts[cat] += 1

    # Report: for each category, show entropy at each layer
    layer_names = list(next(iter(category_layer_entropy.values())).keys())

    # Find commitment point per category (layer with biggest entropy drop)
    print(f"\n  Per-category entropy at each layer:")
    print(f"  {'Category':<20}", end="")
    for name in layer_names:
        short = name.replace("layer", "L").replace("_attn", "a").replace("_mlp", "m")
        print(f" {short:>8}", end="")
    print(f" {'Commit':>10} {'Count':>8}")
    print(f"  {'-'*20}", end="")
    for _ in layer_names:
        print(f" {'-'*8}", end="")
    print(f" {'-'*10} {'-'*8}")

    category_commitment = {}
    sorted_cats = sorted(category_counts.keys(), key=lambda c: -category_counts[c])

    for cat in sorted_cats:
        if category_counts[cat] < 100:
            continue
        print(f"  {cat:<20}", end="")
        prev_ent = None
        biggest_drop = 0
        commit_layer = "?"

        for name in layer_names:
            if name in category_layer_entropy[cat]:
                ent = np.mean(category_layer_entropy[cat][name])
                print(f" {ent:>8.3f}", end="")
                if prev_ent is not None:
                    drop = prev_ent - ent
                    if drop > biggest_drop:
                        biggest_drop = drop
                        commit_layer = name
                prev_ent = ent
            else:
                print(f" {'---':>8}", end="")

        category_commitment[cat] = commit_layer
        count = category_counts[cat]
        print(f" {commit_layer:>10} {count:>8}")

    # Summary: do easy tokens commit earlier?
    print(f"\n  Commitment points by category:")
    for cat in sorted_cats:
        if cat in category_commitment:
            # Also show final-layer accuracy
            if "final" in category_layer_acc[cat]:
                final_acc = np.mean(category_layer_acc[cat]["final"])
            else:
                last_layer = layer_names[-1]
                final_acc = np.mean(category_layer_acc[cat][last_layer])
            print(f"    {cat:<20} commits at: {category_commitment[cat]:<18} final acc: {final_acc:.3f}")

    return category_commitment


# =============================================================================
# TEST 4: Early Exit Quality
# =============================================================================

def test_early_exit(model, dataset, config, n_batches=30):
    """If we predict from each layer, how good is it?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Early Exit Quality")
    print("  How much do later layers actually improve predictions?")
    print("=" * 60)

    model.eval()
    layer_losses = {}
    layer_accs = {}

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            per_layer_logits, _ = model.forward_per_layer(x)

            for name, logits in per_layer_logits.items():
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()

                if name not in layer_losses:
                    layer_losses[name] = []
                    layer_accs[name] = []
                layer_losses[name].append(loss.item())
                layer_accs[name].append(acc)

    layer_names = list(layer_losses.keys())
    final_loss = np.mean(layer_losses[layer_names[-1]])
    final_acc = np.mean(layer_accs[layer_names[-1]])

    print(f"\n  {'Layer':<20} {'Loss':>10} {'Accuracy':>10} {'vs Final':>10} {'% of final':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for name in layer_names:
        loss = np.mean(layer_losses[name])
        acc = np.mean(layer_accs[name])
        delta = loss - final_loss
        pct = 100 * acc / final_acc if final_acc > 0 else 0
        print(f"  {name:<20} {loss:>10.4f} {acc:>10.4f} {delta:>+10.4f} {pct:>11.1f}%")

    # Where is 90% of final accuracy achieved?
    print(f"\n  Final layer: loss={final_loss:.4f}, accuracy={final_acc:.4f}")
    for name in layer_names:
        acc = np.mean(layer_accs[name])
        if acc >= 0.9 * final_acc:
            print(f"  90% of final accuracy reached at: {name} ({acc:.4f} >= {0.9*final_acc:.4f})")
            break

    for name in layer_names:
        acc = np.mean(layer_accs[name])
        if acc >= 0.95 * final_acc:
            print(f"  95% of final accuracy reached at: {name} ({acc:.4f} >= {0.95*final_acc:.4f})")
            break

    return layer_losses, layer_accs


# =============================================================================
# VERDICT
# =============================================================================

def verdict(layer_results, band_results, category_commitment):
    print("\n" + "=" * 60)
    print("  VERDICT: Where Does the Model Decide?")
    print("=" * 60)

    # Commitment point from entropy
    layers_ordered = list(layer_results.keys())
    biggest_drop_layer = max(
        [(name, layer_results[name]["drop"]) for name in layers_ordered if name != "embedding"],
        key=lambda x: x[1]
    )

    print(f"\n  1. COMMITMENT POINT: {biggest_drop_layer[0]}")
    print(f"     Biggest entropy drop: {biggest_drop_layer[1]:.4f}")
    print(f"     The model gains the most confidence here.")

    # Band requirement
    baseline_ent, group_results = band_results
    best_group = min(group_results.items(), key=lambda x: x[1]["loss"])
    print(f"\n  2. MINIMUM VIABLE BANDS: {best_group[0]}")
    print(f"     Loss with only these bands: {best_group[1]['loss']:.4f}")
    print(f"     Loss increase vs full: {best_group[1]['loss_delta']:+.4f}")

    # Token-dependent depth
    commit_layers = list(category_commitment.values())
    unique_commits = set(commit_layers)
    print(f"\n  3. TOKEN-DEPENDENT DEPTH:")
    if len(unique_commits) == 1:
        print(f"     All token types commit at the same layer: {unique_commits.pop()}")
        print(f"     >> The kernel is fixed-depth.")
    else:
        print(f"     Different token types commit at different layers:")
        for cat, layer in sorted(category_commitment.items()):
            print(f"       {cat:<20} -> {layer}")
        print(f"     >> The kernel depth is DYNAMIC — varies by token difficulty.")

    # OS kernel mapping
    print(f"\n  OS KERNEL MAPPING:")
    print(f"    BIOS (always needed):  Low bands + Layer 0")
    print(f"    Kernel (decision):     {biggest_drop_layer[0]}")
    print(f"    Minimum bands:         {best_group[0]}")
    print(f"    Applications (optional): Everything after commitment + remaining bands")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  COMMITMENT POINT — Where Does the Model Decide?")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    model = train_model(config, dataset)

    # Run all tests
    layer_results = test_per_layer_entropy(model, dataset, config)
    band_results = test_band_contribution(model, dataset, config)
    category_commitment = test_token_dependent_depth(model, dataset, config)
    test_early_exit(model, dataset, config)

    # Verdict
    verdict(layer_results, band_results, category_commitment)


if __name__ == "__main__":
    main()
