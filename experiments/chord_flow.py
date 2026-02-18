"""
Chord Flow -- Can the Model Process Chords Instead of Tokens?

PREREQUISITE: Phase 7 (Concept Composition) showed:
- Characters compose into context-dependent chords by layer 2 attention
- 33-million-fold divergence growth: same character -> different representations
- Concept clustering peaks at layer2_post_attn

Phase 9 (Commitment Point) showed:
- Layer3_mlp is where the model decides
- Layers 0-2 build context progressively

Phase 10 (Early Exit) showed:
- Token-level early exit saves 2-4% in a 4-layer model
- The savings come from skipping layers for easy tokens

THE NEW IDEA:
What if instead of skipping layers for individual tokens, we let
attention compose characters into chords (words), then MERGE the
chord positions before sending them through the expensive upper layers?

The savings come not from depth (skipping layers) but from WIDTH
(reducing sequence length). Attention is O(n^2) -- cutting sequence
length in half saves ~75% of attention compute in the upper layers.

The approach:
1. Run layers 0-2 normally (characters -> chords)
2. Detect chord boundaries from representation similarity
3. Pool characters within each chord into one vector
4. Run layer 3+ on the shorter chord sequence
5. Expand back to character level for prediction

This is dynamic token merging, applied at the exact point where
Phase 7 says composition has already happened.
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
# Model (same architecture as other phases)
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

    def forward_through_layer(self, idx, up_to_layer):
        """Run forward pass through layers 0..up_to_layer, return hidden states."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        h = tok_emb + pos_emb
        for i in range(up_to_layer + 1):
            h = self.blocks[i](h)
        return h

    def forward_from_layer(self, h, from_layer):
        """Continue forward pass from from_layer onward, return logits."""
        for i in range(from_layer, len(self.blocks)):
            h = self.blocks[i](h)
        h = self.ln_f(h)
        return self.lm_head(h)


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
# Chord Detection Utilities
# =============================================================================

def detect_chord_boundaries(hidden_states, threshold):
    """
    Detect chord boundaries by measuring cosine similarity between adjacent positions.
    High similarity = same chord. Drop below threshold = new chord starts.

    Returns boundary mask: True at positions where a new chord begins.
    Position 0 is always a boundary (first chord starts there).
    """
    # hidden_states: [B, T, C]
    B, T, C = hidden_states.shape

    # Cosine similarity between adjacent positions
    h_norm = F.normalize(hidden_states, dim=-1)
    # sim[t] = similarity between position t and t+1
    sim = (h_norm[:, :-1, :] * h_norm[:, 1:, :]).sum(dim=-1)  # [B, T-1]

    # Boundary = where similarity drops below threshold
    boundaries = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device)
    boundaries[:, 0] = True  # first position always starts a chord
    boundaries[:, 1:] = sim < threshold  # low similarity = new chord

    return boundaries, sim


def pool_chords(hidden_states, boundaries):
    """
    Pool positions within each chord into a single vector (mean pooling).
    Returns:
    - pooled: [B, max_chords, C] - pooled chord representations
    - chord_lengths: [B, max_chords] - how many tokens in each chord
    - chord_map: [B, T] - which chord each token belongs to
    - n_chords: [B] - number of chords per batch element
    """
    B, T, C = hidden_states.shape
    device = hidden_states.device

    # Assign chord IDs: cumulative sum of boundaries
    chord_map = boundaries.long().cumsum(dim=1) - 1  # [B, T], 0-indexed

    n_chords = chord_map[:, -1] + 1  # [B]
    max_chords = n_chords.max().item()

    pooled = torch.zeros(B, max_chords, C, device=device)
    chord_lengths = torch.zeros(B, max_chords, dtype=torch.long, device=device)

    for b in range(B):
        for chord_id in range(n_chords[b].item()):
            mask = chord_map[b] == chord_id
            count = mask.sum().item()
            if count > 0:
                pooled[b, chord_id] = hidden_states[b, mask].mean(dim=0)
                chord_lengths[b, chord_id] = count

    return pooled, chord_lengths, chord_map, n_chords


def expand_chords(chord_hidden, chord_map, T):
    """
    Expand pooled chord representations back to token level.
    Each token gets the representation of its chord.
    """
    B, max_chords, C = chord_hidden.shape
    expanded = torch.zeros(B, T, C, device=chord_hidden.device)

    for b in range(B):
        for t in range(T):
            chord_id = chord_map[b, t].item()
            expanded[b, t] = chord_hidden[b, chord_id]

    return expanded


# =============================================================================
# TEST 1: Chord Boundary Detection
# =============================================================================

def test_chord_detection(model, dataset, config, n_batches=10):
    """
    After layer 2 attention, do adjacent characters in the same word
    have higher similarity than characters across word boundaries?
    Can we detect natural chord boundaries?
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Chord Boundary Detection")
    print("  Do words form natural similarity clusters by layer 2?")
    print("=" * 60)

    model.eval()

    # Collect similarity stats at word boundaries vs within words
    within_word_sims = []
    cross_boundary_sims = []

    # Also collect at different layers to see where chords emerge
    layer_within = {i: [] for i in range(config.n_layer)}
    layer_cross = {i: [] for i in range(config.n_layer)}

    with torch.no_grad():
        for batch_idx in range(n_batches):
            x, y = dataset.get_batch("val", config)
            B, T = x.shape

            # Decode to find actual word boundaries
            for b in range(min(B, 4)):  # sample a few per batch
                chars = [dataset.itos[x[b, t].item()] for t in range(T)]

                # Word boundary = space, newline, or punctuation
                is_boundary = []
                for t in range(1, T):
                    prev_char = chars[t-1]
                    curr_char = chars[t]
                    boundary = (prev_char in ' \n' or curr_char in ' \n' or
                                prev_char in '.,;:!?\'-' or curr_char in '.,;:!?\'-' or
                                prev_char.isupper() != curr_char.isupper())
                    is_boundary.append(boundary)

                # Get hidden states at each layer
                tok_emb = model.wte(x[b:b+1])
                pos_emb = model.wpe(T)
                h = tok_emb + pos_emb

                for layer_idx in range(config.n_layer):
                    h = model.blocks[layer_idx](h)

                    h_norm = F.normalize(h[0], dim=-1)
                    sims = (h_norm[:-1] * h_norm[1:]).sum(dim=-1).cpu().numpy()

                    for t in range(len(is_boundary)):
                        if is_boundary[t]:
                            layer_cross[layer_idx].append(sims[t])
                        else:
                            layer_within[layer_idx].append(sims[t])

    # Report
    print(f"\n  Adjacent-position cosine similarity (within-word vs cross-boundary):")
    print(f"\n  {'Layer':<16} {'Within word':>12} {'Cross boundary':>15} {'Gap':>10} {'Signal':>10}")
    print(f"  {'-'*16} {'-'*12} {'-'*15} {'-'*10} {'-'*10}")

    best_gap = 0
    best_layer = 0
    for layer_idx in range(config.n_layer):
        w_mean = np.mean(layer_within[layer_idx])
        c_mean = np.mean(layer_cross[layer_idx])
        gap = w_mean - c_mean
        signal = "STRONG" if gap > 0.05 else "moderate" if gap > 0.02 else "weak" if gap > 0 else "none"
        print(f"  layer{layer_idx}_mlp    {w_mean:>12.4f} {c_mean:>15.4f} {gap:>+10.4f} {signal:>10}")
        if gap > best_gap:
            best_gap = gap
            best_layer = layer_idx

    print(f"\n  Best chord separation: layer{best_layer} (gap = {best_gap:+.4f})")

    # Distribution of similarities at best layer
    w_arr = np.array(layer_within[best_layer])
    c_arr = np.array(layer_cross[best_layer])

    print(f"\n  Similarity distribution at layer {best_layer}:")
    print(f"    Within-word:    mean={np.mean(w_arr):.4f}, std={np.std(w_arr):.4f}, "
          f"min={np.min(w_arr):.4f}, max={np.max(w_arr):.4f}")
    print(f"    Cross-boundary: mean={np.mean(c_arr):.4f}, std={np.std(c_arr):.4f}, "
          f"min={np.min(c_arr):.4f}, max={np.max(c_arr):.4f}")

    # What threshold would separate them?
    all_sims = np.concatenate([w_arr, c_arr])
    all_labels = np.concatenate([np.ones(len(w_arr)), np.zeros(len(c_arr))])
    thresholds_to_try = np.arange(0.5, 1.0, 0.02)

    print(f"\n  Threshold sweep for chord detection at layer {best_layer}:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg chord len':>14}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for thresh in thresholds_to_try:
        # Positions with sim >= thresh are "within word" (no boundary)
        # Positions with sim < thresh are "boundary"
        predicted_boundary = all_sims < thresh
        actual_boundary = all_labels == 0

        tp = np.sum(predicted_boundary & actual_boundary)
        fp = np.sum(predicted_boundary & ~actual_boundary)
        fn = np.sum(~predicted_boundary & actual_boundary)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Estimate avg chord length
        n_boundaries = np.sum(all_sims < thresh) + 1  # +1 for first position
        avg_chord_len = len(all_sims) / n_boundaries if n_boundaries > 0 else len(all_sims)

        print(f"  {thresh:>10.2f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {avg_chord_len:>14.1f}")

    return best_layer, best_gap, layer_within, layer_cross


# =============================================================================
# TEST 2: Chord Pooling Accuracy
# =============================================================================

def test_chord_pooling(model, dataset, config, merge_layer=2, n_batches=30):
    """
    Pool token representations into chords at merge_layer, then run
    the remaining layers on the shorter sequence. Expand back to token
    level and measure accuracy vs the full pipeline.
    """
    print("\n" + "=" * 60)
    print("  TEST 2: Chord Pooling Accuracy")
    print("  Pool characters into chords, run upper layers on chords.")
    print(f"  Merge point: after layer {merge_layer}")
    print("=" * 60)

    model.eval()

    thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Full pipeline baseline
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

    print(f"\n  Full pipeline baseline accuracy: {baseline_acc:.4f}")
    print(f"\n  {'Threshold':>10} {'Avg chords':>11} {'Compression':>12} {'Chord acc':>10} {'vs base':>10} {'Attn saving':>12}")
    print(f"  {'-'*10} {'-'*11} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

    for thresh in thresholds:
        total_tokens = 0
        chord_correct = 0
        total_chords = 0
        total_seq_len = 0

        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.get_batch("val", config)
                B, T = x.shape
                total_tokens += B * T
                total_seq_len += B * T

                # Run through merge_layer
                h = model.forward_through_layer(x, merge_layer)

                # Detect chord boundaries
                boundaries, _ = detect_chord_boundaries(h, thresh)

                # Pool into chords
                pooled, chord_lengths, chord_map, n_chords = pool_chords(h, boundaries)
                total_chords += n_chords.sum().item()

                # Run remaining layers on pooled chords
                # We need to handle the causal mask carefully here.
                # For each batch element, the pooled sequence is shorter.
                # Process each batch element individually for correct masking.
                for b in range(B):
                    nc = n_chords[b].item()
                    if nc == 0:
                        continue

                    chord_h = pooled[b:b+1, :nc, :]  # [1, nc, C]

                    # Run remaining layers
                    # The attention mask needs to be causal over chord positions
                    for layer_idx in range(merge_layer + 1, len(model.blocks)):
                        block = model.blocks[layer_idx]
                        # Manual forward to handle variable-length sequences
                        residual = chord_h
                        chord_h_norm = block.ln_1(chord_h)
                        # Attention with proper causal mask for this length
                        bsz, tgt_len, emb = chord_h_norm.shape
                        qkv = block.attn.c_attn(chord_h_norm)
                        q, k, v = qkv.split(block.attn.n_embd, dim=2)
                        head_dim = emb // block.attn.n_head
                        q = q.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                        k = k.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                        v = v.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
                        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=x.device))
                        att = att.masked_fill(causal.view(1, 1, tgt_len, tgt_len) == 0, float("-inf"))
                        att = F.softmax(att, dim=-1)
                        attn_out = att @ v
                        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, emb)
                        attn_out = block.attn.c_proj(attn_out)
                        chord_h = residual + attn_out
                        chord_h = chord_h + block.mlp(block.ln_2(chord_h))

                    chord_h = model.ln_f(chord_h)
                    chord_logits = model.lm_head(chord_h)  # [1, nc, vocab]

                    # Expand back: each token gets its chord's prediction
                    for t in range(T):
                        chord_id = chord_map[b, t].item()
                        if chord_id < nc:
                            pred = chord_logits[0, chord_id].argmax().item()
                            if pred == y[b, t].item():
                                chord_correct += 1

        chord_acc = chord_correct / total_tokens if total_tokens > 0 else 0
        avg_chords = total_chords / (n_batches * config.batch_size)
        compression = avg_chords / config.block_size
        # Attention saving: O(n^2), so saving = 1 - (n_chord/n_token)^2
        # But only for the upper layers (n_layer - merge_layer - 1 out of n_layer)
        upper_fraction = (config.n_layer - merge_layer - 1) / config.n_layer
        attn_saving = upper_fraction * (1 - compression ** 2) * 100
        delta = chord_acc - baseline_acc

        print(f"  {thresh:>10.2f} {avg_chords:>11.1f} {compression:>11.1f}x {chord_acc:>10.4f} {delta:>+10.4f} {attn_saving:>11.1f}%")

    return baseline_acc


# =============================================================================
# TEST 3: Chord-Level Entropy vs Token-Level Entropy
# =============================================================================

def test_chord_entropy(model, dataset, config, merge_layer=2, n_batches=20):
    """
    Does pooling characters into chords produce LOWER entropy predictions
    than individual tokens? Chords carry more context per unit -- they
    should be more decisive.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Chord-Level vs Token-Level Entropy")
    print("  Do chords produce more confident predictions?")
    print("=" * 60)

    model.eval()

    token_entropies = []
    chord_entropies = []  # per threshold
    thresholds = [0.70, 0.80, 0.90]

    # Token-level entropy at merge layer
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            B, T = x.shape

            # Get hidden states at merge layer
            h = model.forward_through_layer(x, merge_layer)

            # Token-level: predict from each position directly
            h_final = model.ln_f(h)
            logits = model.lm_head(h_final)
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            token_entropies.extend(ent.view(-1).cpu().numpy().tolist())

    token_ent_mean = np.mean(token_entropies)
    token_ent_std = np.std(token_entropies)

    print(f"\n  Token-level entropy at layer {merge_layer}: {token_ent_mean:.4f} +/- {token_ent_std:.4f}")

    # Chord-level entropy at different thresholds
    for thresh in thresholds:
        chord_ents_this = []

        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.get_batch("val", config)
                B, T = x.shape

                h = model.forward_through_layer(x, merge_layer)
                boundaries, _ = detect_chord_boundaries(h, thresh)
                pooled, chord_lengths, chord_map, n_chords = pool_chords(h, boundaries)

                # Predict from pooled representations directly (no further layers)
                for b in range(B):
                    nc = n_chords[b].item()
                    if nc == 0:
                        continue
                    chord_h = pooled[b, :nc, :]  # [nc, C]
                    chord_h_final = model.ln_f(chord_h)
                    chord_logits = model.lm_head(chord_h_final)
                    chord_probs = F.softmax(chord_logits, dim=-1)
                    chord_ent = -(chord_probs * torch.log(chord_probs + 1e-10)).sum(dim=-1)
                    chord_ents_this.extend(chord_ent.cpu().numpy().tolist())

        chord_ent_mean = np.mean(chord_ents_this)
        chord_ent_std = np.std(chord_ents_this)
        avg_chords = len(chord_ents_this) / (n_batches * config.batch_size)
        delta = chord_ent_mean - token_ent_mean
        pct_change = 100 * delta / token_ent_mean if token_ent_mean > 0 else 0

        print(f"  Chord entropy (thresh={thresh:.2f}): {chord_ent_mean:.4f} +/- {chord_ent_std:.4f}  "
              f"(delta={delta:+.4f}, {pct_change:+.1f}%, ~{avg_chords:.0f} chords/seq)")

    return token_ent_mean


# =============================================================================
# TEST 4: Head-to-Head Comparison
# =============================================================================

def test_head_to_head(model, dataset, config, n_batches=30):
    """
    Three strategies compared:
    1. Full pipeline (baseline)
    2. Token-level early exit (Phase 10 approach)
    3. Chord flow (this phase)

    Measure accuracy and effective compute for each.
    """
    print("\n" + "=" * 60)
    print("  TEST 4: Head-to-Head Comparison")
    print("  Full pipeline vs token early exit vs chord flow")
    print("=" * 60)

    model.eval()

    # 1. Full pipeline
    full_correct = 0
    full_total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            logits, _ = model(x, y)
            preds = logits.argmax(dim=-1)
            full_correct += (preds == y).sum().item()
            full_total += y.numel()
    full_acc = full_correct / full_total

    # 2. Token early exit (best from Phase 10: layer 2, threshold 1.0)
    exit_layer = 2
    exit_thresh = 1.0
    early_correct = 0
    early_total = 0
    early_exited = 0

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            B, T = x.shape

            # Run through exit_layer
            h = model.forward_through_layer(x, exit_layer)

            # Compute entropy
            h_norm_exit = model.ln_f(h)
            logits_exit = model.lm_head(h_norm_exit)
            probs_exit = F.softmax(logits_exit, dim=-1)
            entropy = -(probs_exit * torch.log(probs_exit + 1e-10)).sum(dim=-1)

            exit_mask = entropy < exit_thresh

            # Full pipeline for non-exited
            h_full = h.clone()
            for i in range(exit_layer + 1, len(model.blocks)):
                h_full = model.blocks[i](h_full)
            h_full = model.ln_f(h_full)
            logits_full = model.lm_head(h_full)

            combined = torch.where(exit_mask.unsqueeze(-1).expand_as(logits_full),
                                   logits_exit, logits_full)
            preds = combined.argmax(dim=-1)
            early_correct += (preds == y).sum().item()
            early_total += y.numel()
            early_exited += exit_mask.sum().item()

    early_acc = early_correct / early_total
    early_pct_exit = 100 * early_exited / early_total
    # Compute: exited tokens skip (4 - 2 - 1) = 1 layer out of 4
    early_compute = 1 - (early_pct_exit / 100) * (1 / config.n_layer)

    # 3. Chord flow (merge at layer 2, threshold 0.80)
    merge_layer = 2
    chord_thresh = 0.80
    chord_correct = 0
    chord_total = 0
    total_chords = 0

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            B, T = x.shape
            chord_total += B * T

            h = model.forward_through_layer(x, merge_layer)
            boundaries, _ = detect_chord_boundaries(h, chord_thresh)
            pooled, chord_lengths, chord_map, n_chords = pool_chords(h, boundaries)
            total_chords += n_chords.sum().item()

            for b in range(B):
                nc = n_chords[b].item()
                if nc == 0:
                    continue

                chord_h = pooled[b:b+1, :nc, :]

                for layer_idx in range(merge_layer + 1, len(model.blocks)):
                    block = model.blocks[layer_idx]
                    residual = chord_h
                    chord_h_norm = block.ln_1(chord_h)
                    bsz, tgt_len, emb = chord_h_norm.shape
                    qkv = block.attn.c_attn(chord_h_norm)
                    q, k, v = qkv.split(block.attn.n_embd, dim=2)
                    head_dim = emb // block.attn.n_head
                    q = q.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                    k = k.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                    v = v.view(bsz, tgt_len, block.attn.n_head, head_dim).transpose(1, 2)
                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
                    causal = torch.tril(torch.ones(tgt_len, tgt_len, device=x.device))
                    att = att.masked_fill(causal.view(1, 1, tgt_len, tgt_len) == 0, float("-inf"))
                    att = F.softmax(att, dim=-1)
                    attn_out = att @ v
                    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, tgt_len, emb)
                    attn_out = block.attn.c_proj(attn_out)
                    chord_h = residual + attn_out
                    chord_h = chord_h + block.mlp(block.ln_2(chord_h))

                chord_h = model.ln_f(chord_h)
                chord_logits = model.lm_head(chord_h)

                for t in range(T):
                    chord_id = chord_map[b, t].item()
                    if chord_id < nc:
                        pred = chord_logits[0, chord_id].argmax().item()
                        if pred == y[b, t].item():
                            chord_correct += 1

    chord_acc = chord_correct / chord_total
    avg_chords_per_seq = total_chords / (n_batches * config.batch_size)
    compression = avg_chords_per_seq / config.block_size
    # Upper layers see compression^2 of original attention cost
    upper_fraction = (config.n_layer - merge_layer - 1) / config.n_layer
    chord_compute = 1 - upper_fraction * (1 - compression ** 2)

    print(f"\n  {'Strategy':<25} {'Accuracy':>10} {'vs baseline':>12} {'Retention':>10} {'Rel. compute':>13}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*13}")
    print(f"  {'Full pipeline':<25} {full_acc:>10.4f} {'---':>12} {'100.0%':>10} {'100.0%':>13}")
    print(f"  {'Token early exit':<25} {early_acc:>10.4f} {early_acc - full_acc:>+12.4f} "
          f"{100*early_acc/full_acc:>9.1f}% {100*early_compute:>12.1f}%")
    print(f"  {'Chord flow':<25} {chord_acc:>10.4f} {chord_acc - full_acc:>+12.4f} "
          f"{100*chord_acc/full_acc:>9.1f}% {100*chord_compute:>12.1f}%")

    print(f"\n  Chord flow details:")
    print(f"    Merge layer: {merge_layer}")
    print(f"    Similarity threshold: {chord_thresh}")
    print(f"    Avg chords per sequence: {avg_chords_per_seq:.1f} (from {config.block_size} tokens)")
    print(f"    Sequence compression: {compression:.2f}x")
    print(f"    Token early exit: {early_pct_exit:.1f}% exit, skip 1 layer each")

    return full_acc, early_acc, chord_acc


# =============================================================================
# TEST 5: What Do Chords Look Like?
# =============================================================================

def test_chord_examples(model, dataset, config, merge_layer=2):
    """
    Show actual examples of detected chords. Do they align with words?
    """
    print("\n" + "=" * 60)
    print("  TEST 5: Chord Examples")
    print("  Do detected chords align with words?")
    print("=" * 60)

    model.eval()
    thresholds = [0.70, 0.80, 0.90]

    # Get a batch and pick a few sequences
    x, y = dataset.get_batch("val", config)

    with torch.no_grad():
        h = model.forward_through_layer(x, merge_layer)

        for thresh in thresholds:
            print(f"\n  --- Threshold: {thresh:.2f} ---")
            boundaries, sims = detect_chord_boundaries(h, thresh)

            for b in range(min(3, x.shape[0])):
                chars = [dataset.itos[x[b, t].item()] for t in range(x.shape[1])]
                bounds = boundaries[b].cpu().numpy()

                # Build chord strings
                chords = []
                current_chord = ""
                for t in range(len(chars)):
                    if bounds[t] and current_chord:
                        chords.append(current_chord)
                        current_chord = ""
                    current_chord += chars[t]
                if current_chord:
                    chords.append(current_chord)

                # Show first 60 chars with chord boundaries marked
                text = "".join(chars[:80])
                # Replace newlines for display
                text_display = text.replace('\n', '\\n')

                n_chords = len([c for c in chords if len("".join(chars[:80])) > 0])
                avg_len = np.mean([len(c) for c in chords[:40]])

                print(f"\n    Sample {b+1}: {len(chords)} chords, avg length {avg_len:.1f}")
                # Show chords with | separator (first 20 chords)
                chord_display = "|".join(chords[:25])
                # Replace newlines for display
                chord_display = chord_display.replace('\n', '\\n')
                if len(chord_display) > 100:
                    chord_display = chord_display[:100] + "..."
                print(f"    [{chord_display}]")

    # Check word alignment
    print(f"\n  --- Word Alignment Analysis ---")
    thresh = 0.80

    with torch.no_grad():
        h = model.forward_through_layer(x, merge_layer)
        boundaries, _ = detect_chord_boundaries(h, thresh)

        word_boundary_hits = 0
        word_boundary_total = 0
        false_splits = 0
        total_splits = 0

        for b in range(min(32, x.shape[0])):
            chars = [dataset.itos[x[b, t].item()] for t in range(x.shape[1])]
            bounds = boundaries[b].cpu().numpy()

            for t in range(1, len(chars)):
                is_word_boundary = (chars[t-1] in ' \n' or chars[t] in ' \n')
                is_chord_boundary = bounds[t]

                if is_word_boundary:
                    word_boundary_total += 1
                    if is_chord_boundary:
                        word_boundary_hits += 1

                if is_chord_boundary and not is_word_boundary:
                    false_splits += 1
                if is_chord_boundary:
                    total_splits += 1

    recall = word_boundary_hits / word_boundary_total if word_boundary_total > 0 else 0
    precision = (total_splits - false_splits) / total_splits if total_splits > 0 else 0

    print(f"  At threshold {thresh}:")
    print(f"    Word boundaries detected: {word_boundary_hits}/{word_boundary_total} ({100*recall:.1f}% recall)")
    print(f"    False splits (mid-word): {false_splits}/{total_splits} ({100*false_splits/total_splits:.1f}% of all splits)")
    print(f"    Precision (split = real boundary): {100*precision:.1f}%")


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict(full_acc, early_acc, chord_acc, best_layer, best_gap):
    print("\n" + "=" * 60)
    print("  VERDICT: Is Chord Flow Viable?")
    print("=" * 60)

    chord_retention = 100 * chord_acc / full_acc if full_acc > 0 else 0
    early_retention = 100 * early_acc / full_acc if full_acc > 0 else 0

    if best_gap < 0.01:
        print(f"\n  NO. Chord boundaries are not detectable at layer 2.")
        print(f"  Within-word vs cross-boundary similarity gap: {best_gap:+.4f}")
        print(f"  The model doesn't form distinct spatial clusters for words")
        print(f"  at the representation level. Characters in 'king' are not")
        print(f"  more similar to each other than to characters in other words.")
        print(f"  Phase 7's chord composition works via attention mixing,")
        print(f"  not via representation proximity.")
    elif chord_retention >= 95:
        print(f"\n  YES. Chord flow is viable.")
        print(f"    Chord accuracy: {chord_acc:.4f} ({chord_retention:.1f}% of baseline)")
        print(f"    vs token early exit: {early_acc:.4f} ({early_retention:.1f}% of baseline)")
        print(f"    Chord boundary gap: {best_gap:+.4f} (at layer {best_layer})")
    elif chord_retention >= 80:
        print(f"\n  PARTIALLY. Chord flow works but loses accuracy.")
        print(f"    Chord accuracy: {chord_acc:.4f} ({chord_retention:.1f}% of baseline)")
        print(f"    vs token early exit: {early_acc:.4f} ({early_retention:.1f}% of baseline)")
        print(f"    Chord boundary gap: {best_gap:+.4f} (at layer {best_layer})")
        print(f"\n  The model forms detectable chords, but pooling them loses")
        print(f"  information that the upper layers need. Each position in a")
        print(f"  word carries position-specific information that mean-pooling")
        print(f"  destroys.")
    else:
        print(f"\n  NOT YET. Chord flow loses too much accuracy.")
        print(f"    Chord accuracy: {chord_acc:.4f} ({chord_retention:.1f}% of baseline)")
        print(f"    vs token early exit: {early_acc:.4f} ({early_retention:.1f}% of baseline)")
        print(f"    Chord boundary gap: {best_gap:+.4f} (at layer {best_layer})")
        print(f"\n  Possible reasons:")
        print(f"    - Mean pooling destroys positional information within chords")
        print(f"    - The model needs per-character resolution for next-char prediction")
        print(f"    - Chords are compositional (attention-mixed) not spatial (proximity-based)")
        if best_gap > 0.02:
            print(f"\n  However, chord DETECTION works (gap = {best_gap:+.4f}).")
            print(f"  The issue is pooling, not detection. A learned chord encoder")
            print(f"  (instead of mean pooling) might preserve the needed information.")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  CHORD FLOW TEST")
    print("  Can chords replace tokens in the upper layers?")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    model = train_model(config, dataset)

    best_layer, best_gap, _, _ = test_chord_detection(model, dataset, config)
    test_chord_pooling(model, dataset, config, merge_layer=best_layer)
    test_chord_entropy(model, dataset, config, merge_layer=best_layer)
    full_acc, early_acc, chord_acc = test_head_to_head(model, dataset, config)
    test_chord_examples(model, dataset, config, merge_layer=best_layer)

    print_verdict(full_acc, early_acc, chord_acc, best_layer, best_gap)


if __name__ == "__main__":
    main()
