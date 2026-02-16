"""
Harmonic Transformer -- Character-Level Language Model Without Tokens

A minimal transformer that uses harmonic phase encoding instead of
learned embeddings and operates on raw characters instead of tokens.

Three modes:
  1. baseline    -- random Gaussian embeddings, trainable (standard approach)
  2. harmonic    -- harmonic phase embeddings, trainable (structured init)
  3. frozen      -- harmonic phase embeddings, NOT trainable (pure geometry)

Each character is assigned a phase angle on the unit circle:
  theta_c = c * 2 * pi / vocab_size

Its embedding vector is the harmonic expansion:
  [cos(theta), sin(theta), cos(2*theta), sin(2*theta), ..., cos(N*theta), sin(N*theta)]

This gives each character a unique, structured position in embedding space
where relationships between characters are encoded geometrically.

No tokenizer. No BPE. No subword vocabulary. Just circles and harmonics.
"""

import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Model
    n_layer = 4
    n_head = 4
    n_embd = 128        # embedding dimension (must be even for cos/sin pairs)
    block_size = 256     # context window (characters, not tokens)
    dropout = 0.1

    # Training
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Harmonic Embedding Layer
# =============================================================================

class HarmonicEmbedding(nn.Module):
    """
    Replaces nn.Embedding with harmonic phase encoding.

    Each input index c gets mapped to:
      theta_c = c * 2 * pi / num_embeddings
      embed = [cos(1*theta), sin(1*theta), cos(2*theta), sin(2*theta), ...]

    The result is a deterministic, structured embedding where every
    dimension has a defined geometric meaning.
    """

    def __init__(self, num_embeddings, embedding_dim, trainable=True):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even for cos/sin pairs"

        n_harmonics = embedding_dim // 2

        # Build the embedding table
        angles = torch.arange(num_embeddings, dtype=torch.float32) * (2 * math.pi / num_embeddings)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)

        # Outer product: [num_embeddings, n_harmonics]
        phase_matrix = angles.unsqueeze(1) * harmonics.unsqueeze(0)

        # Interleave cos and sin: [num_embeddings, embedding_dim]
        embedding = torch.zeros(num_embeddings, embedding_dim)
        embedding[:, 0::2] = torch.cos(phase_matrix)
        embedding[:, 1::2] = torch.sin(phase_matrix)

        # Scale to match typical embedding magnitude
        embedding = embedding * (1.0 / math.sqrt(n_harmonics))

        if trainable:
            self.weight = nn.Parameter(embedding)
        else:
            self.register_buffer("weight", embedding)

    def forward(self, x):
        return F.embedding(x, self.weight)


# =============================================================================
# Harmonic Positional Encoding
# =============================================================================

class HarmonicPositionalEncoding(nn.Module):
    """
    Positional encoding using the same harmonic principle.
    Position p gets phase angle theta_p = p * 2 * pi / max_len.
    """

    def __init__(self, max_len, embedding_dim, trainable=True):
        super().__init__()
        assert embedding_dim % 2 == 0

        n_harmonics = embedding_dim // 2

        positions = torch.arange(max_len, dtype=torch.float32)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)

        # Use different frequency scaling for position (log-spaced like sinusoidal PE)
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


# =============================================================================
# Transformer Components
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
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
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


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


# =============================================================================
# The Model
# =============================================================================

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
        elif mode == "harmonic":
            self.wte = HarmonicEmbedding(vocab_size, config.n_embd, trainable=True)
            self.wpe = HarmonicPositionalEncoding(config.block_size, config.n_embd, trainable=True)
        elif mode == "frozen":
            self.wte = HarmonicEmbedding(vocab_size, config.n_embd, trainable=False)
            self.wpe = HarmonicPositionalEncoding(config.block_size, config.n_embd, trainable=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying (only for baseline where embeddings are nn.Embedding)
        if mode == "baseline":
            self.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

        # Scale residual projections
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

        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_chars, temperature=0.8, top_k=None):
        for _ in range(max_new_chars):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =============================================================================
# Data Loading
# =============================================================================

def get_shakespeare_data():
    """Download Shakespeare text if needed."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(filepath):
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)
        print("Done.")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def prepare_data(text, config):
    """Prepare character-level data."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # 90/10 train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode


def get_batch(data, config):
    """Get a random batch of training data."""
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)


# =============================================================================
# Training
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    model.eval()
    out = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out


def train_model(mode, train_data, val_data, vocab_size, config):
    """Train a single model and return loss history."""
    print(f"\n{'=' * 60}")
    print(f"  Training: {mode.upper()}")
    print(f"{'=' * 60}")

    model = HarmonicGPT(config, vocab_size, mode=mode).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_history = []
    start_time = time.time()

    for iter_num in range(config.max_iters):
        # Evaluate periodically
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - start_time
            print(f"  step {iter_num:>5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
            loss_history.append({
                "step": iter_num,
                "train": losses["train"].item(),
                "val": losses["val"].item(),
            })

        # Training step
        X, Y = get_batch(train_data, config)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"  Training complete in {total_time:.1f}s")

    return model, loss_history


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print("=" * 60)
    print("  Harmonic Transformer -- Character-Level Language Model")
    print("  No tokens. No BPE. Just characters and harmonics.")
    print(f"  Device: {config.device}")
    print("=" * 60)

    # Load data
    text = get_shakespeare_data()
    train_data, val_data, vocab_size, encode, decode = prepare_data(text, config)
    print(f"\n  Dataset: {len(text):,} characters, {vocab_size} unique")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")
    print(f"  Model: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    print(f"  Context: {config.block_size} characters")

    # Train all three modes
    results = {}
    models = {}

    for mode in ["baseline", "harmonic", "frozen"]:
        model, history = train_model(mode, train_data, val_data, vocab_size, config)
        results[mode] = history
        models[mode] = model

    # ==========================================================================
    # Comparison
    # ==========================================================================
    print(f"\n{'=' * 60}")
    print("  COMPARISON: Final Validation Loss")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Mode':<12} {'Val Loss':>10} {'Train Loss':>12}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 12}")

    for mode in ["baseline", "harmonic", "frozen"]:
        final = results[mode][-1]
        print(f"  {mode:<12} {final['val']:>10.4f} {final['train']:>12.4f}")

    # ==========================================================================
    # Convergence comparison
    # ==========================================================================
    print(f"\n{'=' * 60}")
    print("  CONVERGENCE: Loss at Each Checkpoint")
    print(f"{'=' * 60}")
    print()
    print(f"  {'Step':>6}", end="")
    for mode in ["baseline", "harmonic", "frozen"]:
        print(f"  {mode:>12}", end="")
    print()
    print(f"  {'----':>6}", end="")
    for _ in ["baseline", "harmonic", "frozen"]:
        print(f"  {'----------':>12}", end="")
    print()

    max_entries = max(len(results[m]) for m in results)
    for i in range(max_entries):
        step = results["baseline"][i]["step"] if i < len(results["baseline"]) else "?"
        print(f"  {step:>6}", end="")
        for mode in ["baseline", "harmonic", "frozen"]:
            if i < len(results[mode]):
                print(f"  {results[mode][i]['val']:>12.4f}", end="")
            else:
                print(f"  {'':>12}", end="")
        print()

    # ==========================================================================
    # Sample generation
    # ==========================================================================
    print(f"\n{'=' * 60}")
    print("  SAMPLE GENERATION (500 characters each)")
    print(f"{'=' * 60}")

    seed = encode("\n")
    seed_tensor = torch.tensor([seed], dtype=torch.long, device=config.device)

    for mode in ["baseline", "harmonic", "frozen"]:
        print(f"\n  --- {mode.upper()} ---")
        model = models[mode]
        model.eval()
        generated = model.generate(seed_tensor.clone(), max_new_chars=500, temperature=0.8, top_k=40)
        text_out = decode(generated[0].tolist())
        # Indent generated text
        for line in text_out.split("\n"):
            print(f"  {line}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    baseline_final = results["baseline"][-1]["val"]
    harmonic_final = results["harmonic"][-1]["val"]
    frozen_final = results["frozen"][-1]["val"]

    print()
    print(f"  Baseline (random init, trainable):  {baseline_final:.4f}")
    print(f"  Harmonic (structured init, trainable): {harmonic_final:.4f}")
    print(f"  Frozen   (structured, NOT trainable):  {frozen_final:.4f}")
    print()

    if harmonic_final < baseline_final:
        pct = (1 - harmonic_final / baseline_final) * 100
        print(f"  Harmonic embeddings OUTPERFORM baseline by {pct:.1f}% on val loss.")
    elif harmonic_final > baseline_final:
        pct = (harmonic_final / baseline_final - 1) * 100
        print(f"  Harmonic embeddings underperform baseline by {pct:.1f}% on val loss.")
    else:
        print(f"  Harmonic embeddings MATCH baseline on val loss.")

    if frozen_final < baseline_final * 1.1:
        print(f"  Frozen harmonic embeddings within 10% of baseline --")
        print(f"  geometric structure alone carries most of the signal.")
    elif frozen_final < baseline_final * 1.5:
        print(f"  Frozen harmonic embeddings within 50% of baseline --")
        print(f"  geometry provides a useful starting point.")
    else:
        print(f"  Frozen harmonic embeddings significantly worse --")
        print(f"  pure geometry insufficient without adaptation.")

    # Convergence speed
    # Check which mode first reaches baseline's step-1000 loss
    if len(results["baseline"]) >= 3:
        baseline_early = results["baseline"][2]["val"]  # step 1000
        harmonic_reached = None
        for entry in results["harmonic"]:
            if entry["val"] <= baseline_early:
                harmonic_reached = entry["step"]
                break
        if harmonic_reached is not None and harmonic_reached < 1000:
            print(f"  Harmonic reached baseline's step-1000 loss ({baseline_early:.4f}) at step {harmonic_reached} --")
            print(f"  {1000 / harmonic_reached:.1f}x faster convergence.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
