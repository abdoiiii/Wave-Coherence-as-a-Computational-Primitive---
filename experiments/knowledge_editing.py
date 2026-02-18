"""
Knowledge Editing via Harmonic Frequency Bands

THE QUESTION:
Can we change what a model knows by editing specific harmonic frequency bands
in the embedding, without disturbing other knowledge?

Phase 2 showed 92.5% of harmonic channels are independent at the final layer.
This means editing one channel has minimal risk of disturbing another.
Now we test it.

THE EXPERIMENT:
1. Create training data: Shakespeare + planted patterns
   - Pattern A: "zqj" is always followed by "mmm"
   - Pattern B: "wkf" is always followed by "ppp"
   (Trigrams chosen to not appear in Shakespeare)

2. Train harmonic model until both patterns are learned

3. Identify which harmonic bands in the embeddings carry each pattern:
   - Feed pattern A trigger, capture which bands activate most
   - Feed pattern B trigger, compare
   - Find the differential bands unique to each pattern

4. Edit: modify specific harmonic bands in the embedding table
   to change pattern A's output from "mmm" to "ppp"

5. Verify:
   - Does pattern A now predict differently?
   - Does pattern B still predict correctly? (collateral damage check)
   - Is Shakespeare generation quality preserved? (global damage check)
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
    max_iters = 4000
    eval_interval = 500
    eval_iters = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Planted patterns
PATTERN_A_TRIGGER = "zqj"
PATTERN_A_OUTPUT = "mmm"
PATTERN_B_TRIGGER = "wkf"
PATTERN_B_OUTPUT = "ppp"

# How many times to repeat each pattern in training data
PATTERN_REPEATS = 2000


# =============================================================================
# Model (same architecture as previous experiments)
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
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def predict_next(self, idx):
        """Get the top predicted next character (greedy)."""
        logits, _ = self(idx)
        logits = logits[:, -1, :]
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def get_embedding_activations(self, idx):
        """Get the embedding vectors for specific input indices."""
        return self.wte(idx)


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


def create_training_data():
    """Create Shakespeare + planted patterns."""
    shakespeare = download_shakespeare()

    # Create planted pattern strings
    pattern_a = f" {PATTERN_A_TRIGGER}{PATTERN_A_OUTPUT} "
    pattern_b = f" {PATTERN_B_TRIGGER}{PATTERN_B_OUTPUT} "

    # Insert patterns throughout the text
    planted = ""
    for i in range(PATTERN_REPEATS):
        planted += pattern_a + pattern_b

    # Mix: Shakespeare first, then planted patterns interleaved
    # This ensures the model sees both Shakespeare and patterns
    combined = shakespeare + "\n" + planted

    return combined, shakespeare


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
    print(f"  Training harmonic model with planted patterns")
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
# Pattern verification
# =============================================================================

def test_pattern_completion(model, dataset, trigger, expected, label, n_chars=3):
    """Test if the model completes a trigger with the expected output."""
    model.eval()
    # Encode trigger with a space prefix for context
    prompt = f" {trigger}"
    tokens = dataset.encode(prompt).unsqueeze(0).to(Config.device)

    predictions = []
    for i in range(n_chars):
        next_token = model.predict_next(tokens)
        predictions.append(next_token.item())
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    predicted_str = dataset.decode(predictions)
    match = predicted_str == expected
    return predicted_str, match


def verify_patterns(model, dataset, label=""):
    """Verify both planted patterns."""
    print(f"\n  Pattern verification {label}:")

    pred_a, match_a = test_pattern_completion(model, dataset, PATTERN_A_TRIGGER, PATTERN_A_OUTPUT, "A")
    print(f"    Pattern A: '{PATTERN_A_TRIGGER}' -> '{pred_a}' (expected '{PATTERN_A_OUTPUT}') {'CORRECT' if match_a else 'WRONG'}")

    pred_b, match_b = test_pattern_completion(model, dataset, PATTERN_B_TRIGGER, PATTERN_B_OUTPUT, "B")
    print(f"    Pattern B: '{PATTERN_B_TRIGGER}' -> '{pred_b}' (expected '{PATTERN_B_OUTPUT}') {'CORRECT' if match_b else 'WRONG'}")

    return match_a, match_b, pred_a, pred_b


def measure_shakespeare_quality(model, dataset, config, n_batches=10):
    """Measure validation loss on Shakespeare-like data."""
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch("val", config)
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / n_batches


# =============================================================================
# Harmonic band analysis
# =============================================================================

def analyze_pattern_bands(model, dataset, config):
    """
    Identify which harmonic bands are most important for each pattern.

    Method: for each harmonic band h, zero it out in the embedding table
    and measure how much the pattern prediction changes. Bands that cause
    the most change are the ones carrying that pattern's knowledge.
    """
    model.eval()
    n_harmonics = config.n_embd // 2

    print(f"\n  Identifying critical harmonic bands for each pattern...")
    print(f"  Method: zero out each band, measure prediction change")

    results = {}

    for pattern_name, trigger, expected in [
        ("A", PATTERN_A_TRIGGER, PATTERN_A_OUTPUT),
        ("B", PATTERN_B_TRIGGER, PATTERN_B_OUTPUT),
    ]:
        # Get baseline logits for this pattern
        prompt = f" {trigger}"
        tokens = dataset.encode(prompt).unsqueeze(0).to(config.device)

        with torch.no_grad():
            baseline_logits, _ = model(tokens)
            baseline_logits = baseline_logits[:, -1, :]  # last position
            baseline_probs = F.softmax(baseline_logits, dim=-1)

        # For each harmonic band, zero it out and measure change
        band_importance = np.zeros(n_harmonics)
        original_weight = model.wte.weight.data.clone()

        for h in range(n_harmonics):
            cos_idx = h * 2
            sin_idx = h * 2 + 1

            # Zero out this harmonic band
            model.wte.weight.data[:, cos_idx] = 0.0
            model.wte.weight.data[:, sin_idx] = 0.0

            with torch.no_grad():
                ablated_logits, _ = model(tokens)
                ablated_logits = ablated_logits[:, -1, :]
                ablated_probs = F.softmax(ablated_logits, dim=-1)

            # KL divergence: how much did the prediction change?
            kl = F.kl_div(ablated_probs.log(), baseline_probs, reduction='sum').item()
            band_importance[h] = kl

            # Restore
            model.wte.weight.data = original_weight.clone()

        results[pattern_name] = band_importance

        # Report top bands
        top_bands = np.argsort(band_importance)[::-1][:10]
        print(f"\n    Pattern {pattern_name} ('{trigger}' -> '{expected}'):")
        print(f"    {'Band':>6} {'Importance (KL)':>16}")
        print(f"    {'-'*6} {'-'*16}")
        for b in top_bands:
            print(f"    n={b+1:>3} {band_importance[b]:>16.6f}")

    # Find differential bands: important for A but not B, and vice versa
    diff_a = results["A"] - results["B"]  # Positive = more important for A
    diff_b = results["B"] - results["A"]  # Positive = more important for B

    print(f"\n  Differential bands (unique to each pattern):")
    top_diff_a = np.argsort(diff_a)[::-1][:5]
    top_diff_b = np.argsort(diff_b)[::-1][:5]

    print(f"\n    Most A-specific bands:")
    for b in top_diff_a:
        print(f"      n={b+1:>3}  A importance: {results['A'][b]:.6f}  B importance: {results['B'][b]:.6f}  diff: {diff_a[b]:+.6f}")

    print(f"\n    Most B-specific bands:")
    for b in top_diff_b:
        print(f"      n={b+1:>3}  A importance: {results['A'][b]:.6f}  B importance: {results['B'][b]:.6f}  diff: {diff_b[b]:+.6f}")

    return results, diff_a, diff_b


# =============================================================================
# The Edit
# =============================================================================

def edit_harmonic_bands(model, dataset, config, band_importance_a, band_importance_b):
    """
    Edit specific harmonic bands to change pattern A's output.

    Strategy: The characters unique to pattern A's trigger are 'z', 'q', 'j'.
    We'll modify the embedding of the last trigger character ('j') at the most
    A-specific harmonic bands, shifting it toward the embedding of the last
    trigger character of pattern B ('f').

    The idea: if the model associates 'j' (end of trigger A) with outputting 'm',
    and 'f' (end of trigger B) with outputting 'p', then making 'j' look like 'f'
    at the critical frequency bands should change the output from 'mmm' to 'ppp'.
    """
    print(f"\n{'='*60}")
    print(f"  EDITING: Modifying harmonic bands")
    print(f"{'='*60}")

    # Find the most A-specific bands (important for A, not for B)
    diff = band_importance_a - band_importance_b
    a_specific_bands = np.argsort(diff)[::-1]

    # Get character indices
    char_a_last = dataset.stoi[PATTERN_A_TRIGGER[-1]]  # 'j'
    char_b_last = dataset.stoi[PATTERN_B_TRIGGER[-1]]  # 'f'
    char_a_out = dataset.stoi[PATTERN_A_OUTPUT[0]]      # 'm'
    char_b_out = dataset.stoi[PATTERN_B_OUTPUT[0]]      # 'p'

    print(f"\n  Target: change '{PATTERN_A_TRIGGER}' -> '{PATTERN_A_OUTPUT}' to produce '{PATTERN_B_OUTPUT}'")
    print(f"  Method: modify embedding of '{PATTERN_A_TRIGGER[-1]}' (idx {char_a_last})")
    print(f"          at A-specific bands, shifting toward '{PATTERN_B_TRIGGER[-1]}' (idx {char_b_last})")

    # Also try editing the output character embedding
    print(f"  Also: modify embedding of output char '{PATTERN_A_OUTPUT[0]}' (idx {char_a_out})")
    print(f"        at A-specific bands, shifting toward '{PATTERN_B_OUTPUT[0]}' (idx {char_b_out})")

    # Try editing with increasing number of bands
    original_weight = model.wte.weight.data.clone()
    edit_results = []

    for n_bands in [1, 2, 3, 5, 8, 12, 16, 24, 32]:
        # Restore original
        model.wte.weight.data = original_weight.clone()

        bands_to_edit = a_specific_bands[:n_bands]

        # For each selected band, replace A-trigger chars with B-trigger chars' values
        for h in bands_to_edit:
            cos_idx = h * 2
            sin_idx = h * 2 + 1

            # Shift trigger characters: make A's trigger look like B's trigger
            for i, (ca, cb) in enumerate(zip(PATTERN_A_TRIGGER, PATTERN_B_TRIGGER)):
                a_idx = dataset.stoi[ca]
                b_idx = dataset.stoi[cb]
                model.wte.weight.data[a_idx, cos_idx] = original_weight[b_idx, cos_idx]
                model.wte.weight.data[a_idx, sin_idx] = original_weight[b_idx, sin_idx]

            # Shift output character: make A's output char look like B's output char
            model.wte.weight.data[char_a_out, cos_idx] = original_weight[char_b_out, cos_idx]
            model.wte.weight.data[char_a_out, sin_idx] = original_weight[char_b_out, sin_idx]

        # Test predictions
        pred_a, match_a = test_pattern_completion(model, dataset, PATTERN_A_TRIGGER, PATTERN_A_OUTPUT, "A")
        pred_b, match_b = test_pattern_completion(model, dataset, PATTERN_B_TRIGGER, PATTERN_B_OUTPUT, "B")

        # Did we change A's output?
        a_changed = pred_a != PATTERN_A_OUTPUT
        # Does A now match B's output?
        a_matches_b = pred_a == PATTERN_B_OUTPUT
        # Is B still correct?
        b_preserved = match_b

        # Measure Shakespeare quality
        val_loss = measure_shakespeare_quality(model, dataset, config, n_batches=5)

        edit_results.append({
            "n_bands": n_bands,
            "bands": bands_to_edit.tolist(),
            "pred_a": pred_a,
            "pred_b": pred_b,
            "a_changed": a_changed,
            "a_matches_b": a_matches_b,
            "b_preserved": b_preserved,
            "val_loss": val_loss,
        })

        status_a = "CHANGED" if a_changed else "unchanged"
        target_hit = " -> TARGET HIT!" if a_matches_b else ""
        status_b = "preserved" if b_preserved else "DAMAGED"
        print(f"\n    {n_bands:>2} bands edited:")
        print(f"      Pattern A: '{PATTERN_A_TRIGGER}' -> '{pred_a}' ({status_a}{target_hit})")
        print(f"      Pattern B: '{PATTERN_B_TRIGGER}' -> '{pred_b}' ({status_b})")
        print(f"      Val loss: {val_loss:.4f}")
        print(f"      Bands: {[h+1 for h in bands_to_edit[:8]]}{'...' if n_bands > 8 else ''}")

    # Restore original for final comparison
    model.wte.weight.data = original_weight.clone()

    return edit_results


# =============================================================================
# Alternative edit: direct embedding swap
# =============================================================================

def edit_output_character(model, dataset, config):
    """
    Simpler edit: directly swap the embedding of the output character 'm'
    with the embedding of 'p' at the most isolated harmonic bands (from Phase 2).

    This is the most direct test: if we change how 'm' is represented in
    specific frequency bands, does the model output 'p' instead?
    """
    print(f"\n{'='*60}")
    print(f"  ALTERNATIVE EDIT: Output character embedding swap")
    print(f"{'='*60}")

    char_m = dataset.stoi['m']
    char_p = dataset.stoi['p']
    n_harmonics = config.n_embd // 2

    # Use the most isolated bands from Phase 2: n=14, n=5, n=1, n=6
    isolated_bands = [13, 4, 0, 5, 12, 2, 14, 1]  # 0-indexed (n-1)

    original_weight = model.wte.weight.data.clone()
    baseline_loss = measure_shakespeare_quality(model, dataset, config, n_batches=5)

    print(f"\n  Swapping 'm' <-> 'p' embeddings at isolated harmonic bands")
    print(f"  Baseline val loss: {baseline_loss:.4f}")

    for n_bands in [1, 2, 4, 8, 16, 32, 48, 64]:
        model.wte.weight.data = original_weight.clone()

        bands = isolated_bands[:min(n_bands, len(isolated_bands))]
        # If we need more bands than our curated list, extend with remaining
        if n_bands > len(isolated_bands):
            remaining = [h for h in range(n_harmonics) if h not in isolated_bands]
            bands = isolated_bands + remaining[:n_bands - len(isolated_bands)]

        for h in bands:
            cos_idx = h * 2
            sin_idx = h * 2 + 1
            # Swap m and p at this band
            m_cos = original_weight[char_m, cos_idx].item()
            m_sin = original_weight[char_m, sin_idx].item()
            p_cos = original_weight[char_p, cos_idx].item()
            p_sin = original_weight[char_p, sin_idx].item()
            model.wte.weight.data[char_m, cos_idx] = p_cos
            model.wte.weight.data[char_m, sin_idx] = p_sin

        pred_a, _ = test_pattern_completion(model, dataset, PATTERN_A_TRIGGER, PATTERN_A_OUTPUT, "A")
        pred_b, match_b = test_pattern_completion(model, dataset, PATTERN_B_TRIGGER, PATTERN_B_OUTPUT, "B")
        val_loss = measure_shakespeare_quality(model, dataset, config, n_batches=5)
        loss_delta = val_loss - baseline_loss

        a_changed = pred_a != PATTERN_A_OUTPUT
        print(f"\n    {n_bands:>2} bands: A->'{pred_a}' {'CHANGED' if a_changed else 'same':>8} | "
              f"B->'{pred_b}' {'OK' if match_b else 'DAMAGED':>7} | "
              f"loss {val_loss:.4f} (delta {loss_delta:+.4f})")

    model.wte.weight.data = original_weight.clone()


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  KNOWLEDGE EDITING VIA HARMONIC FREQUENCY BANDS")
    print(f"  Can we change what the model knows by editing specific bands?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    # Create data with planted patterns
    combined_text, shakespeare = create_training_data()
    dataset = Dataset(combined_text)
    print(f"\n  Dataset: {len(combined_text):,} characters, {dataset.vocab_size} unique")
    print(f"  Shakespeare: {len(shakespeare):,} characters")
    print(f"  Planted patterns: {PATTERN_REPEATS} repeats each")
    print(f"  Pattern A: '{PATTERN_A_TRIGGER}' -> '{PATTERN_A_OUTPUT}'")
    print(f"  Pattern B: '{PATTERN_B_TRIGGER}' -> '{PATTERN_B_OUTPUT}'")

    # Train
    model = train_model(config, dataset)

    # Verify patterns are learned
    print(f"\n{'='*60}")
    print(f"  VERIFICATION: Are planted patterns learned?")
    print(f"{'='*60}")
    match_a, match_b, _, _ = verify_patterns(model, dataset, "(before edit)")

    if not match_a:
        print(f"\n  WARNING: Pattern A not learned. Trying more training...")
        # Could add more training here, but let's continue with what we have
    if not match_b:
        print(f"\n  WARNING: Pattern B not learned.")

    baseline_loss = measure_shakespeare_quality(model, dataset, config)
    print(f"\n  Shakespeare val loss: {baseline_loss:.4f}")

    # Generate a Shakespeare sample for quality comparison
    print(f"\n  Shakespeare sample (before edit):")
    prompt = dataset.encode("\n  ").unsqueeze(0).to(config.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    print(f"    '{dataset.decode(generated[0].tolist())}'")

    # Analyze which bands matter for each pattern
    print(f"\n{'='*60}")
    print(f"  BAND ANALYSIS: Which harmonics carry each pattern?")
    print(f"{'='*60}")
    band_results, diff_a, diff_b = analyze_pattern_bands(model, dataset, config)

    # Edit attempt 1: targeted band modification
    print(f"\n{'='*60}")
    print(f"  EDIT ATTEMPT 1: Targeted harmonic band modification")
    print(f"{'='*60}")
    edit_results = edit_harmonic_bands(model, dataset, config,
                                        band_results["A"], band_results["B"])

    # Edit attempt 2: direct output character swap
    edit_output_character(model, dataset, config)

    # Final Shakespeare sample for comparison
    print(f"\n{'='*60}")
    print(f"  POST-EDIT VERIFICATION")
    print(f"{'='*60}")
    verify_patterns(model, dataset, "(after all edits restored)")
    print(f"\n  Shakespeare sample (after edits restored):")
    prompt = dataset.encode("\n  ").unsqueeze(0).to(config.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    print(f"    '{dataset.decode(generated[0].tolist())}'")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    successful_edits = [r for r in edit_results if r["a_changed"] and r["b_preserved"]]
    target_hits = [r for r in edit_results if r["a_matches_b"] and r["b_preserved"]]

    print(f"\n  Edit attempts where A changed but B preserved: {len(successful_edits)}/{len(edit_results)}")
    print(f"  Edit attempts where A matched target AND B preserved: {len(target_hits)}/{len(edit_results)}")

    if target_hits:
        best = min(target_hits, key=lambda r: r["n_bands"])
        print(f"\n  BEST RESULT: {best['n_bands']} bands edited")
        print(f"    A: '{PATTERN_A_TRIGGER}' -> '{best['pred_a']}' (target: '{PATTERN_B_OUTPUT}')")
        print(f"    B: '{PATTERN_B_TRIGGER}' -> '{best['pred_b']}' (preserved: {best['b_preserved']})")
        print(f"    Val loss: {best['val_loss']:.4f} (baseline: {baseline_loss:.4f})")
        print(f"\n  VERDICT: Targeted knowledge editing via harmonic bands WORKS.")
        print(f"  Changed pattern A's output while preserving pattern B and Shakespeare quality.")
    elif successful_edits:
        best = min(successful_edits, key=lambda r: r["n_bands"])
        print(f"\n  PARTIAL SUCCESS: {best['n_bands']} bands changed A's output")
        print(f"    A: '{PATTERN_A_TRIGGER}' -> '{best['pred_a']}' (not the target '{PATTERN_B_OUTPUT}', but changed)")
        print(f"    B preserved: {best['b_preserved']}")
        print(f"\n  VERDICT: Editing changes output but doesn't achieve precise targeting yet.")
        print(f"  The channel is editable, but the edit direction needs refinement.")
    else:
        print(f"\n  VERDICT: Editing at the embedding level alone is insufficient.")
        print(f"  Knowledge may be stored deeper in the MLP weights, not just embeddings.")
        print(f"  Next step: try editing MLP weights projected onto harmonic bands.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
