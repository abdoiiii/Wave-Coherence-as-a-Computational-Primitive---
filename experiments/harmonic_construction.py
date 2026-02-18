"""
Harmonic Construction — Can We Build New Identity From Scratch?

Previous result: 80.7% swap rate proved identity follows geometry.
But swapping copies existing characters. The real question:

Can we CONSTRUCT a harmonic vector from scratch — one that doesn't
copy any existing character — and have the model process it predictably?

THREE TESTS:

1. INTERPOLATION: Blend two characters' embeddings (alpha * A + (1-alpha) * B).
   If the model processes geometry, the output probability distribution should
   smoothly interpolate between A-like and B-like behavior.

2. FRACTIONAL POSITION: Use the harmonic formula with a non-integer character
   index (e.g., c=10.5). This creates a genuinely new harmonic signature that
   doesn't match any existing character. Does the model process it coherently?

3. CHIMERA: Take specific harmonic bands from different characters. Mix
   'e'-bands with 't'-bands. Does the model show blended behavior from
   both source characters?

For all tests, we inject the constructed vector into a rare character slot
and measure how the model processes it IN CONTEXT (not as output).
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
# Model (identical to all experiments)
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

    @torch.no_grad()
    def get_full_logits(self, idx):
        logits, _ = self(idx)
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


# =============================================================================
# Helper: measure context effect
# =============================================================================

def measure_context_effect(model, dataset, config, context_text, char_slot_idx):
    """
    Feed a context ending with char_slot_idx. Return the output probability
    distribution. This lets us measure how the model treats a character
    IN CONTEXT — what does it predict AFTER seeing this character?
    """
    model.eval()
    tokens = dataset.encode(context_text).unsqueeze(0).to(config.device)
    # Append the character we're testing
    test_token = torch.tensor([[char_slot_idx]], device=config.device)
    full_input = torch.cat([tokens, test_token], dim=1)

    with torch.no_grad():
        logits = model.get_full_logits(full_input)
        # Get prediction at the position AFTER our injected character
        probs = F.softmax(logits[0, -1, :], dim=-1)

    return probs.cpu()


# =============================================================================
# TEST 1: Interpolation
# =============================================================================

def test_interpolation(model, dataset, config):
    """
    Blend two characters' embeddings: new = alpha * A + (1-alpha) * B
    Does the output distribution smoothly interpolate?
    """
    print(f"\n{'='*60}")
    print(f"  TEST 1: INTERPOLATION")
    print(f"  Does blending geometry blend behavior?")
    print(f"{'='*60}")

    # Use 'q' as our injection slot (rare in Shakespeare)
    slot_char = 'q'
    slot_idx = dataset.stoi[slot_char]

    # Characters to blend
    char_a, char_b = 'e', 'a'
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    emb_a = orig_emb[idx_a].clone()
    emb_b = orig_emb[idx_b].clone()
    head_a = orig_head[idx_a].clone()
    head_b = orig_head[idx_b].clone()

    # Test contexts — common Shakespeare patterns where e/a appear
    test_contexts = ["th", "wh", "sh", "st", "m", "h", "w", "c", "b", "f",
                     "pl", "tr", "gr", "pr", "sp", "sw", "cr", "fr", "br", "dr"]

    # Get baseline distributions for char_a and char_b in each context
    print(f"\n  Blending '{char_a}' and '{char_b}', injecting into '{slot_char}' slot")
    print(f"  Measuring output distribution after '{slot_char}' in {len(test_contexts)} contexts")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Get reference distributions
    ref_a_probs = []
    ref_b_probs = []
    valid_contexts = []

    for ctx in test_contexts:
        try:
            pa = measure_context_effect(model, dataset, config, ctx, idx_a)
            pb = measure_context_effect(model, dataset, config, ctx, idx_b)
            ref_a_probs.append(pa)
            ref_b_probs.append(pb)
            valid_contexts.append(ctx)
        except (KeyError, RuntimeError):
            continue

    # Now test interpolated embeddings
    print(f"\n  alpha=0.0 means pure '{char_b}', alpha=1.0 means pure '{char_a}'")
    print(f"\n  {'alpha':>7} | {'KL to A':>9} | {'KL to B':>9} | {'Top pred':>10} | {'Pred conf':>10}")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}")

    interpolation_kl_a = []
    interpolation_kl_b = []

    for alpha in alphas:
        # Construct blended embedding
        blended_emb = alpha * emb_a + (1 - alpha) * emb_b
        blended_head = alpha * head_a + (1 - alpha) * head_b

        # Inject into slot
        model.wte.weight.data[slot_idx] = blended_emb
        model.lm_head.weight.data[slot_idx] = blended_head

        # Measure across all contexts
        kl_to_a_list = []
        kl_to_b_list = []
        all_preds = []

        for i, ctx in enumerate(valid_contexts):
            p_blend = measure_context_effect(model, dataset, config, ctx, slot_idx)

            # KL divergence to reference distributions
            kl_a = F.kl_div(p_blend.log(), ref_a_probs[i], reduction='sum').item()
            kl_b = F.kl_div(p_blend.log(), ref_b_probs[i], reduction='sum').item()
            kl_to_a_list.append(kl_a)
            kl_to_b_list.append(kl_b)

            top_pred = torch.argmax(p_blend).item()
            all_preds.append(dataset.itos[top_pred])

        avg_kl_a = np.mean(kl_to_a_list)
        avg_kl_b = np.mean(kl_to_b_list)
        interpolation_kl_a.append(avg_kl_a)
        interpolation_kl_b.append(avg_kl_b)

        # Most common prediction
        from collections import Counter
        pred_counts = Counter(all_preds)
        top_pred_char, top_pred_count = pred_counts.most_common(1)[0]

        print(f"  {alpha:>7.1f} | {avg_kl_a:>9.4f} | {avg_kl_b:>9.4f} | {repr(top_pred_char):>10} | {top_pred_count}/{len(valid_contexts):>5}")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    # Check monotonicity: does KL to A decrease as alpha increases?
    kl_a_monotonic = all(interpolation_kl_a[i] >= interpolation_kl_a[i+1] - 0.01
                         for i in range(len(interpolation_kl_a)-1))
    kl_b_monotonic = all(interpolation_kl_b[i] <= interpolation_kl_b[i+1] - 0.01
                         for i in range(len(interpolation_kl_b)-1))

    # Compute correlation between alpha and KL distances
    alphas_arr = np.array(alphas)
    kl_a_arr = np.array(interpolation_kl_a)
    kl_b_arr = np.array(interpolation_kl_b)
    corr_a = np.corrcoef(alphas_arr, kl_a_arr)[0, 1]
    corr_b = np.corrcoef(alphas_arr, kl_b_arr)[0, 1]

    print(f"\n  Correlation: alpha vs KL-to-A: {corr_a:+.4f} (expect negative: closer to A as alpha rises)")
    print(f"  Correlation: alpha vs KL-to-B: {corr_b:+.4f} (expect positive: farther from B as alpha rises)")

    smooth = corr_a < -0.7 and corr_b > 0.7
    print(f"\n  Smooth interpolation: {'YES' if smooth else 'NO'}")

    return smooth, corr_a, corr_b


# =============================================================================
# TEST 2: Fractional position — genuinely new harmonic vector
# =============================================================================

def test_fractional_position(model, dataset, config):
    """
    Construct embeddings from the harmonic formula using non-integer
    character indices. These create genuinely NEW harmonic signatures
    that no character in the vocabulary has.
    """
    print(f"\n{'='*60}")
    print(f"  TEST 2: FRACTIONAL POSITION")
    print(f"  Construct genuinely new harmonic vectors from the formula")
    print(f"{'='*60}")

    slot_char = 'q'
    slot_idx = dataset.stoi[slot_char]
    vocab_size = dataset.vocab_size
    n_harmonics = config.n_embd // 2

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # Pick two adjacent characters in the vocab
    # Find 'e' and 'f' — adjacent in alphabet
    char_lo = 'e'
    char_hi = 'f'
    idx_lo = dataset.stoi[char_lo]
    idx_hi = dataset.stoi[char_hi]

    print(f"\n  Constructing vectors between '{char_lo}' (idx {idx_lo}) and '{char_hi}' (idx {idx_hi})")
    print(f"  Using harmonic formula: cos(n * c * 2pi / {vocab_size}), sin(...)")

    test_contexts = ["th", "wh", "sh", "st", "m", "h", "w", "c", "b",
                     "pl", "tr", "gr", "pr", "sp", "cr", "br", "li", "be", "re"]

    # Get reference distributions
    valid_contexts = []
    ref_lo_probs = []
    ref_hi_probs = []

    for ctx in test_contexts:
        try:
            p_lo = measure_context_effect(model, dataset, config, ctx, idx_lo)
            p_hi = measure_context_effect(model, dataset, config, ctx, idx_hi)
            ref_lo_probs.append(p_lo)
            ref_hi_probs.append(p_hi)
            valid_contexts.append(ctx)
        except (KeyError, RuntimeError):
            continue

    # Construct fractional embeddings
    fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

    print(f"\n  frac=0.0 means '{char_lo}', frac=1.0 means '{char_hi}'")
    print(f"\n  {'frac':>6} | {'c_value':>8} | {'KL to lo':>9} | {'KL to hi':>9} | {'Top pred':>10} | {'Coherent':>9}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")

    kl_lo_list = []
    kl_hi_list = []

    for frac in fractions:
        # Fractional character index
        c_val = idx_lo + frac * (idx_hi - idx_lo)

        # Construct from harmonic formula (matching the training-time initialization)
        theta = c_val * (2 * math.pi / vocab_size)
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        new_emb = torch.zeros(config.n_embd)
        new_emb[0::2] = torch.cos(harmonics * theta)
        new_emb[1::2] = torch.sin(harmonics * theta)
        new_emb = new_emb * (1.0 / math.sqrt(n_harmonics))

        # For lm_head, interpolate between the two characters' rows
        new_head = (1 - frac) * orig_head[idx_lo] + frac * orig_head[idx_hi]

        # Inject
        model.wte.weight.data[slot_idx] = new_emb.to(config.device)
        model.lm_head.weight.data[slot_idx] = new_head

        # Measure
        kl_to_lo = []
        kl_to_hi = []
        preds = []
        coherent_count = 0

        for i, ctx in enumerate(valid_contexts):
            p_new = measure_context_effect(model, dataset, config, ctx, slot_idx)
            kl_l = F.kl_div(p_new.log(), ref_lo_probs[i], reduction='sum').item()
            kl_h = F.kl_div(p_new.log(), ref_hi_probs[i], reduction='sum').item()
            kl_to_lo.append(kl_l)
            kl_to_hi.append(kl_h)

            top = torch.argmax(p_new).item()
            preds.append(dataset.itos[top])

            # Is the prediction a reasonable character? (letter or space)
            if dataset.itos[top].isalpha() or dataset.itos[top] == ' ':
                coherent_count += 1

        avg_kl_lo = np.mean(kl_to_lo)
        avg_kl_hi = np.mean(kl_to_hi)
        kl_lo_list.append(avg_kl_lo)
        kl_hi_list.append(avg_kl_hi)

        from collections import Counter
        top_pred = Counter(preds).most_common(1)[0][0]
        coherent_pct = coherent_count / len(valid_contexts) * 100

        print(f"  {frac:>6.1f} | {c_val:>8.2f} | {avg_kl_lo:>9.4f} | {avg_kl_hi:>9.4f} | {repr(top_pred):>10} | {coherent_pct:>7.0f}%")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    # Check: does KL to lo increase with fraction?
    corr_lo = np.corrcoef(fractions, kl_lo_list)[0, 1]
    corr_hi = np.corrcoef(fractions, kl_hi_list)[0, 1]

    print(f"\n  Correlation: frac vs KL-to-'{char_lo}': {corr_lo:+.4f} (expect positive)")
    print(f"  Correlation: frac vs KL-to-'{char_hi}': {corr_hi:+.4f} (expect negative)")

    return corr_lo, corr_hi


# =============================================================================
# TEST 3: Chimera — mix bands from different characters
# =============================================================================

def test_chimera(model, dataset, config):
    """
    Build a chimera: take low harmonics from one character and high harmonics
    from another. Does the model show blended behavior?
    """
    print(f"\n{'='*60}")
    print(f"  TEST 3: CHIMERA")
    print(f"  Low harmonics from one char, high harmonics from another")
    print(f"{'='*60}")

    slot_char = 'q'
    slot_idx = dataset.stoi[slot_char]
    n_harmonics = config.n_embd // 2

    # Source characters
    char_lo = 'e'  # low harmonics source
    char_hi = 't'  # high harmonics source
    idx_lo = dataset.stoi[char_lo]
    idx_hi = dataset.stoi[char_hi]

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    test_contexts = ["th", "wh", "sh", "st", "m", "h", "w", "c", "b",
                     "pl", "tr", "gr", "pr", "sp", "cr", "br", "li", "be", "re"]

    # Reference distributions
    valid_contexts = []
    ref_lo_probs = []
    ref_hi_probs = []

    for ctx in test_contexts:
        try:
            p_lo = measure_context_effect(model, dataset, config, ctx, idx_lo)
            p_hi = measure_context_effect(model, dataset, config, ctx, idx_hi)
            ref_lo_probs.append(p_lo)
            ref_hi_probs.append(p_hi)
            valid_contexts.append(ctx)
        except (KeyError, RuntimeError):
            continue

    print(f"\n  Chimera: low bands from '{char_lo}', high bands from '{char_hi}'")
    print(f"  Testing split points across {n_harmonics} harmonic bands")

    print(f"\n  {'Split':>6} | {'Lo bands':>9} | {'Hi bands':>9} | {'KL to lo':>9} | {'KL to hi':>9} | {'Top pred':>10}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")

    split_points = [0, 4, 8, 16, 24, 32, 48, 56, 64]

    for split in split_points:
        # Build chimera embedding
        chimera_emb = torch.zeros(config.n_embd)
        chimera_head = torch.zeros(dataset.vocab_size)

        # Low harmonics from char_lo
        for h in range(split):
            ci, si = h * 2, h * 2 + 1
            chimera_emb[ci] = orig_emb[idx_lo, ci]
            chimera_emb[si] = orig_emb[idx_lo, si]

        # High harmonics from char_hi
        for h in range(split, n_harmonics):
            ci, si = h * 2, h * 2 + 1
            chimera_emb[ci] = orig_emb[idx_hi, ci]
            chimera_emb[si] = orig_emb[idx_hi, si]

        # lm_head: weighted blend based on split ratio
        ratio = split / n_harmonics
        chimera_head = ratio * orig_head[idx_lo] + (1 - ratio) * orig_head[idx_hi]

        # Inject
        model.wte.weight.data[slot_idx] = chimera_emb.to(config.device)
        model.lm_head.weight.data[slot_idx] = chimera_head

        # Measure
        kl_to_lo = []
        kl_to_hi = []
        preds = []

        for i, ctx in enumerate(valid_contexts):
            p_chim = measure_context_effect(model, dataset, config, ctx, slot_idx)
            kl_l = F.kl_div(p_chim.log(), ref_lo_probs[i], reduction='sum').item()
            kl_h = F.kl_div(p_chim.log(), ref_hi_probs[i], reduction='sum').item()
            kl_to_lo.append(kl_l)
            kl_to_hi.append(kl_h)
            preds.append(dataset.itos[torch.argmax(p_chim).item()])

        from collections import Counter
        top_pred = Counter(preds).most_common(1)[0][0]

        print(f"  {split:>6} | {split:>9} | {n_harmonics-split:>9} | {np.mean(kl_to_lo):>9.4f} | {np.mean(kl_to_hi):>9.4f} | {repr(top_pred):>10}")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head


# =============================================================================
# TEST 4: Construct a "new character" and generate with it
# =============================================================================

def test_constructed_generation(model, dataset, config):
    """
    The taste test: construct a character that's halfway between
    'e' and 'a', inject it, and generate text. Does the output
    look like English with a blend of e/a characteristics?
    """
    print(f"\n{'='*60}")
    print(f"  TEST 4: GENERATION WITH CONSTRUCTED CHARACTER")
    print(f"  Create a blend, inject it, generate text")
    print(f"{'='*60}")

    slot_char = 'q'
    slot_idx = dataset.stoi[slot_char]
    char_a, char_b = 'e', 'a'
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    # Generate BEFORE injection (baseline with 'q')
    prompt_text = "\nKING HENRY:\nI shall "
    prompt = dataset.encode(prompt_text).unsqueeze(0).to(config.device)

    model.eval()
    torch.manual_seed(42)
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(150):
            logits = model.get_full_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            tokens = torch.cat([tokens, next_tok], dim=1)
    text_original = dataset.decode(tokens[0].tolist())

    print(f"\n  ORIGINAL (no injection):")
    print(f"    {repr(text_original[:200])}")

    # Now inject 50/50 blend into 'q' slot (both embedding and lm_head)
    blend_emb = 0.5 * orig_emb[idx_a] + 0.5 * orig_emb[idx_b]
    blend_head = 0.5 * orig_head[idx_a] + 0.5 * orig_head[idx_b]
    model.wte.weight.data[slot_idx] = blend_emb
    model.lm_head.weight.data[slot_idx] = blend_head

    # Also replace 'e' with our blend to see the effect in generation
    model.wte.weight.data[idx_a] = blend_emb
    model.lm_head.weight.data[idx_a] = blend_head

    torch.manual_seed(42)
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(150):
            logits = model.get_full_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            tokens = torch.cat([tokens, next_tok], dim=1)
    text_blended = dataset.decode(tokens[0].tolist())

    print(f"\n  WITH 50/50 BLEND (replacing 'e' with e/a blend):")
    print(f"    {repr(text_blended[:200])}")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    # Now do a pure harmonic construction: new theta halfway between e and a
    n_harmonics = config.n_embd // 2
    c_mid = (idx_a + idx_b) / 2.0
    theta_mid = c_mid * (2 * math.pi / dataset.vocab_size)
    harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
    constructed_emb = torch.zeros(config.n_embd)
    constructed_emb[0::2] = torch.cos(harmonics * theta_mid)
    constructed_emb[1::2] = torch.sin(harmonics * theta_mid)
    constructed_emb = constructed_emb * (1.0 / math.sqrt(n_harmonics))

    # Inject the pure harmonic construction
    model.wte.weight.data[idx_a] = constructed_emb.to(config.device)
    model.lm_head.weight.data[idx_a] = 0.5 * orig_head[idx_a] + 0.5 * orig_head[idx_b]

    torch.manual_seed(42)
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(150):
            logits = model.get_full_logits(tokens[:, -config.block_size:])
            probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            tokens = torch.cat([tokens, next_tok], dim=1)
    text_constructed = dataset.decode(tokens[0].tolist())

    print(f"\n  WITH PURE HARMONIC CONSTRUCTION (theta halfway between e and a):")
    print(f"    {repr(text_constructed[:200])}")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    # Count character frequencies
    print(f"\n  Character frequencies:")
    print(f"    {'':>25} {'e':>5} {'a':>5} {'q':>5}")
    print(f"    {'Original':>25} {text_original[:200].count('e'):>5} {text_original[:200].count('a'):>5} {text_original[:200].count('q'):>5}")
    print(f"    {'50/50 blend':>25} {text_blended[:200].count('e'):>5} {text_blended[:200].count('a'):>5} {text_blended[:200].count('q'):>5}")
    print(f"    {'Harmonic construction':>25} {text_constructed[:200].count('e'):>5} {text_constructed[:200].count('a'):>5} {text_constructed[:200].count('q'):>5}")


# =============================================================================
# TEST 5: The acid test — predict what a constructed vector will do
# =============================================================================

def test_prediction_accuracy(model, dataset, config):
    """
    The real proof: if we can PREDICT what a constructed vector will do
    before running it, then we truly understand the geometry.

    Method:
    - For each test context, measure P(output | char_a) and P(output | char_b)
    - Predict: P(output | blend(alpha)) should be approx alpha*P(a) + (1-alpha)*P(b)
    - Compare prediction to actual
    - If correlation is high, we can write arbitrary policies
    """
    print(f"\n{'='*60}")
    print(f"  TEST 5: PREDICTION ACCURACY")
    print(f"  Can we PREDICT what a constructed vector will do?")
    print(f"{'='*60}")

    char_a, char_b = 'e', 'a'
    slot_char = 'q'
    idx_a = dataset.stoi[char_a]
    idx_b = dataset.stoi[char_b]
    slot_idx = dataset.stoi[slot_char]

    orig_emb = model.wte.weight.data.clone()
    orig_head = model.lm_head.weight.data.clone()

    test_contexts = ["th", "wh", "sh", "st", "m", "h", "w", "c", "b", "f",
                     "pl", "tr", "gr", "pr", "sp", "sw", "cr", "fr", "br", "dr"]

    valid_contexts = []
    ref_a = []
    ref_b = []

    for ctx in test_contexts:
        try:
            pa = measure_context_effect(model, dataset, config, ctx, idx_a)
            pb = measure_context_effect(model, dataset, config, ctx, idx_b)
            ref_a.append(pa)
            ref_b.append(pb)
            valid_contexts.append(ctx)
        except (KeyError, RuntimeError):
            continue

    print(f"\n  Testing prediction accuracy across {len(valid_contexts)} contexts")
    print(f"  Prediction model: P(blend) ~= alpha * P('{char_a}') + (1-alpha) * P('{char_b}')")

    alphas = [0.2, 0.4, 0.5, 0.6, 0.8]
    all_correlations = []

    print(f"\n  {'alpha':>7} | {'Avg corr':>9} | {'Min corr':>9} | {'Max corr':>9} | {'Prediction':>11}")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*11}")

    for alpha in alphas:
        # Construct blend
        blend_emb = alpha * orig_emb[idx_a] + (1 - alpha) * orig_emb[idx_b]
        blend_head = alpha * orig_head[idx_a] + (1 - alpha) * orig_head[idx_b]
        model.wte.weight.data[slot_idx] = blend_emb
        model.lm_head.weight.data[slot_idx] = blend_head

        correlations = []
        for i, ctx in enumerate(valid_contexts):
            # Predicted distribution
            predicted = alpha * ref_a[i] + (1 - alpha) * ref_b[i]

            # Actual distribution
            actual = measure_context_effect(model, dataset, config, ctx, slot_idx)

            # Correlation between predicted and actual
            corr = np.corrcoef(predicted.numpy(), actual.numpy())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        avg_corr = np.mean(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)
        all_correlations.extend(correlations)

        quality = "EXCELLENT" if avg_corr > 0.9 else "GOOD" if avg_corr > 0.7 else "MODERATE" if avg_corr > 0.5 else "POOR"
        print(f"  {alpha:>7.1f} | {avg_corr:>9.4f} | {min_corr:>9.4f} | {max_corr:>9.4f} | {quality:>11}")

    # Restore
    model.wte.weight.data = orig_emb
    model.lm_head.weight.data = orig_head

    overall_corr = np.mean(all_correlations)
    print(f"\n  Overall prediction accuracy (correlation): {overall_corr:.4f}")

    if overall_corr > 0.9:
        print(f"  The model is a LINEAR function of its input geometry.")
        print(f"  We can predict behavior for ANY constructed vector.")
    elif overall_corr > 0.7:
        print(f"  The model is APPROXIMATELY linear in input geometry.")
        print(f"  Predictions are reliable but with some nonlinear effects.")
    elif overall_corr > 0.5:
        print(f"  The model shows MODERATE linearity in input geometry.")
        print(f"  Nonlinear processing adds significant deviation.")
    else:
        print(f"  The model is NONLINEAR in input geometry.")
        print(f"  Simple blending predictions do not capture model behavior.")

    return overall_corr


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  HARMONIC CONSTRUCTION")
    print(f"  Can we build new identity from scratch?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train
    model = train_model(config, dataset, mode="harmonic")

    # Run all tests
    smooth, corr_a, corr_b = test_interpolation(model, dataset, config)
    corr_lo, corr_hi = test_fractional_position(model, dataset, config)
    test_chimera(model, dataset, config)
    test_constructed_generation(model, dataset, config)
    pred_accuracy = test_prediction_accuracy(model, dataset, config)

    # Final verdict
    print(f"\n{'='*60}")
    print(f"  FINAL VERDICT: Can we construct new identity from scratch?")
    print(f"{'='*60}")

    print(f"\n  Test 1 (Interpolation):     {'SMOOTH' if smooth else 'ROUGH'} (corr {corr_a:+.3f}/{corr_b:+.3f})")
    print(f"  Test 2 (Fractional):        corr ({corr_lo:+.3f}/{corr_hi:+.3f})")
    print(f"  Test 5 (Prediction):        {pred_accuracy:.3f} correlation")

    score = 0
    if smooth:
        score += 1
    if corr_lo > 0.5:
        score += 1
    if pred_accuracy > 0.7:
        score += 1

    if score >= 3:
        print(f"\n  CONFIRMED: New identity can be constructed from harmonic geometry.")
        print(f"  The model processes constructed vectors predictably.")
        print(f"  You can write new knowledge as harmonic vectors — no training needed.")
    elif score >= 2:
        print(f"\n  PARTIALLY CONFIRMED: Construction works with some predictability.")
        print(f"  The model responds to constructed geometry but not perfectly linearly.")
    elif score >= 1:
        print(f"\n  WEAK: Some construction effects visible but not reliable.")
    else:
        print(f"\n  NEGATIVE: Constructed vectors do not produce predictable behavior.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
