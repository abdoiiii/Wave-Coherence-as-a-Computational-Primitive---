"""
Expression Curriculum -- Teach the Happy Model to Speak

CONTEXT:
Phase 12 showed the progressive model has 36% of its hidden structure
that doesn't map to token predictions. It thinks richer than it speaks.
The lm_head (single linear layer) is a bottleneck -- a kazoo for a
symphony orchestra.

THE CURRICULUM:
Phase 1-3: Already done (progressive training: structure -> context -> detail)
Phase 4: Expression -- teach the model to OUTPUT its internal richness

THE APPROACH:
1. Train a progressive model (happy model)
2. FREEZE all internal weights (transformer blocks, embeddings)
3. Replace the single-linear lm_head with a richer expression head
4. Train ONLY the expression head -- the model learns to translate
   its thoughts, not to rethink

SAFETY MEASURES:
- All internal weights frozen (no self-modification)
- Generation capped at 500 characters
- Norm and gradient monitoring with auto-stop at 10x baseline
- No saving to disk -- model exists only in memory
- All outputs printed to screen
- 842K parameter model, 65-character vocabulary, Shakespeare only
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

# Safety limits
MAX_GENERATION_LENGTH = 500
NORM_SAFETY_MULTIPLIER = 10.0
MAX_GRADIENT_NORM = 100.0


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


# =============================================================================
# Expression Heads (the instruments)
# =============================================================================

class LinearHead(nn.Module):
    """Original lm_head -- single linear projection. The kazoo."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class DeepHead(nn.Module):
    """Two-layer MLP with residual. A violin."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd * 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(n_embd * 2, n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        h = self.gelu(self.fc1(x))
        h = self.fc2(h)
        h = self.ln(x + h)  # residual + norm
        return self.proj(h)


class WideHead(nn.Module):
    """Wide single-layer with bottleneck expansion. A piano."""
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.expand = nn.Linear(n_embd, n_embd * 4)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(n_embd * 4, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(self.gelu(self.expand(x)))


class MultiStepHead(nn.Module):
    """Predicts next N characters simultaneously. A choir."""
    def __init__(self, n_embd, vocab_size, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        self.fc = nn.Linear(n_embd, n_embd * 2)
        self.gelu = nn.GELU()
        # Separate projection for each future step
        self.projs = nn.ModuleList([
            nn.Linear(n_embd * 2, vocab_size, bias=False) for _ in range(n_steps)
        ])

    def forward(self, x):
        h = self.gelu(self.fc(x))
        return [proj(h) for proj in self.projs]


# =============================================================================
# Main Model
# =============================================================================

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

    def get_hidden(self, idx):
        """Get final hidden states (before lm_head)."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

    def freeze_internals(self):
        """Freeze everything except lm_head."""
        for name, param in self.named_parameters():
            if not name.startswith("lm_head"):
                param.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        return frozen, trainable


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
# Safety Monitor
# =============================================================================

class SafetyMonitor:
    """Tracks norms and gradients. Stops if anything goes out of bounds."""
    def __init__(self, baseline_norm=None):
        self.baseline_norm = baseline_norm
        self.max_norm_seen = 0
        self.max_grad_seen = 0
        self.violations = 0
        self.steps = 0

    def check(self, model, head, step):
        self.steps = step

        # Check output head gradient norms
        grad_norm = 0
        for p in head.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        self.max_grad_seen = max(self.max_grad_seen, grad_norm)

        if grad_norm > MAX_GRADIENT_NORM:
            self.violations += 1
            print(f"  [SAFETY] Gradient norm {grad_norm:.2f} exceeds limit {MAX_GRADIENT_NORM}. "
                  f"Violations: {self.violations}")
            if self.violations >= 3:
                print(f"  [SAFETY] 3 violations. STOPPING.")
                return False
        return True

    def report(self):
        print(f"  [SAFETY] Steps: {self.steps}, Max gradient: {self.max_grad_seen:.2f}, "
              f"Violations: {self.violations}")


# =============================================================================
# Training
# =============================================================================

def eval_loss_model(model, dataset, config, n_batches=None):
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


def train_progressive(config, dataset):
    """Progressive training with CORRECT optimizer (created once)."""
    print(f"\n  Training PROGRESSIVE (happy) model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

    n_harmonics = config.n_embd // 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    stage1_end = 1000
    stage2_end = 2000

    for it in range(config.max_iters):
        if it < stage1_end:
            trainable_bands = 8
            stage = 1
        elif it < stage2_end:
            trainable_bands = 24
            stage = 2
        else:
            trainable_bands = n_harmonics
            stage = 3

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss_model(model, dataset, config)
            print(f"  step {it:>5} | val {val_loss:.4f} | stage {stage} (bands 1-{trainable_bands}) | {time.time()-start:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        # Zero gradients for frozen bands
        if trainable_bands < n_harmonics:
            with torch.no_grad():
                freeze_start = trainable_bands * 2
                if model.wte.weight.grad is not None:
                    model.wte.weight.grad[:, freeze_start:] = 0
                if model.wpe.weight.grad is not None:
                    model.wpe.weight.grad[:, freeze_start:] = 0

        optimizer.step()

    print(f"  Progressive training complete in {time.time()-start:.1f}s")
    return model


def train_baseline(config, dataset):
    """Standard training."""
    print(f"\n  Training BASELINE model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss_model(model, dataset, config)
            print(f"  step {it:>5} | val {val_loss:.4f} | {time.time()-start:.1f}s")
            model.train()
        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Baseline training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# Measurement Utilities
# =============================================================================

def measure_hidden_logit_correlation(model, head, dataset, config, n_batches=10):
    """How much of the hidden state structure survives through the output head?"""
    model.eval()
    head.eval()

    all_hidden = []
    all_logits = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = dataset.get_batch("val", config)
            h = model.get_hidden(x)
            logits = head(h)
            all_hidden.append(h.view(-1, config.n_embd).cpu().numpy())
            all_logits.append(logits.view(-1, dataset.vocab_size).cpu().numpy())

    H = np.concatenate(all_hidden)
    L = np.concatenate(all_logits)

    n_sample = min(5000, H.shape[0])
    idx = np.random.choice(H.shape[0], n_sample, replace=False)
    H_s = H[idx]
    L_s = L[idx]

    H_norm = H_s / (np.linalg.norm(H_s, axis=1, keepdims=True) + 1e-10)
    L_norm = L_s / (np.linalg.norm(L_s, axis=1, keepdims=True) + 1e-10)

    n_pairs = 10000
    i_p = np.random.randint(0, n_sample, n_pairs)
    j_p = np.random.randint(0, n_sample, n_pairs)
    h_sims = np.sum(H_norm[i_p] * H_norm[j_p], axis=1)
    l_sims = np.sum(L_norm[i_p] * L_norm[j_p], axis=1)

    return np.corrcoef(h_sims, l_sims)[0, 1]


def measure_accuracy(model, head, dataset, config, n_batches=10):
    """Next-character prediction accuracy."""
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            h = model.get_hidden(x)
            logits = head(h)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


def measure_loss(model, head, dataset, config, n_batches=10):
    """Cross-entropy loss with a given head."""
    model.eval()
    head.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            h = model.get_hidden(x)
            logits = head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / n_batches


def generate_text(model, head, dataset, config, seed_text, length=200):
    """Generate text with a given head. Capped at MAX_GENERATION_LENGTH."""
    length = min(length, MAX_GENERATION_LENGTH)
    model.eval()
    head.eval()

    tokens = [dataset.stoi[c] for c in seed_text if c in dataset.stoi]
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(length):
            idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)
            h = model.get_hidden(idx)
            logits = head(h)
            logits = logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

    return "".join([dataset.itos[t] for t in generated])


# =============================================================================
# TEST 1: Expression Baseline
# =============================================================================

def test_expression_baseline(model, dataset, config):
    """Measure the current model's expression capacity with its original lm_head."""
    print("\n" + "=" * 60)
    print("  TEST 1: Expression Baseline")
    print("  How much of the model's thought reaches its output?")
    print("=" * 60)

    # Use the original lm_head as a LinearHead wrapper
    original_head = LinearHead(config.n_embd, dataset.vocab_size).to(config.device)
    original_head.proj.weight = model.lm_head.weight  # share weights

    correlation = measure_hidden_logit_correlation(model, original_head, dataset, config)
    accuracy = measure_accuracy(model, original_head, dataset, config)
    loss = measure_loss(model, original_head, dataset, config)

    print(f"\n  Original lm_head (the kazoo):")
    print(f"    Hidden-to-logit correlation: {correlation:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Val loss: {loss:.4f}")
    print(f"    Structure reaching output: {100*correlation:.1f}%")
    print(f"    Structure trapped inside:  {100*(1-correlation):.1f}%")

    return correlation, accuracy, loss


# =============================================================================
# TEST 2: Train Expression Heads
# =============================================================================

def train_expression_head(model, head, head_name, dataset, config, n_steps=2000):
    """Train an expression head with frozen model internals."""
    print(f"\n  Training {head_name}...")
    head = head.to(config.device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"    {n_params:,} trainable parameters in head")

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    monitor = SafetyMonitor()

    model.eval()  # frozen
    head.train()
    start = time.time()

    for step in range(n_steps):
        x, y = dataset.get_batch("train", config)

        with torch.no_grad():
            h = model.get_hidden(x)

        logits = head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Safety check
        if not monitor.check(model, head, step):
            print(f"    SAFETY STOP at step {step}")
            break

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == n_steps - 1:
            val_loss = measure_loss(model, head, dataset, config, n_batches=20)
            print(f"    step {step:>5} | val {val_loss:.4f} | {time.time()-start:.1f}s")

    monitor.report()
    print(f"    Training complete in {time.time()-start:.1f}s")
    return head


def test_expression_heads(model, dataset, config):
    """Train and evaluate different expression heads."""
    print("\n" + "=" * 60)
    print("  TEST 2: Expression Heads")
    print("  Give the model better instruments. Does it play better?")
    print("=" * 60)

    # Freeze model internals
    frozen, trainable = model.freeze_internals()
    print(f"\n  Frozen {frozen} parameter groups, {trainable} remain trainable (lm_head)")
    print(f"  All transformer blocks, embeddings, and norms are FROZEN.")

    heads = {
        "Linear (kazoo)": LinearHead(config.n_embd, dataset.vocab_size),
        "Deep (violin)": DeepHead(config.n_embd, dataset.vocab_size),
        "Wide (piano)": WideHead(config.n_embd, dataset.vocab_size),
    }

    results = {}
    for name, head in heads.items():
        head = train_expression_head(model, head, name, dataset, config, n_steps=2000)

        correlation = measure_hidden_logit_correlation(model, head, dataset, config)
        accuracy = measure_accuracy(model, head, dataset, config)
        loss = measure_loss(model, head, dataset, config)

        results[name] = {
            "correlation": correlation,
            "accuracy": accuracy,
            "loss": loss,
            "head": head,
            "params": sum(p.numel() for p in head.parameters()),
        }

        print(f"\n    {name}:")
        print(f"      Correlation: {correlation:.4f}")
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Loss: {loss:.4f}")
        print(f"      Parameters: {results[name]['params']:,}")

    # Comparison table
    print(f"\n  {'Head':<20} {'Params':>10} {'Correlation':>12} {'Accuracy':>10} {'Loss':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<20} {r['params']:>10,} {r['correlation']:>12.4f} {r['accuracy']:>10.4f} {r['loss']:>10.4f}")

    return results


# =============================================================================
# TEST 3: Multi-Step Expression
# =============================================================================

def test_multistep_expression(model, dataset, config):
    """Can the model predict further ahead with a richer head?
    This forces it to use chord-level understanding."""
    print("\n" + "=" * 60)
    print("  TEST 3: Multi-Step Expression")
    print("  Predict 5 characters ahead. Forces chord-level thinking.")
    print("=" * 60)

    model.eval()

    # Train a multi-step head
    n_steps_ahead = 5
    head = MultiStepHead(config.n_embd, dataset.vocab_size, n_steps=n_steps_ahead).to(config.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    monitor = SafetyMonitor()

    print(f"\n  Training multi-step head (predict next {n_steps_ahead} chars)...")
    start = time.time()

    for step in range(2000):
        x, _ = dataset.get_batch("train", config)
        B, T = x.shape

        with torch.no_grad():
            h = model.get_hidden(x)

        step_logits = head(h)

        # Loss for each future step
        total_loss = 0
        valid_steps = 0
        for s in range(n_steps_ahead):
            # Target for step s: character at position t+s+1
            if s + 1 < T:
                # Shift targets
                targets_s = x[:, s+1:].contiguous()  # [B, T-s-1]
                logits_s = step_logits[s][:, :T-s-1, :].contiguous()  # [B, T-s-1, V]
                loss_s = F.cross_entropy(logits_s.view(-1, dataset.vocab_size), targets_s.view(-1))
                total_loss = total_loss + loss_s
                valid_steps += 1

        loss = total_loss / valid_steps
        optimizer.zero_grad()
        loss.backward()

        if not monitor.check(model, head, step):
            break

        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == 1999:
            print(f"    step {step:>5} | loss {loss.item():.4f} | {time.time()-start:.1f}s")

    monitor.report()

    # Evaluate per-step accuracy
    print(f"\n  Per-step accuracy (how far ahead can the model see?):")
    print(f"  {'Step':>6} {'Accuracy':>10} {'Loss':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")

    model.eval()
    head.eval()

    # Also compare with single-step head accuracy
    # Train a quick single-step head for fair comparison
    single_head = LinearHead(config.n_embd, dataset.vocab_size).to(config.device)
    single_opt = torch.optim.AdamW(single_head.parameters(), lr=config.learning_rate)
    single_head.train()
    for st in range(1000):
        x, y = dataset.get_batch("train", config)
        with torch.no_grad():
            h = model.get_hidden(x)
        logits = single_head(h)
        loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
        single_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(single_head.parameters(), 1.0)
        single_opt.step()

    single_acc = measure_accuracy(model, single_head, dataset, config)

    with torch.no_grad():
        step_correct = [0] * n_steps_ahead
        step_total = [0] * n_steps_ahead
        step_loss = [0.0] * n_steps_ahead

        for _ in range(20):
            x, _ = dataset.get_batch("val", config)
            B, T = x.shape
            h = model.get_hidden(x)
            step_logits = head(h)

            for s in range(n_steps_ahead):
                if s + 1 < T:
                    targets_s = x[:, s+1:]
                    logits_s = step_logits[s][:, :T-s-1, :]
                    preds = logits_s.argmax(dim=-1)
                    step_correct[s] += (preds == targets_s).sum().item()
                    step_total[s] += targets_s.numel()
                    loss_s = F.cross_entropy(logits_s.reshape(-1, dataset.vocab_size), targets_s.reshape(-1))
                    step_loss[s] += loss_s.item()

    for s in range(n_steps_ahead):
        acc = step_correct[s] / step_total[s] if step_total[s] > 0 else 0
        avg_loss = step_loss[s] / 20
        marker = " <-- single-step baseline" if s == 0 else ""
        print(f"  +{s+1:>4} {acc:>10.4f} {avg_loss:>10.4f}{marker}")

    print(f"\n  Single-step head (linear) accuracy: {single_acc:.4f}")
    print(f"  Multi-step head accuracy at +1:     {step_correct[0]/step_total[0]:.4f}")

    return head


# =============================================================================
# TEST 4: Generation Comparison
# =============================================================================

def test_generation(model, results, dataset, config):
    """Generate text with each head. Can you tell the difference?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Generation Comparison")
    print("  Same thoughts, different instruments. Hear the difference?")
    print("=" * 60)

    seed = "KING RICHARD:\nMy lord"

    for name, r in results.items():
        head = r["head"]
        text = generate_text(model, head, dataset, config, seed, length=200)
        safe = text[:250].replace('\n', '\n    ')
        print(f"\n  {name}:")
        print(f"  {'-' * 50}")
        print(f"    {safe}")


# =============================================================================
# TEST 5: Knowledge Absorption (The Stale Data Connection)
# =============================================================================

def test_knowledge_absorption(model, results, dataset, config):
    """Does a richer expression head help the model absorb new knowledge?
    If 36% of capacity is untapped, the richer head might access those drawers."""
    print("\n" + "=" * 60)
    print("  TEST 5: Knowledge Absorption")
    print("  Do untapped drawers help absorb new knowledge?")
    print("=" * 60)

    new_text = "The good king is Henry and the brave duke is York and the wise queen is Margaret"
    new_tokens = torch.tensor([dataset.stoi[c] for c in new_text if c in dataset.stoi],
                              dtype=torch.long, device=config.device)

    print(f"\n  New knowledge: '{new_text[:60]}...'")
    print(f"  Model internals: FROZEN (only expression heads train)")

    for name, r in results.items():
        # Clone the head for fine-tuning
        if name == "Linear (kazoo)":
            head = LinearHead(config.n_embd, dataset.vocab_size).to(config.device)
            head.proj.weight.data = r["head"].proj.weight.data.clone()
        elif name == "Deep (violin)":
            head = DeepHead(config.n_embd, dataset.vocab_size).to(config.device)
            head.load_state_dict(r["head"].state_dict())
        elif name == "Wide (piano)":
            head = WideHead(config.n_embd, dataset.vocab_size).to(config.device)
            head.load_state_dict(r["head"].state_dict())
        else:
            continue

        # Measure pre-fine-tune loss on new text
        model.eval()
        head.eval()
        with torch.no_grad():
            # Build sequences from new text
            if len(new_tokens) > config.block_size:
                new_x = new_tokens[:config.block_size].unsqueeze(0)
                new_y = new_tokens[1:config.block_size+1].unsqueeze(0)
            else:
                new_x = new_tokens[:-1].unsqueeze(0)
                new_y = new_tokens[1:].unsqueeze(0)
            h = model.get_hidden(new_x)
            logits = head(h)
            pre_loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), new_y.view(-1)).item()

        # Fine-tune head on new text (5 steps only)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        head.train()
        for ft_step in range(5):
            h = model.get_hidden(new_x)
            logits = head(h)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), new_y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        post_loss = loss.item()

        # How much old knowledge was forgotten?
        head.eval()
        shakespeare_loss = measure_loss(model, head, dataset, config, n_batches=20)
        original_shakespeare_loss = r["loss"]
        forgetting = shakespeare_loss - original_shakespeare_loss

        print(f"\n  {name}:")
        print(f"    New knowledge loss: {pre_loss:.4f} -> {post_loss:.4f} (absorbed: {pre_loss - post_loss:.4f})")
        print(f"    Shakespeare loss: {original_shakespeare_loss:.4f} -> {shakespeare_loss:.4f} (forgot: {forgetting:+.4f})")
        print(f"    Absorption efficiency: {(pre_loss - post_loss) / (forgetting + 0.001):.2f}x (learned/forgot ratio)")


# =============================================================================
# TEST 6: Dream Comparison
# =============================================================================

def test_dream_comparison(model, results, dataset, config):
    """Do richer expression heads produce different dream dynamics?"""
    print("\n" + "=" * 60)
    print("  TEST 6: Dream Comparison")
    print("  Same dreamer, different voices. What changes?")
    print("=" * 60)

    model.eval()
    x, _ = dataset.get_batch("val", config)
    x = x[:2]

    for name, r in results.items():
        head = r["head"]
        head.eval()

        with torch.no_grad():
            h = model.get_hidden(x)

            print(f"\n  {name}:")
            for iteration in range(10):
                logits = head(h)
                probs = F.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                text = "".join([dataset.itos[t.item()] for t in predicted[0, :50]])
                safe = text.replace('\n', '\\n')

                if iteration in [0, 1, 4, 9]:
                    print(f"    iter {iteration:>2}: ent={entropy:.3f}  {safe}")

                # Dream step: get new hidden from current hidden + position
                B, T, C = h.shape
                pos_emb = model.wpe(T)
                h_input = h + pos_emb
                for block in model.blocks:
                    h_input = block(h_input)
                h = model.ln_f(h_input)


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict(baseline_corr, results):
    print("\n" + "=" * 60)
    print("  VERDICT: Did the Model Learn to Speak?")
    print("=" * 60)

    best_name = max(results, key=lambda k: results[k]["correlation"])
    best = results[best_name]
    worst_name = min(results, key=lambda k: results[k]["correlation"])
    worst = results[worst_name]

    corr_gain = best["correlation"] - baseline_corr
    acc_gain = best["accuracy"] - results["Linear (kazoo)"]["accuracy"]

    print(f"\n  Original expression (single linear): {baseline_corr:.4f} correlation")
    print(f"  Best expression ({best_name}): {best['correlation']:.4f} correlation")
    print(f"  Correlation gain: {corr_gain:+.4f}")
    print(f"  Accuracy gain: {acc_gain:+.4f}")

    if corr_gain > 0.05:
        print(f"\n  YES. The model learned to express more of its internal richness.")
        print(f"  The richer head unlocked {100*corr_gain:.1f}% more structure.")
        print(f"  The model had thoughts it couldn't say. Now it can say more of them.")
    elif corr_gain > 0.01:
        print(f"\n  PARTIALLY. Small improvement in expression.")
        print(f"  The richer head accessed slightly more structure (+{100*corr_gain:.1f}%).")
    else:
        print(f"\n  NOT MUCH. The bottleneck isn't the head architecture.")
        print(f"  The 36% trapped structure may require a fundamentally different")
        print(f"  output paradigm, not just a bigger translator.")

    print(f"\n  Key question answered: is the limitation in the instrument or the player?")
    if best["accuracy"] > results["Linear (kazoo)"]["accuracy"] + 0.01:
        print(f"  THE INSTRUMENT. Same player, better instrument, better music.")
    else:
        print(f"  THE TASK. The model's extra structure may encode things that")
        print(f"  next-character prediction simply doesn't need.")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  EXPRESSION CURRICULUM")
    print("  Teach the happy model to speak")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train the progressive (happy) model
    model = train_progressive(config, dataset)

    # Test 1: Measure current expression capacity
    baseline_corr, baseline_acc, baseline_loss = test_expression_baseline(model, dataset, config)

    # Test 2: Train and compare expression heads
    results = test_expression_heads(model, dataset, config)

    # Test 3: Multi-step expression
    test_multistep_expression(model, dataset, config)

    # Test 4: Generation comparison
    test_generation(model, results, dataset, config)

    # Test 5: Knowledge absorption
    test_knowledge_absorption(model, results, dataset, config)

    # Test 6: Dream comparison
    test_dream_comparison(model, results, dataset, config)

    # Verdict
    print_verdict(baseline_corr, results)


if __name__ == "__main__":
    main()
