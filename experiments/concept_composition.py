"""
Concept Composition Test — Characters or Chords?

THE QUESTION:
Does the model see k, i, n, g as four separate harmonic vectors?
Or does it compose "king" as a concept — a chord built from the
interaction of those harmonics through attention?

THE TEST:
1. Feed the same character in different contexts:
   - 'g' in "king" vs 'g' in "dog" vs 'g' in "going"
   - 'e' in "the" vs 'e' in "queen" vs 'e' in "here"
2. Capture the internal representation at EVERY layer
3. At the embedding layer: should be IDENTICAL (same char = same vector)
4. After attention: if representations DIVERGE, the model is composing
5. Track WHICH harmonic bands diverge — where do concepts live?

PREDICTION:
- Low bands (structure) should stay similar across contexts
- High bands (identity) should diverge as context modulates them
- This would mean concepts are built on structural foundations
  — exactly what progressive learning aims to exploit
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
# Model (same as all experiments)
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

    def forward_with_all_activations(self, idx):
        """Capture representation of EVERY position at EVERY layer."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        h = tok_emb + pos_emb

        activations = {}
        activations["embedding"] = tok_emb.detach()
        activations["embed+pos"] = h.detach()

        for i, block in enumerate(self.blocks):
            ln1_out = block.ln_1(h)
            attn_out = block.attn(ln1_out)
            h = h + attn_out
            activations[f"layer{i}_post_attn"] = h.detach()
            ln2_out = block.ln_2(h)
            mlp_out = block.mlp(ln2_out)
            h = h + mlp_out
            activations[f"layer{i}_post_mlp"] = h.detach()

        h = self.ln_f(h)
        activations["final"] = h.detach()
        return activations

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
        return torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)


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
# Test 1: Same character, different contexts — does representation diverge?
# =============================================================================

def test_context_divergence(model, dataset, config):
    """
    Feed the same character in different contexts. Measure how much
    the representation diverges at each layer.
    """
    print(f"\n{'='*60}")
    print(f"  TEST 1: Context Divergence")
    print(f"  Does the same character look different in different contexts?")
    print(f"{'='*60}")

    model.eval()
    n_harmonics = config.n_embd // 2

    # Test groups: same target character, different contexts
    # Format: (context_before, target_char, label)
    test_groups = {
        'e': [
            ("th", 'e', "the"),
            ("her", 'e', "here"),
            ("lov", 'e', "love"),
            ("nam", 'e', "name"),
            ("com", 'e', "come"),
            ("tak", 'e', "take"),
        ],
        'n': [
            ("ki", 'n', "kin(g)"),
            ("quee", 'n', "queen"),
            ("whe", 'n', "when"),
            ("the", 'n', "then"),
            ("i", 'n', "in"),
            ("ca", 'n', "can"),
        ],
        'o': [
            ("wh", 'o', "who"),
            ("g", 'o', "go"),
            ("n", 'o', "no"),
            ("d", 'o', "do"),
            ("t", 'o', "to"),
            ("als", 'o', "also"),
        ],
        't': [
            ("bu", 't', "but"),
            ("no", 't', "not"),
            ("ha", 't', "hat"),
            ("wha", 't', "what"),
            ("tha", 't', "that"),
            ("ge", 't', "get"),
        ],
    }

    layers = ["embedding", "embed+pos",
              "layer0_post_attn", "layer0_post_mlp",
              "layer1_post_attn", "layer1_post_mlp",
              "layer2_post_attn", "layer2_post_mlp",
              "layer3_post_attn", "layer3_post_mlp",
              "final"]

    all_divergences = {}

    for target_char, contexts in test_groups.items():
        print(f"\n  Character '{target_char}' in {len(contexts)} contexts:")

        # Collect representations at each layer
        layer_reps = {layer: [] for layer in layers}

        for ctx_text, tchar, label in contexts:
            full_text = ctx_text + tchar
            tokens = dataset.encode(full_text).unsqueeze(0).to(config.device)

            with torch.no_grad():
                acts = model.forward_with_all_activations(tokens)

            # Get the representation at the TARGET character's position
            target_pos = len(full_text) - 1  # last position
            for layer in layers:
                rep = acts[layer][0, target_pos, :].cpu().numpy()
                layer_reps[layer].append(rep)

        # Compute pairwise cosine distances at each layer
        layer_divergences = {}
        for layer in layers:
            reps = np.array(layer_reps[layer])
            # Pairwise cosine similarity
            norms = np.linalg.norm(reps, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized = reps / norms
            sim_matrix = normalized @ normalized.T
            # Average off-diagonal similarity
            n = len(reps)
            total_sim = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_sim += sim_matrix[i, j]
                    count += 1
            avg_sim = total_sim / count if count > 0 else 1.0
            divergence = 1.0 - avg_sim  # 0 = identical, 1 = orthogonal
            layer_divergences[layer] = divergence

        all_divergences[target_char] = layer_divergences

        # Print compact table
        print(f"    {'Layer':>20} | {'Divergence':>11} | {'Interpretation':>20}")
        print(f"    {'-'*20}-+-{'-'*11}-+-{'-'*20}")

        for layer in layers:
            div = layer_divergences[layer]
            if div < 0.01:
                interp = "identical"
            elif div < 0.1:
                interp = "similar"
            elif div < 0.3:
                interp = "DIVERGING"
            elif div < 0.5:
                interp = "DIFFERENT"
            else:
                interp = "VERY DIFFERENT"
            print(f"    {layer:>20} | {div:>11.4f} | {interp:>20}")

    return all_divergences


# =============================================================================
# Test 2: Which bands carry context vs character identity?
# =============================================================================

def test_band_divergence(model, dataset, config):
    """
    For the same character in different contexts, measure which
    HARMONIC BANDS diverge most. This tells us:
    - Low bands diverge = structure is context-dependent
    - High bands diverge = identity is context-dependent
    - Specific bands diverge = those bands carry conceptual info
    """
    print(f"\n{'='*60}")
    print(f"  TEST 2: Which Bands Carry Context?")
    print(f"  Low bands = structure, high bands = identity")
    print(f"{'='*60}")

    model.eval()
    n_harmonics = config.n_embd // 2

    test_contexts = {
        'e': [("th", "the"), ("her", "here"), ("lov", "love"),
              ("nam", "name"), ("com", "come"), ("tak", "take"),
              ("giv", "give"), ("mak", "make"), ("hat", "hate"), ("lif", "life")],
        'n': [("ki", "kin"), ("quee", "queen"), ("whe", "when"),
              ("the", "then"), ("i", "in"), ("ca", "can"),
              ("ma", "man"), ("su", "sun"), ("ru", "run"), ("so", "son")],
    }

    for target_char, contexts in test_contexts.items():
        print(f"\n  Character '{target_char}' — per-band divergence at final layer:")

        # Collect final-layer representations
        final_reps = []
        for ctx_text, label in contexts:
            full_text = ctx_text + target_char
            tokens = dataset.encode(full_text).unsqueeze(0).to(config.device)
            with torch.no_grad():
                acts = model.forward_with_all_activations(tokens)
            target_pos = len(full_text) - 1
            rep = acts["final"][0, target_pos, :].cpu().numpy()
            final_reps.append(rep)

        final_reps = np.array(final_reps)

        # Per-band variance across contexts
        band_variance = np.zeros(n_harmonics)
        band_mean_energy = np.zeros(n_harmonics)
        for h in range(n_harmonics):
            ci, si = h * 2, h * 2 + 1
            cos_vals = final_reps[:, ci]
            sin_vals = final_reps[:, si]
            band_variance[h] = np.var(cos_vals) + np.var(sin_vals)
            band_mean_energy[h] = np.mean(cos_vals**2 + sin_vals**2)

        # Normalize: variance relative to energy
        relative_variance = np.zeros(n_harmonics)
        for h in range(n_harmonics):
            if band_mean_energy[h] > 1e-8:
                relative_variance[h] = band_variance[h] / band_mean_energy[h]

        # Report: low bands vs high bands
        low_bands = relative_variance[:16]
        mid_bands = relative_variance[16:40]
        high_bands = relative_variance[40:]

        avg_low = np.mean(low_bands)
        avg_mid = np.mean(mid_bands)
        avg_high = np.mean(high_bands)

        print(f"    Low bands  (1-16):  avg relative variance = {avg_low:.6f}")
        print(f"    Mid bands  (17-40): avg relative variance = {avg_mid:.6f}")
        print(f"    High bands (41-64): avg relative variance = {avg_high:.6f}")

        # Top 10 most context-sensitive bands
        top_bands = np.argsort(relative_variance)[::-1][:10]
        print(f"\n    Most context-sensitive bands:")
        print(f"    {'Band':>6} | {'Rel variance':>13} | {'Region':>8}")
        print(f"    {'-'*6}-+-{'-'*13}-+-{'-'*8}")
        for b in top_bands:
            region = "LOW" if b < 16 else "MID" if b < 40 else "HIGH"
            print(f"    n={b+1:>3} | {relative_variance[b]:>13.6f} | {region:>8}")

        # Top 10 most stable bands (context-INDEPENDENT)
        stable_bands = np.argsort(relative_variance)[:10]
        print(f"\n    Most context-stable bands (character identity preserved):")
        print(f"    {'Band':>6} | {'Rel variance':>13} | {'Region':>8}")
        print(f"    {'-'*6}-+-{'-'*13}-+-{'-'*8}")
        for b in stable_bands:
            region = "LOW" if b < 16 else "MID" if b < 40 else "HIGH"
            print(f"    n={b+1:>3} | {relative_variance[b]:>13.6f} | {region:>8}")


# =============================================================================
# Test 3: Concept formation — does "king" form a unique chord?
# =============================================================================

def test_concept_formation(model, dataset, config):
    """
    Compare the representation after full words vs partial words.
    Does "king" at the 'g' position form a distinct pattern that
    is more than just the 'g' character?

    Also compare: do similar concepts (king/queen, love/hate)
    have more similar representations than unrelated words?
    """
    print(f"\n{'='*60}")
    print(f"  TEST 3: Concept Formation")
    print(f"  Do words form unique composite representations?")
    print(f"{'='*60}")

    model.eval()

    # Word pairs: similar meaning, opposite meaning, unrelated
    word_groups = {
        "royalty": ["king", "queen", "lord", "duke"],
        "emotion": ["love", "hate", "fear", "hope"],
        "action": ["come", "go", "take", "give"],
        "body": ["hand", "head", "heart", "face"],
    }

    # Get representation of last character of each word
    word_reps = {}

    for group_name, words in word_groups.items():
        for word in words:
            # Pad with some context
            context = "the " + word
            tokens = dataset.encode(context).unsqueeze(0).to(config.device)
            with torch.no_grad():
                acts = model.forward_with_all_activations(tokens)
            # Last character of the word
            target_pos = len(context) - 1
            rep = acts["final"][0, target_pos, :].cpu().numpy()
            word_reps[word] = rep

    # Compute similarity matrix
    all_words = []
    for words in word_groups.values():
        all_words.extend(words)

    n_words = len(all_words)
    sim_matrix = np.zeros((n_words, n_words))

    for i in range(n_words):
        for j in range(n_words):
            ri = word_reps[all_words[i]]
            rj = word_reps[all_words[j]]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                sim_matrix[i, j] = np.dot(ri, rj) / (norm_i * norm_j)

    # Print similarity matrix
    print(f"\n  Cosine similarity between word representations (final layer):")
    header = "          " + "".join(f"{w:>8}" for w in all_words)
    print(f"  {header}")
    for i, word in enumerate(all_words):
        row = f"  {word:>8} "
        for j in range(n_words):
            val = sim_matrix[i, j]
            row += f"{val:>8.3f}"
        print(row)

    # Within-group vs between-group similarity
    within_sims = []
    between_sims = []

    group_indices = {}
    idx = 0
    for group_name, words in word_groups.items():
        group_indices[group_name] = list(range(idx, idx + len(words)))
        idx += len(words)

    for gname, indices in group_indices.items():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                within_sims.append(sim_matrix[indices[i], indices[j]])

    for g1, indices1 in group_indices.items():
        for g2, indices2 in group_indices.items():
            if g1 >= g2:
                continue
            for i in indices1:
                for j in indices2:
                    between_sims.append(sim_matrix[i, j])

    avg_within = np.mean(within_sims)
    avg_between = np.mean(between_sims)

    print(f"\n  Within-group avg similarity:  {avg_within:.4f}")
    print(f"  Between-group avg similarity: {avg_between:.4f}")
    print(f"  Difference:                   {avg_within - avg_between:+.4f}")

    if avg_within > avg_between + 0.05:
        print(f"  >> CONCEPTS DETECTED: Same-category words cluster together.")
        print(f"  >> The model composes characters into meaningful representations.")
    elif avg_within > avg_between:
        print(f"  >> WEAK CLUSTERING: Slight tendency for same-category words to cluster.")
    else:
        print(f"  >> NO CLUSTERING: Word category doesn't predict representation similarity.")

    return avg_within, avg_between


# =============================================================================
# Test 4: Layer-by-layer concept emergence
# =============================================================================

def test_concept_emergence(model, dataset, config):
    """
    Track when concepts form. At which layer do same-category words
    start clustering together?
    """
    print(f"\n{'='*60}")
    print(f"  TEST 4: When Do Concepts Emerge?")
    print(f"  Track clustering at each layer")
    print(f"{'='*60}")

    model.eval()

    word_groups = {
        "royalty": ["king", "queen", "lord", "duke"],
        "emotion": ["love", "hate", "fear", "hope"],
        "action": ["come", "go", "take", "give"],
        "body": ["hand", "head", "heart", "face"],
    }

    layers = ["embedding", "embed+pos",
              "layer0_post_attn", "layer0_post_mlp",
              "layer1_post_attn", "layer1_post_mlp",
              "layer2_post_attn", "layer2_post_mlp",
              "layer3_post_attn", "layer3_post_mlp",
              "final"]

    all_words = []
    group_labels = []
    for gname, words in word_groups.items():
        for w in words:
            all_words.append(w)
            group_labels.append(gname)

    print(f"\n  {'Layer':>20} | {'Within-group':>13} | {'Between-group':>14} | {'Gap':>8} | {'Clustering':>11}")
    print(f"  {'-'*20}-+-{'-'*13}-+-{'-'*14}-+-{'-'*8}-+-{'-'*11}")

    for layer in layers:
        # Get representations
        reps = []
        for word in all_words:
            context = "the " + word
            tokens = dataset.encode(context).unsqueeze(0).to(config.device)
            with torch.no_grad():
                acts = model.forward_with_all_activations(tokens)
            target_pos = len(context) - 1
            rep = acts[layer][0, target_pos, :].cpu().numpy()
            reps.append(rep)

        reps = np.array(reps)
        norms = np.linalg.norm(reps, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = reps / norms
        sim_matrix = normalized @ normalized.T

        # Within vs between
        within = []
        between = []
        for i in range(len(all_words)):
            for j in range(i+1, len(all_words)):
                if group_labels[i] == group_labels[j]:
                    within.append(sim_matrix[i, j])
                else:
                    between.append(sim_matrix[i, j])

        avg_w = np.mean(within) if within else 0
        avg_b = np.mean(between) if between else 0
        gap = avg_w - avg_b

        clustering = "YES" if gap > 0.05 else "weak" if gap > 0.01 else "no"
        print(f"  {layer:>20} | {avg_w:>13.4f} | {avg_b:>14.4f} | {gap:>+8.4f} | {clustering:>11}")


# =============================================================================
# Main
# =============================================================================

def main():
    config = Config()

    print(f"{'='*60}")
    print(f"  CONCEPT COMPOSITION TEST")
    print(f"  Characters or chords?")
    print(f"  Device: {config.device}")
    print(f"{'='*60}")

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    model = train_model(config, dataset, mode="harmonic")

    # Run tests
    divergences = test_context_divergence(model, dataset, config)
    test_band_divergence(model, dataset, config)
    concept_within, concept_between = test_concept_formation(model, dataset, config)
    test_concept_emergence(model, dataset, config)

    # Final verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT: Characters or Chords?")
    print(f"{'='*60}")

    # Average divergence at final layer across all characters
    avg_final_div = np.mean([d["final"] for d in divergences.values()])
    avg_emb_div = np.mean([d["embedding"] for d in divergences.values()])

    print(f"\n  Embedding divergence (same char, diff context): {avg_emb_div:.4f}")
    print(f"  Final layer divergence:                         {avg_final_div:.4f}")
    print(f"  Divergence growth:                              {avg_final_div / max(avg_emb_div, 1e-8):.1f}x")

    print(f"\n  Concept clustering gap: {concept_within - concept_between:+.4f}")

    if avg_final_div > 0.1 and concept_within > concept_between:
        print(f"\n  CHORDS. The model composes characters into context-dependent")
        print(f"  representations. The same character plays a different harmonic")
        print(f"  chord depending on what came before it. And those chords cluster")
        print(f"  by meaning — 'king' is closer to 'queen' than to 'love'.")
        print(f"  Concepts live in the interaction, not the individual notes.")
    elif avg_final_div > 0.1:
        print(f"\n  PARTIAL CHORDS. Context modulates representations but")
        print(f"  semantic clustering is weak. The model sees context")
        print(f"  but may not form strong concepts at this scale.")
    else:
        print(f"\n  CHARACTERS. Representations stay largely unchanged by context.")
        print(f"  The model processes characters independently.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
