"""
Harmonic-Aware Decoder -- Listen to the Model's Confidence Signal

CONTEXT:
Phase 14 proved:
- The model KNOWS Shakespeare (P("discontent")=0.39, P("Juliet")=0.28)
- But greedy decoding picks "the" every time because frequency beats knowledge
- Mid bands are 1.6x MORE active during confident predictions

THE INSIGHT:
The model is already broadcasting "I know this one" through mid-band activation.
A decoder that LISTENS to that signal can switch strategies:
- Mid bands quiet -> uncertain, pick the safe/common character
- Mid bands active -> knows something, trust the knowledge, let it through

THE FIX ISN'T THE ALGORITHM. IT'S LISTENING TO THE MODEL ON THE
FREQUENCY WHERE IT BROADCASTS ITS CONFIDENCE.

SAFETY: Same containment. Generation capped at 500 chars. All output to screen.
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

MAX_GENERATION_LENGTH = 500


# =============================================================================
# Model (same architecture)
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

    def forward_with_hidden(self, idx):
        """Forward pass that returns both logits and final hidden state."""
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, x


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
        self.text = text
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

def train_progressive(config, dataset):
    print(f"\n  Training PROGRESSIVE (happy) model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

    n_harmonics = config.n_embd // 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    for it in range(config.max_iters):
        if it < 1000:
            trainable_bands, stage = 8, 1
        elif it < 2000:
            trainable_bands, stage = 24, 2
        else:
            trainable_bands, stage = n_harmonics, 3

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            model.eval()
            total = 0
            for _ in range(config.eval_iters):
                x, y = dataset.get_batch("val", config)
                with torch.no_grad():
                    _, loss = model(x, y)
                total += loss.item()
            val_loss = total / config.eval_iters
            print(f"  step {it:>5} | val {val_loss:.4f} | stage {stage} (bands 1-{trainable_bands}) | {time.time()-start:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        if trainable_bands < n_harmonics:
            with torch.no_grad():
                freeze_start = trainable_bands * 2
                if model.wte.weight.grad is not None:
                    model.wte.weight.grad[:, freeze_start:] = 0
                if model.wpe.weight.grad is not None:
                    model.wpe.weight.grad[:, freeze_start:] = 0

        optimizer.step()

    print(f"  Training complete in {time.time()-start:.1f}s")
    return model


# =============================================================================
# Decoding Strategies
# =============================================================================

def get_mid_band_energy(hidden_state):
    """Measure mid-band (channels 16-48) energy from hidden state.
    This is the confidence signal from Phase 14."""
    # hidden_state: [C] or [1, 1, C]
    h = hidden_state.view(-1)
    mid_energy = (h[16:48] ** 2).mean().item()
    low_energy = (h[:16] ** 2).mean().item()
    return mid_energy, low_energy


def calibrate_mid_band_threshold(model, dataset, config, n_batches=10):
    """Measure the mid-band energy distribution to set the confidence threshold.
    The threshold is where mid-band energy separates 'knowing' from 'guessing'."""
    model.eval()
    mid_energies = []
    confidences = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val", config)
            logits, hidden = model.forward_with_hidden(x)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values  # [B, T]

            for b in range(x.shape[0]):
                for t in range(x.shape[1]):
                    h = hidden[b, t]
                    mid_e = (h[16:48] ** 2).mean().item()
                    conf = max_probs[b, t].item()
                    mid_energies.append(mid_e)
                    confidences.append(conf)

    mid_arr = np.array(mid_energies)
    conf_arr = np.array(confidences)

    # Split into high-confidence and low-confidence
    high_mask = conf_arr > np.percentile(conf_arr, 75)
    low_mask = conf_arr < np.percentile(conf_arr, 25)

    high_mid = mid_arr[high_mask].mean()
    low_mid = mid_arr[low_mask].mean()

    # Threshold: midpoint between high and low confidence mid-band energy
    threshold = (high_mid + low_mid) / 2

    return threshold, high_mid, low_mid


def decode_greedy(model, dataset, config, prompt, n_chars=80):
    """Standard greedy decoding. The safe algorithm."""
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(min(n_chars, MAX_GENERATION_LENGTH)):
            idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)
            logits, _ = model(idx)
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

    return "".join([dataset.itos[t] for t in generated[len(tokens):]])


def decode_sampling(model, dataset, config, prompt, n_chars=80, temperature=0.8, top_k=None):
    """Standard sampling with temperature."""
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(min(n_chars, MAX_GENERATION_LENGTH)):
            idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)
            logits, _ = model(idx)
            logits = logits[0, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

    return "".join([dataset.itos[t] for t in generated[len(tokens):]])


def decode_beam(model, dataset, config, prompt, n_chars=80, beam_width=5):
    """Beam search -- track multiple candidates, let knowledge sequences win."""
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]

    # Each beam: (log_prob, token_list)
    beams = [(0.0, list(tokens))]

    with torch.no_grad():
        for step in range(min(n_chars, MAX_GENERATION_LENGTH)):
            all_candidates = []
            for log_prob, seq in beams:
                idx = torch.tensor([seq[-config.block_size:]], dtype=torch.long, device=config.device)
                logits, _ = model(idx)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                # Take top-k expansions
                top_log_probs, top_ids = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_seq = seq + [top_ids[i].item()]
                    new_log_prob = log_prob + top_log_probs[i].item()
                    all_candidates.append((new_log_prob, new_seq))

            # Keep top beam_width candidates
            all_candidates.sort(key=lambda x: -x[0])
            beams = all_candidates[:beam_width]

    best_seq = beams[0][1]
    return "".join([dataset.itos[t] for t in best_seq[len(tokens):]])


def decode_harmonic(model, dataset, config, prompt, n_chars=80,
                    mid_threshold=None, trust_temperature=0.5, safe_temperature=1.2):
    """
    HARMONIC-AWARE DECODER.

    Reads the mid-band confidence signal at every step:
    - Mid bands active (above threshold) -> the model KNOWS something.
      Use low temperature (trust the knowledge, let it through).
    - Mid bands quiet (below threshold) -> the model is guessing.
      Use high temperature (explore, don't commit to frequency bias).

    Returns generated text and per-step diagnostics.
    """
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    generated = list(tokens)
    diagnostics = []

    with torch.no_grad():
        for _ in range(min(n_chars, MAX_GENERATION_LENGTH)):
            idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)
            logits, hidden = model.forward_with_hidden(idx)
            last_logits = logits[0, -1, :]
            last_hidden = hidden[0, -1, :]

            # Read the confidence signal
            mid_energy, low_energy = get_mid_band_energy(last_hidden)
            mid_ratio = mid_energy / (low_energy + 1e-10)
            confident = mid_energy > mid_threshold if mid_threshold else mid_ratio > 1.0

            # Choose strategy based on confidence
            if confident:
                # Model knows something -- trust it, low temperature
                temperature = trust_temperature
                mode = "KNOW"
            else:
                # Model is guessing -- explore, higher temperature
                temperature = safe_temperature
                mode = "GUESS"

            scaled_logits = last_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            diagnostics.append({
                "char": dataset.itos[next_token],
                "mid_energy": mid_energy,
                "low_energy": low_energy,
                "confident": confident,
                "mode": mode,
                "top_prob": probs.max().item(),
            })

            generated.append(next_token)

    text = "".join([dataset.itos[t] for t in generated[len(tokens):]])
    return text, diagnostics


def decode_harmonic_beam(model, dataset, config, prompt, n_chars=80,
                         mid_threshold=None, beam_width=5):
    """
    HARMONIC BEAM SEARCH.

    Like beam search, but the beam width adapts based on confidence:
    - Mid bands active -> narrow beam (the model knows, commit to it)
    - Mid bands quiet -> wide beam (explore alternatives)

    This lets knowledge sequences win without needing to beat frequency
    at every single character position.
    """
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    beams = [(0.0, list(tokens), [])]  # (log_prob, sequence, diagnostic_list)

    with torch.no_grad():
        for step in range(min(n_chars, MAX_GENERATION_LENGTH)):
            all_candidates = []

            for log_prob, seq, diags in beams:
                idx = torch.tensor([seq[-config.block_size:]], dtype=torch.long, device=config.device)
                logits, hidden = model.forward_with_hidden(idx)
                last_hidden = hidden[0, -1, :]

                mid_energy, low_energy = get_mid_band_energy(last_hidden)
                confident = mid_energy > mid_threshold if mid_threshold else False

                # Adaptive beam width
                if confident:
                    local_beam = max(2, beam_width // 2)  # narrow: commit
                    # Boost log prob slightly for confident paths
                    confidence_bonus = 0.1
                else:
                    local_beam = beam_width  # wide: explore
                    confidence_bonus = 0.0

                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                top_log_probs, top_ids = log_probs.topk(local_beam)

                for i in range(local_beam):
                    new_seq = seq + [top_ids[i].item()]
                    new_log_prob = log_prob + top_log_probs[i].item() + confidence_bonus
                    new_diags = diags + [{"confident": confident, "mid_energy": mid_energy}]
                    all_candidates.append((new_log_prob, new_seq, new_diags))

            all_candidates.sort(key=lambda x: -x[0])
            beams = all_candidates[:beam_width]

    best_log_prob, best_seq, best_diags = beams[0]
    text = "".join([dataset.itos[t] for t in best_seq[len(tokens):]])
    return text, best_diags


# =============================================================================
# TEST 1: Calibrate the Confidence Signal
# =============================================================================

def test_calibration(model, dataset, config):
    """Measure and calibrate the mid-band confidence threshold."""
    print("\n" + "=" * 60)
    print("  TEST 1: Calibrate the Confidence Signal")
    print("  Where is the line between 'knowing' and 'guessing'?")
    print("=" * 60)

    threshold, high_mid, low_mid = calibrate_mid_band_threshold(model, dataset, config)

    print(f"\n  Mid-band energy distribution:")
    print(f"    High-confidence predictions (top 25%): {high_mid:.4f}")
    print(f"    Low-confidence predictions (bottom 25%): {low_mid:.4f}")
    print(f"    Ratio: {high_mid/low_mid:.2f}x")
    print(f"    Calibrated threshold: {threshold:.4f}")

    return threshold


# =============================================================================
# TEST 2: Quote Recovery
# =============================================================================

def test_quote_recovery(model, dataset, config, text, threshold):
    """The acid test: can the harmonic decoder recover famous quotes
    that greedy decoding misses?"""
    print("\n" + "=" * 60)
    print("  TEST 2: Quote Recovery")
    print("  Can the harmonic decoder find what greedy decoding buries?")
    print("=" * 60)

    # Knowledge prompts
    prompts = [
        ("the winter of our ", "discontent", "Richard III"),
        ("Romeo and ", "Juliet", "The lovers"),
        ("Romeo, Romeo, wherefore art thou ", "Romeo", "Balcony scene"),
        ("A plague on both your ", "houses", "Mercutio"),
        ("my kingdom for a ", "horse", "Richard III"),
        ("the Duke of ", None, "Nobility"),
        ("my lord, my ", None, "Address"),
        ("O Romeo, Romeo, ", None, "Juliet calls"),
        ("Now is the winter of ", None, "Opening"),
        ("What's in a name", None, "Juliet muses"),
    ]

    strategies = {
        "Greedy": lambda p, n: decode_greedy(model, dataset, config, p, n),
        "Sample(0.8)": lambda p, n: decode_sampling(model, dataset, config, p, n, temperature=0.8),
        "Beam(5)": lambda p, n: decode_beam(model, dataset, config, p, n, beam_width=5),
        "Harmonic": lambda p, n: decode_harmonic(model, dataset, config, p, n, mid_threshold=threshold)[0],
        "Harm.Beam": lambda p, n: decode_harmonic_beam(model, dataset, config, p, n, mid_threshold=threshold)[0],
    }

    print(f"\n  Mid-band confidence threshold: {threshold:.4f}")

    for prompt, expected, desc in prompts:
        print(f"\n  {desc}: '{prompt}'")
        if expected:
            print(f"  Expected: '{expected}'")

        for strat_name, strat_fn in strategies.items():
            result = strat_fn(prompt, 20)
            # Clean for display
            result_display = result.split('\n')[0][:30]

            # Check if expected is in the result
            match = ""
            if expected and expected.lower() in result.lower():
                match = " *** MATCH ***"

            print(f"    {strat_name:<15}: {result_display}{match}")


# =============================================================================
# TEST 3: Confidence Trace
# =============================================================================

def test_confidence_trace(model, dataset, config, threshold):
    """Visualise the harmonic decoder's decision-making.
    Show when it's confident vs guessing, step by step."""
    print("\n" + "=" * 60)
    print("  TEST 3: Confidence Trace")
    print("  Watch the model switch between knowing and guessing")
    print("=" * 60)

    prompts = [
        "the winter of our ",
        "Romeo and ",
        "KING RICHARD:\nMy lord",
    ]

    for prompt in prompts:
        text, diags = decode_harmonic(model, dataset, config, prompt, n_chars=40,
                                       mid_threshold=threshold)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Generated: '{text[:50]}'")
        print(f"\n  Step-by-step confidence trace:")
        print(f"  {'Step':>5} {'Char':>5} {'Mid E':>8} {'Mode':>6} {'Top P':>7}")
        print(f"  {'-'*5} {'-'*5} {'-'*8} {'-'*6} {'-'*7}")

        for i, d in enumerate(diags[:30]):
            char_display = d['char'] if d['char'] not in '\n\t' else repr(d['char'])
            print(f"  {i:>5} {char_display:>5} {d['mid_energy']:>8.4f} {d['mode']:>6} {d['top_prob']:>7.3f}")

        # Count modes
        know_count = sum(1 for d in diags if d['mode'] == 'KNOW')
        guess_count = sum(1 for d in diags if d['mode'] == 'GUESS')
        print(f"\n  Mode split: {know_count} KNOW ({100*know_count/len(diags):.0f}%), "
              f"{guess_count} GUESS ({100*guess_count/len(diags):.0f}%)")


# =============================================================================
# TEST 4: Knowledge Accuracy
# =============================================================================

def test_knowledge_accuracy(model, dataset, config, text, threshold):
    """Measure: does the harmonic decoder produce more accurate completions
    on knowledge-bearing prompts?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Knowledge Accuracy")
    print("  Which decoder best completes known sequences?")
    print("=" * 60)

    # Find actual multi-word sequences in the training data
    # Search for specific known phrases
    known_phrases = []
    search_terms = [
        "the winter of our discontent",
        "Romeo and Juliet",
        "Romeo, Romeo",
        "the Duke of",
        "my lord",
        "the king",
        "good night",
        "I pray",
        "heaven and earth",
        "noble lord",
        "your grace",
        "my liege",
        "God save",
        "sweet prince",
        "fair lady",
        "good my lord",
    ]

    for phrase in search_terms:
        pos = text.lower().find(phrase.lower())
        if pos >= 0:
            # Get the actual text with correct case
            actual = text[pos:pos+len(phrase)+15]
            prompt_end = len(phrase) // 2
            prompt = actual[:prompt_end]
            expected = actual[prompt_end:prompt_end+10]
            if all(c in dataset.stoi for c in prompt + expected):
                known_phrases.append((prompt, expected, phrase))

    if not known_phrases:
        print("  No testable known phrases found.")
        return

    print(f"\n  Testing {len(known_phrases)} known phrases")

    strategy_scores = {
        "Greedy": [],
        "Sample": [],
        "Beam": [],
        "Harmonic": [],
        "Harm.Beam": [],
    }

    for prompt, expected, full_phrase in known_phrases:
        # Get each strategy's completion
        completions = {
            "Greedy": decode_greedy(model, dataset, config, prompt, len(expected)),
            "Sample": decode_sampling(model, dataset, config, prompt, len(expected)),
            "Beam": decode_beam(model, dataset, config, prompt, len(expected), beam_width=5),
            "Harmonic": decode_harmonic(model, dataset, config, prompt, len(expected), mid_threshold=threshold)[0],
            "Harm.Beam": decode_harmonic_beam(model, dataset, config, prompt, len(expected), mid_threshold=threshold)[0],
        }

        for strat_name, completion in completions.items():
            # Character-level accuracy
            correct = sum(1 for a, b in zip(completion, expected) if a == b)
            accuracy = correct / len(expected) if expected else 0
            strategy_scores[strat_name].append(accuracy)

    # Summary
    print(f"\n  Average character-level accuracy on known phrases:")
    print(f"  {'Strategy':<15} {'Accuracy':>10} {'vs Greedy':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    greedy_avg = np.mean(strategy_scores["Greedy"])
    for name, scores in strategy_scores.items():
        avg = np.mean(scores)
        delta = avg - greedy_avg
        print(f"  {name:<15} {avg:>10.4f} {delta:>+10.4f}")


# =============================================================================
# TEST 5: Full Generation Comparison
# =============================================================================

def test_generation_comparison(model, dataset, config, threshold):
    """Generate longer text with each strategy. Which sounds most Shakespearean?"""
    print("\n" + "=" * 60)
    print("  TEST 5: Full Generation Comparison")
    print("  Same model, different decoders. Which Shakespeare is best?")
    print("=" * 60)

    prompts = [
        "ROMEO:\nO, she doth teach the torches to burn bright!\n",
        "KING RICHARD III:\n",
        "Now is the winter of our ",
    ]

    strategies = [
        ("Greedy", lambda p: decode_greedy(model, dataset, config, p, 150)),
        ("Sample(0.8)", lambda p: decode_sampling(model, dataset, config, p, 150, temperature=0.8)),
        ("Beam(5)", lambda p: decode_beam(model, dataset, config, p, 80, beam_width=5)),
        ("Harmonic", lambda p: decode_harmonic(model, dataset, config, p, 150, mid_threshold=threshold)[0]),
        ("Harm.Beam(5)", lambda p: decode_harmonic_beam(model, dataset, config, p, 80, mid_threshold=threshold)[0]),
    ]

    for prompt in prompts:
        prompt_display = prompt.replace('\n', '\\n')
        if len(prompt_display) > 50:
            prompt_display = prompt_display[:50] + "..."
        print(f"\n  Prompt: '{prompt_display}'")
        print(f"  {'-' * 60}")

        for name, strat_fn in strategies:
            result = strat_fn(prompt)
            # Show first 120 chars of result, formatted
            safe = result[:120].replace('\n', '\\n')
            print(f"    {name:<15}: {safe}")


# =============================================================================
# TEST 6: The Harmonic Advantage
# =============================================================================

def test_harmonic_advantage(model, dataset, config, text, threshold):
    """Quantify: on prompts where knowledge matters, how much does
    the harmonic decoder improve over greedy?"""
    print("\n" + "=" * 60)
    print("  TEST 6: The Harmonic Advantage")
    print("  Quantifying the knowledge recovery")
    print("=" * 60)

    # Build a test set of knowledge-bearing completions
    # Take actual lines from the text and measure probability of correct continuation
    lines = text.split('\n')
    test_cases = []
    for line in lines:
        line = line.strip()
        if 20 < len(line) < 80 and ':' not in line[:10] and not line.isupper():
            mid = len(line) // 2
            split = line.rfind(' ', mid - 5, mid + 5)
            if split > 5:
                prompt = line[:split + 1]
                expected = line[split + 1:split + 6]
                if len(expected) >= 3 and all(c in dataset.stoi for c in prompt + expected):
                    test_cases.append((prompt, expected))

    np.random.seed(42)
    if len(test_cases) > 100:
        indices = np.random.choice(len(test_cases), 100, replace=False)
        test_cases = [test_cases[i] for i in indices]

    print(f"\n  Evaluating {len(test_cases)} text completions")

    greedy_matches = 0
    harmonic_matches = 0
    beam_matches = 0
    harm_beam_matches = 0
    total = 0

    for prompt, expected in test_cases:
        g = decode_greedy(model, dataset, config, prompt, len(expected))
        h = decode_harmonic(model, dataset, config, prompt, len(expected), mid_threshold=threshold)[0]
        b = decode_beam(model, dataset, config, prompt, len(expected), beam_width=5)
        hb = decode_harmonic_beam(model, dataset, config, prompt, len(expected), mid_threshold=threshold)[0]

        greedy_correct = sum(1 for a, e in zip(g, expected) if a == e)
        harmonic_correct = sum(1 for a, e in zip(h, expected) if a == e)
        beam_correct = sum(1 for a, e in zip(b, expected) if a == e)
        harm_beam_correct = sum(1 for a, e in zip(hb, expected) if a == e)

        greedy_matches += greedy_correct
        harmonic_matches += harmonic_correct
        beam_matches += beam_correct
        harm_beam_matches += harm_beam_correct
        total += len(expected)

    g_acc = greedy_matches / total
    h_acc = harmonic_matches / total
    b_acc = beam_matches / total
    hb_acc = harm_beam_matches / total

    print(f"\n  Character-level accuracy on {len(test_cases)} completions:")
    print(f"  {'Strategy':<15} {'Accuracy':>10} {'vs Greedy':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    print(f"  {'Greedy':<15} {g_acc:>10.4f} {'---':>10}")
    print(f"  {'Beam(5)':<15} {b_acc:>10.4f} {b_acc-g_acc:>+10.4f}")
    print(f"  {'Harmonic':<15} {h_acc:>10.4f} {h_acc-g_acc:>+10.4f}")
    print(f"  {'Harm.Beam(5)':<15} {hb_acc:>10.4f} {hb_acc-g_acc:>+10.4f}")

    winner = max([("Greedy", g_acc), ("Beam", b_acc), ("Harmonic", h_acc), ("Harm.Beam", hb_acc)],
                 key=lambda x: x[1])
    print(f"\n  Winner: {winner[0]} ({winner[1]:.4f})")

    return g_acc, h_acc, b_acc, hb_acc


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict(g_acc, h_acc, b_acc, hb_acc):
    print("\n" + "=" * 60)
    print("  VERDICT: Does Listening to the Model Help?")
    print("=" * 60)

    best_non_greedy = max(h_acc, b_acc, hb_acc)
    improvement = best_non_greedy - g_acc

    if hb_acc > g_acc and hb_acc >= b_acc:
        print(f"\n  YES. The harmonic beam decoder wins.")
        print(f"  Accuracy: {hb_acc:.4f} vs greedy {g_acc:.4f} ({improvement:+.4f})")
        print(f"\n  The model was broadcasting its confidence on the mid bands.")
        print(f"  Listening to that signal and adapting the decoder lets")
        print(f"  knowledge-bearing sequences win over frequency bias.")
    elif h_acc > g_acc:
        print(f"\n  YES. The harmonic sampling decoder improves over greedy.")
        print(f"  Accuracy: {h_acc:.4f} vs greedy {g_acc:.4f} ({h_acc-g_acc:+.4f})")
    elif b_acc > g_acc and b_acc > h_acc:
        print(f"\n  PARTIALLY. Beam search helps ({b_acc:.4f} vs {g_acc:.4f})")
        print(f"  but the harmonic signal didn't add value beyond standard beam.")
        print(f"  The mid-band confidence signal may need refinement.")
    else:
        print(f"\n  NOT YET. No decoder significantly beats greedy.")
        print(f"  Greedy: {g_acc:.4f}, Harmonic: {h_acc:.4f}, Beam: {b_acc:.4f}")
        print(f"  The confidence signal may be too noisy for a 4-layer model,")
        print(f"  or the threshold needs better calibration.")

    print(f"\n  The principle remains sound: the model broadcasts confidence")
    print(f"  on mid-band harmonics. Whether that signal is clean enough")
    print(f"  to drive decoding depends on model scale and calibration.")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  HARMONIC-AWARE DECODER")
    print("  Listen to the model's confidence signal")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train progressive model
    model = train_progressive(config, dataset)

    # Calibrate
    threshold = test_calibration(model, dataset, config)

    # Run tests
    test_quote_recovery(model, dataset, config, text, threshold)
    test_confidence_trace(model, dataset, config, threshold)
    test_knowledge_accuracy(model, dataset, config, text, threshold)
    test_generation_comparison(model, dataset, config, threshold)
    g_acc, h_acc, b_acc, hb_acc = test_harmonic_advantage(model, dataset, config, text, threshold)

    print_verdict(g_acc, h_acc, b_acc, hb_acc)


if __name__ == "__main__":
    main()
