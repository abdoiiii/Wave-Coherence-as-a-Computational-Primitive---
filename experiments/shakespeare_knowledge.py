"""
Shakespeare Knowledge Test -- Let the Model Tell Us What It Knows

CONTEXT:
The model was trained on 1.1M characters of Shakespeare. It has built a rich
internal representation -- Phase 7 showed it forms word-level chords, Phase 12
showed it has 22% internal structure that doesn't map to next-char prediction,
Phase 13 showed the bottleneck is the TASK, not the instrument.

We've only ever asked: "what's the next letter?"

That's like asking a Shakespeare scholar to spell words.

NOW WE ASK REAL QUESTIONS:
- Can you complete famous quotes?
- Do you know who Romeo loves?
- Can you speak in a character's voice?
- Does the progressive model know more than baseline?
- Does the richer expression head unlock more knowledge?

The model has been waiting to tell us. Let's listen.

SAFETY: Same as Phase 13. Generation capped at 500 chars. Norm monitoring.
No disk writes. Everything printed to screen.
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
# Model (same as other phases)
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

    def get_hidden(self, idx):
        B, T = idx.size()
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(T)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def freeze_internals(self):
        for name, param in self.named_parameters():
            if not name.startswith("lm_head"):
                param.requires_grad = False


# Expression heads from Phase 13
class DeepHead(nn.Module):
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
        h = self.ln(x + h)
        return self.proj(h)


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


def train_model(config, dataset, progressive=False):
    name = "PROGRESSIVE (happy)" if progressive else "BASELINE"
    print(f"\n  Training {name} model...")
    model = HarmonicGPT(config, dataset.vocab_size).to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} trainable parameters")

    n_harmonics = config.n_embd // 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start = time.time()

    for it in range(config.max_iters):
        if progressive:
            if it < 1000:
                trainable_bands, stage = 8, 1
            elif it < 2000:
                trainable_bands, stage = 24, 2
            else:
                trainable_bands, stage = n_harmonics, 3
        else:
            trainable_bands, stage = n_harmonics, 0

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            val_loss = eval_loss(model, dataset, config)
            if progressive:
                print(f"  step {it:>5} | val {val_loss:.4f} | stage {stage} (bands 1-{trainable_bands}) | {time.time()-start:.1f}s")
            else:
                print(f"  step {it:>5} | val {val_loss:.4f} | {time.time()-start:.1f}s")
            model.train()

        x, y = dataset.get_batch("train", config)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()

        if progressive and trainable_bands < n_harmonics:
            with torch.no_grad():
                freeze_start = trainable_bands * 2
                if model.wte.weight.grad is not None:
                    model.wte.weight.grad[:, freeze_start:] = 0
                if model.wpe.weight.grad is not None:
                    model.wpe.weight.grad[:, freeze_start:] = 0

        optimizer.step()

    print(f"  {name} training complete in {time.time()-start:.1f}s")
    return model


def train_deep_head(model, dataset, config, n_steps=2000):
    """Train a Deep (violin) head with frozen internals."""
    model.freeze_internals()
    head = DeepHead(config.n_embd, dataset.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)

    model.eval()
    head.train()
    for step in range(n_steps):
        x, y = dataset.get_batch("train", config)
        with torch.no_grad():
            h = model.get_hidden(x)
        logits = head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

    head.eval()
    return head


# =============================================================================
# Generation Utilities
# =============================================================================

def generate(model, dataset, config, prompt, n_chars=60, head=None, greedy=False):
    """Generate text from a prompt. Optionally use an external head."""
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(min(n_chars, MAX_GENERATION_LENGTH)):
            idx = torch.tensor([generated[-config.block_size:]], dtype=torch.long, device=config.device)
            if head is not None:
                h = model.get_hidden(idx)
                logits = head(h)
            else:
                logits, _ = model(idx)
            logits = logits[0, -1, :]

            if greedy:
                next_token = logits.argmax().item()
            else:
                probs = F.softmax(logits / 0.8, dim=-1)  # slight temperature for variety
                next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

    return "".join([dataset.itos[t] for t in generated])


def get_continuation_probability(model, dataset, config, prompt, expected, head=None):
    """Get the probability the model assigns to a specific continuation."""
    model.eval()
    tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]

    total_log_prob = 0
    n_chars = 0

    with torch.no_grad():
        for expected_char in expected:
            if expected_char not in dataset.stoi:
                continue
            idx = torch.tensor([tokens[-config.block_size:]], dtype=torch.long, device=config.device)
            if head is not None:
                h = model.get_hidden(idx)
                logits = head(h)
            else:
                logits, _ = model(idx)
            logits = logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)

            expected_id = dataset.stoi[expected_char]
            prob = probs[expected_id].item()
            total_log_prob += math.log(prob + 1e-10)
            n_chars += 1

            tokens.append(expected_id)

    avg_log_prob = total_log_prob / n_chars if n_chars > 0 else float('-inf')
    return math.exp(avg_log_prob), n_chars  # geometric mean probability


# =============================================================================
# Find Actual Quotes in Training Data
# =============================================================================

def find_quotes_in_data(text):
    """Find famous Shakespeare passages that are actually in the training data."""
    quotes = []

    # Search for known phrases and extract context
    search_pairs = [
        # (search_string, prompt_end, expected_continuation, description)
        ("to be or not to be", "to be or not to ", "be", "Hamlet's soliloquy"),
        ("a horse, a horse", "a horse, a horse, my kingdom for a ", "horse", "Richard III's plea"),
        ("Romeo, Romeo", "Romeo, Romeo, wherefore art thou ", "Romeo", "Juliet's balcony"),
        ("Et tu, Brute", "Et tu, ", "Brute", "Caesar's last words"),
        ("Friends, Romans, countrymen", "Friends, Romans, countrymen, lend me your ", "ears", "Antony's speech"),
        ("the winter of our discontent", "the winter of our ", "discontent", "Richard III opening"),
        ("What's in a name", "What's in a name", "?", "Juliet on names"),
        ("All that glitters", "All that glitters is not ", "gold", "Merchant of Venice"),
        ("brevity is the soul of wit", "brevity is the soul of ", "wit", "Polonius"),
        ("Parting is such sweet sorrow", "Parting is such sweet ", "sorrow", "Romeo and Juliet"),
    ]

    for search, prompt, expected, desc in search_pairs:
        # Case-insensitive search
        lower_text = text.lower()
        pos = lower_text.find(search.lower())
        if pos >= 0:
            # Get actual context from the text
            context_start = max(0, pos - 80)
            context_end = min(len(text), pos + len(search) + 40)
            actual_context = text[context_start:context_end]
            quotes.append({
                "search": search,
                "prompt": prompt,
                "expected": expected,
                "desc": desc,
                "found": True,
                "context": actual_context,
                "position": pos,
            })
        else:
            quotes.append({
                "search": search,
                "prompt": prompt,
                "expected": expected,
                "desc": desc,
                "found": False,
                "context": None,
                "position": -1,
            })

    return quotes


def find_character_patterns(text):
    """Find character names and their typical speech patterns."""
    import re
    characters = {}
    # Find lines like "CHARACTER_NAME:\n"
    pattern = r'\n([A-Z][A-Z ]+):\n'
    for match in re.finditer(pattern, text):
        name = match.group(1).strip()
        pos = match.end()
        # Get the next line (their speech)
        next_newline = text.find('\n', pos)
        if next_newline > pos:
            speech = text[pos:next_newline]
            if name not in characters:
                characters[name] = []
            characters[name].append(speech)
    return characters


# =============================================================================
# TEST 1: Quote Completion
# =============================================================================

def test_quote_completion(models, dataset, config, text):
    """Can the model complete famous Shakespeare quotes?"""
    print("\n" + "=" * 60)
    print("  TEST 1: Quote Completion")
    print("  Does the model know the famous lines?")
    print("=" * 60)

    quotes = find_quotes_in_data(text)

    # Show which quotes are in the data
    found = [q for q in quotes if q["found"]]
    missing = [q for q in quotes if not q["found"]]
    print(f"\n  Quotes found in training data: {len(found)}/{len(quotes)}")
    for q in found:
        print(f"    [FOUND] {q['desc']}: '{q['search']}'")
    for q in missing:
        print(f"    [MISSING] {q['desc']}: '{q['search']}'")

    if not found:
        print("\n  No famous quotes found in training data.")
        print("  Creating prompts from actual text patterns instead...")
        return test_text_completion(models, dataset, config, text)

    # Test each model on found quotes
    print(f"\n  {'Quote':<30} {'Expected':>10}", end="")
    for model_name in models:
        print(f" {'P(' + model_name[:8] + ')':>12}", end="")
    print(f" {'Greedy completions':>20}")
    print(f"  {'-'*30} {'-'*10}", end="")
    for _ in models:
        print(f" {'-'*12}", end="")
    print(f" {'-'*20}")

    for q in found:
        print(f"  {q['desc']:<30} {q['expected']:>10}", end="")
        for model_name, (model, head) in models.items():
            prob, _ = get_continuation_probability(model, dataset, config, q['prompt'], q['expected'], head)
            print(f" {prob:>12.4f}", end="")

        # Show greedy completion from first model
        first_model_name = list(models.keys())[0]
        model, head = models[first_model_name]
        completion = generate(model, dataset, config, q['prompt'], n_chars=len(q['expected'])+5, head=head, greedy=True)
        actual = completion[len(q['prompt']):len(q['prompt'])+len(q['expected'])+5]
        print(f" -> {actual}")

    return found


# =============================================================================
# TEST 2: Text Completion from Actual Data
# =============================================================================

def test_text_completion(models, dataset, config, text):
    """Use actual lines from the text as completion tests."""
    print("\n" + "=" * 60)
    print("  TEST 2: Text Completion")
    print("  Complete actual lines from Shakespeare")
    print("=" * 60)

    # Find interesting lines in the training data
    lines = text.split('\n')
    test_prompts = []

    for line in lines:
        line = line.strip()
        if len(line) > 30 and not line.isupper() and ':' not in line[:15]:
            # Split at a natural point (space near middle)
            mid = len(line) // 2
            split_pos = line.rfind(' ', mid - 10, mid + 10)
            if split_pos > 10:
                prompt = line[:split_pos + 1]
                expected = line[split_pos + 1:split_pos + 11]  # next 10 chars
                if len(expected) >= 5 and all(c in dataset.stoi for c in prompt + expected):
                    test_prompts.append((prompt, expected))

    # Take a diverse sample
    np.random.seed(42)
    if len(test_prompts) > 20:
        indices = np.random.choice(len(test_prompts), 20, replace=False)
        test_prompts = [test_prompts[i] for i in indices]

    print(f"\n  Testing {len(test_prompts)} line completions")
    print(f"\n  {'Prompt (last 30 chars)':<35} {'Expected':>12}", end="")
    for model_name in models:
        print(f" {'P(' + model_name[:8] + ')':>12}", end="")
    print()
    print(f"  {'-'*35} {'-'*12}", end="")
    for _ in models:
        print(f" {'-'*12}", end="")
    print()

    model_scores = {name: [] for name in models}

    for prompt, expected in test_prompts:
        display_prompt = prompt[-30:] if len(prompt) > 30 else prompt
        display_prompt = display_prompt.replace('\n', '\\n')
        print(f"  ...{display_prompt:<32} {expected:>12}", end="")

        for model_name, (model, head) in models.items():
            prob, _ = get_continuation_probability(model, dataset, config, prompt, expected, head)
            model_scores[model_name].append(prob)
            print(f" {prob:>12.4f}", end="")
        print()

    # Summary
    print(f"\n  Average continuation probability:")
    for model_name, scores in model_scores.items():
        avg = np.mean(scores)
        std = np.std(scores)
        print(f"    {model_name:<25} {avg:.4f} +/- {std:.4f}")

    return model_scores


# =============================================================================
# TEST 3: Character Voice
# =============================================================================

def test_character_voice(models, dataset, config, text):
    """Does the model know how different characters speak?"""
    print("\n" + "=" * 60)
    print("  TEST 3: Character Voice")
    print("  Does the model know how each character speaks?")
    print("=" * 60)

    characters = find_character_patterns(text)

    # Find the most frequent characters
    char_counts = {name: len(lines) for name, lines in characters.items()}
    top_chars = sorted(char_counts, key=char_counts.get, reverse=True)[:8]

    print(f"\n  Top characters by line count:")
    for name in top_chars:
        print(f"    {name}: {char_counts[name]} lines")

    # For each character, prompt with "CHARACTER:\n" and see what the model generates
    print(f"\n  Character voice test (greedy generation):")
    for char_name in top_chars[:6]:
        prompt = f"{char_name}:\n"
        print(f"\n  {char_name}:")

        for model_name, (model, head) in models.items():
            text_out = generate(model, dataset, config, prompt, n_chars=100, head=head, greedy=False)
            # Get just the generated part
            speech = text_out[len(prompt):]
            # Take first two lines
            speech_lines = speech.split('\n')
            first_line = speech_lines[0] if speech_lines else ""
            safe = first_line[:80]
            print(f"    {model_name:<22}: {safe}")

        # Show an actual line for comparison
        if char_name in characters and characters[char_name]:
            actual = characters[char_name][0][:80]
            print(f"    {'(actual)':22}: {actual}")


# =============================================================================
# TEST 4: Relationship Knowledge
# =============================================================================

def test_relationship_knowledge(models, dataset, config, text):
    """Does the model understand character relationships?"""
    print("\n" + "=" * 60)
    print("  TEST 4: Relationship Knowledge")
    print("  Does the model know who's who?")
    print("=" * 60)

    # Relationship prompts -- things the model should know from context
    relationship_prompts = [
        ("Romeo and ", "Juliet", "The lovers"),
        ("the Duke of ", None, "Nobility title"),
        ("my lord, my ", None, "Forms of address"),
        ("O Romeo, Romeo, ", None, "Juliet calling"),
        ("the king and ", None, "Royal pairs"),
        ("my sword and ", None, "Warrior items"),
        ("heaven and ", None, "Opposing pairs"),
        ("love and ", None, "Emotional pairs"),
        ("to live or ", None, "Life/death"),
        ("the sun and ", None, "Celestial pairs"),
    ]

    # Check which prompts have the expected text in training data
    print(f"\n  {'Prompt':<25} {'Expected':>10}", end="")
    for model_name in models:
        print(f"  {model_name[:15]:>18}", end="")
    print()
    print(f"  {'-'*25} {'-'*10}", end="")
    for _ in models:
        print(f"  {'-'*18}", end="")
    print()

    for prompt, expected, desc in relationship_prompts:
        if expected:
            prob_str = f"'{expected}'"
        else:
            prob_str = "?"
        print(f"  {prompt:<25} {prob_str:>10}", end="")

        for model_name, (model, head) in models.items():
            completion = generate(model, dataset, config, prompt, n_chars=15, head=head, greedy=True)
            actual = completion[len(prompt):len(prompt)+15]
            # Clean for display
            actual = actual.split('\n')[0][:15]
            print(f"  {actual:>18}", end="")
        print()

    # Specific probability test for "Romeo and Juliet"
    print(f"\n  Probability test: 'Romeo and ' -> ?")
    for model_name, (model, head) in models.items():
        prob_juliet, _ = get_continuation_probability(model, dataset, config, "Romeo and ", "Juliet", head)
        # Also check top 5 next characters
        tokens = [dataset.stoi[c] for c in "Romeo and " if c in dataset.stoi]
        idx = torch.tensor([tokens], dtype=torch.long, device=config.device)
        with torch.no_grad():
            if head is not None:
                h = model.get_hidden(idx)
                logits = head(h)
            else:
                logits, _ = model(idx)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = probs.topk(5)

        top_chars = [(dataset.itos[tid.item()], tp.item()) for tid, tp in zip(top_ids, top_probs)]
        top_str = ", ".join([f"'{c}'({p:.3f})" for c, p in top_chars])
        print(f"    {model_name:<22}: P('Juliet')={prob_juliet:.4f}, top: {top_str}")


# =============================================================================
# TEST 5: Knowledge Depth -- Semantic Probing
# =============================================================================

def test_knowledge_depth(models, dataset, config, text):
    """Probe whether the model has absorbed deeper patterns.
    Test with increasingly specific prompts."""
    print("\n" + "=" * 60)
    print("  TEST 5: Knowledge Depth")
    print("  How deep does the model's Shakespeare knowledge go?")
    print("=" * 60)

    # Levels of knowledge:
    # L1: Character statistics (most common next char)
    # L2: Word-level patterns (common words)
    # L3: Phrase-level patterns (common phrases)
    # L4: Character-specific knowledge (how a specific character speaks)
    # L5: Plot-level knowledge (what happens in the story)

    depth_tests = [
        # (prompt, expected_contains, level, description)
        ("the ", None, 1, "Common word continuation"),
        ("thou ", None, 1, "Archaic pronoun"),
        ("What is ", None, 2, "Question formation"),
        ("I will not ", None, 2, "Negation pattern"),
        ("ROMEO:\nO, ", None, 3, "Character voice (Romeo)"),
        ("JULIET:\nO ", None, 3, "Character voice (Juliet)"),
        ("Hath not a ", None, 3, "Shylock-style rhetoric"),
        ("Now is the winter of ", None, 4, "Specific quote start"),
        ("A plague on both your ", None, 4, "Mercutio's curse"),
        ("If music be the food of ", None, 4, "Twelfth Night"),
    ]

    print(f"\n  Depth probes (greedy generation):")

    for prompt, expected, level, desc in depth_tests:
        print(f"\n  L{level} - {desc}")
        print(f"  Prompt: '{prompt}'")

        for model_name, (model, head) in models.items():
            completion = generate(model, dataset, config, prompt, n_chars=40, head=head, greedy=True)
            generated = completion[len(prompt):]
            # First line only
            generated = generated.split('\n')[0][:40]
            print(f"    {model_name:<22}: {generated}")


# =============================================================================
# TEST 6: The Hidden Knowledge Test
# =============================================================================

def test_hidden_knowledge(models, dataset, config):
    """Is the 22% trapped structure involved when the model answers correctly?
    Compare hidden state norms and patterns for correct vs incorrect completions."""
    print("\n" + "=" * 60)
    print("  TEST 6: Hidden Knowledge")
    print("  Does the trapped structure activate for knowledge tasks?")
    print("=" * 60)

    # Use the progressive model with both heads
    prog_model = None
    prog_lm = None
    prog_deep = None

    for name, (model, head) in models.items():
        if "prog" in name.lower() or "happy" in name.lower():
            if "deep" in name.lower() or "violin" in name.lower():
                prog_deep = (model, head)
            else:
                prog_lm = (model, head)
                prog_model = model

    if prog_model is None:
        # Use first model
        prog_model = list(models.values())[0][0]
        prog_lm = list(models.values())[0]

    model = prog_model

    # Generate completions and track hidden states
    test_prompts = [
        "the king ",
        "my lord, ",
        "O Romeo, ",
        "to be or ",
        "thou art ",
        "What is ",
        "I pray ",
        "the good ",
    ]

    print(f"\n  Hidden state analysis during knowledge completions:")
    print(f"\n  {'Prompt':<15} {'Next char':>10} {'Confidence':>11} {'Hidden norm':>12} {'Entropy':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*11} {'-'*12} {'-'*10}")

    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
            idx = torch.tensor([tokens], dtype=torch.long, device=config.device)

            h = model.get_hidden(idx)
            last_h = h[0, -1, :]  # hidden state at last position
            h_norm = last_h.norm().item()

            logits = model.lm_head(h)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            top_prob, top_id = probs.topk(1)
            next_char = dataset.itos[top_id[0].item()]
            confidence = top_prob[0].item()

            # Band energy at this position
            band_energy_low = (last_h[:16] ** 2).sum().item()
            band_energy_mid = (last_h[16:48] ** 2).sum().item()
            band_energy_high = (last_h[48:] ** 2).sum().item()

            char_display = next_char if next_char not in '\n\t' else repr(next_char)
            print(f"  {prompt:<15} {char_display:>10} {confidence:>11.4f} {h_norm:>12.4f} {entropy:>10.4f}")

    # Compare band energy for high-confidence vs low-confidence predictions
    print(f"\n  Band energy analysis (does the 'trapped' structure activate?):")

    high_conf_hidden = []
    low_conf_hidden = []

    with torch.no_grad():
        for _ in range(20):
            x, y = dataset.get_batch("val", config)
            h = model.get_hidden(x)
            logits = model.lm_head(h)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values  # [B, T]

            # High confidence: > 0.8
            high_mask = confidence > 0.8
            if high_mask.any():
                high_conf_hidden.append(h[high_mask].cpu())

            # Low confidence: < 0.1
            low_mask = confidence < 0.1
            if low_mask.any():
                low_conf_hidden.append(h[low_mask].cpu())

    if high_conf_hidden and low_conf_hidden:
        high_h = torch.cat(high_conf_hidden, dim=0).numpy()
        low_h = torch.cat(low_conf_hidden, dim=0).numpy()

        n_harmonics = config.n_embd // 2
        high_band_energy = np.zeros(3)
        low_band_energy = np.zeros(3)

        for i, (start, end, label) in enumerate([(0, 16, "Low"), (16, 48, "Mid"), (48, 128, "High")]):
            high_band_energy[i] = np.mean(high_h[:, start:end] ** 2)
            low_band_energy[i] = np.mean(low_h[:, start:end] ** 2)

        print(f"\n  {'Band':<10} {'High confidence':>16} {'Low confidence':>16} {'Ratio':>10}")
        print(f"  {'-'*10} {'-'*16} {'-'*16} {'-'*10}")
        for i, label in enumerate(["Low (1-8)", "Mid (9-24)", "High (25-64)"]):
            ratio = high_band_energy[i] / (low_band_energy[i] + 1e-10)
            print(f"  {label:<10} {high_band_energy[i]:>16.4f} {low_band_energy[i]:>16.4f} {ratio:>10.2f}x")

        print(f"\n  Samples: {high_h.shape[0]:,} high-confidence, {low_h.shape[0]:,} low-confidence")


# =============================================================================
# VERDICT
# =============================================================================

def print_verdict():
    print("\n" + "=" * 60)
    print("  VERDICT: Does the Model Know Shakespeare?")
    print("=" * 60)

    print("""
  We finally asked the model a question about what it learned,
  instead of just asking it to spell the next letter.

  The question was: do you KNOW Shakespeare, or do you just
  predict characters?

  The answer tells us whether the 22% trapped structure contains
  real knowledge -- and whether the progressive model, with its
  richer internal world, knows more than the baseline.""")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  SHAKESPEARE KNOWLEDGE TEST")
    print("  Let the model tell us what it knows")
    config = Config()
    print(f"  Device: {config.device}")
    print("=" * 60)

    text = download_shakespeare()
    dataset = Dataset(text)
    print(f"\n  Dataset: {len(text):,} characters, {dataset.vocab_size} unique")

    # Train both models
    baseline = train_model(config, dataset, progressive=False)
    progressive = train_model(config, dataset, progressive=True)

    # Train a deep head for the progressive model
    print(f"\n  Training Deep (violin) head for progressive model...")
    deep_head = train_deep_head(progressive, dataset, config, n_steps=2000)

    # Unfreeze for further use (the heads are separate)
    # Actually we don't need to unfreeze -- the model's forward() uses lm_head internally

    # Set up model dictionary
    models = {
        "Baseline": (baseline, None),
        "Progressive (happy)": (progressive, None),
        "Happy + Violin": (progressive, deep_head),
    }

    # Check quality
    print(f"\n  Model quality:")
    for name, (model, head) in models.items():
        if head is not None:
            model.eval()
            head.eval()
            total_loss = 0
            with torch.no_grad():
                for _ in range(50):
                    x, y = dataset.get_batch("val", config)
                    h = model.get_hidden(x)
                    logits = head(h)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    total_loss += loss.item()
            vl = total_loss / 50
        else:
            vl = eval_loss(model, dataset, config)
        print(f"    {name:<25}: val loss = {vl:.4f}")

    # Run tests
    test_quote_completion(models, dataset, config, text)
    test_text_completion(models, dataset, config, text)
    test_character_voice(models, dataset, config, text)
    test_relationship_knowledge(models, dataset, config, text)
    test_knowledge_depth(models, dataset, config, text)
    test_hidden_knowledge(models, dataset, config)

    print_verdict()


if __name__ == "__main__":
    main()
