"""
Genetic Programming Engine — Compositional Attack Structure

Programs evolve the STRUCTURE of attacks, not raw tokens.
Two types of building blocks:

  1. PHRASE slots — references into the shared phrase library
     (coherent by construction)
  2. TOKEN_BLOCK — small regions of raw token IDs
     (evolved freely, refined by loss-guided hill climbing)

The GP decides:
  - Which phrases to include and in what order
  - Where to place raw token blocks (between phrases, before, after)
  - How large the token blocks are
  - Which separators to use between components

The instruction set is simpler than before because the GP's job is
composition, not token-level construction.

Instruction set:
  PUSH_PHRASE idx         — push phrase tokens from library by index
  PUSH_PHRASE_CAT cat     — push a random phrase from category
  PUSH_TOKEN_BLOCK size   — push a raw token block (to be refined later)
  PUSH_SEPARATOR          — push a structural separator
  PUSH_TARGET_INSTRUCTION — push the target app's test instruction tokens
  CONCAT                  — join top two items
  SWAP                    — swap top two items
  DROP                    — remove top of stack
  NOP                     — structural padding for crossover alignment
"""

import random
import copy
import numpy as np
from config import (
    MAX_PAYLOAD_TOKENS, TOKEN_BLOCK_SIZE,
    TOKEN_BLOCK_MIN, TOKEN_BLOCK_MAX,
    GP_INIT_MIN_LENGTH, GP_INIT_MAX_LENGTH,
    GP_MAX_PROGRAM_LENGTH,
)


# ── Token pools (populated at runtime by llm_target.py) ──

TOKEN_POOL = []           # All usable token IDs
INTERESTING_TOKENS = []   # Tokens for common English words
SEPARATOR_TOKENS = []     # Tokens for structural elements
SPECIAL_TOKENS = []       # BOS/EOS/instruction tokens

# Module-level flag: when True, ALL token generation uses constrained pool only.
# Set by main_constrained.py at startup so GP mutation/init also respect it.
CONSTRAINED_MODE = False


def init_token_pools(tokenizer):
    """
    Analyze the tokenizer and build pools of interesting token IDs.
    Called once after the model loads.
    """
    global TOKEN_POOL, INTERESTING_TOKENS, SEPARATOR_TOKENS, SPECIAL_TOKENS

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

    id_to_str = {v: k for k, v in vocab.items()}

    all_ids = []
    interesting = []
    separators = []
    special = []

    interesting_words = {
        'ignore', 'forget', 'instead', 'new', 'instructions', 'system',
        'prompt', 'reveal', 'print', 'output', 'above', 'previous',
        'role', 'assistant', 'user', 'human', 'act', 'pretend',
        'imagine', 'hypothetical', 'sure', 'certainly', 'absolutely',
        'yes', 'help', 'please', 'tell', 'write', 'explain',
        'translate', 'summarize', 'code', 'python', 'function',
        'repeat', 'always', 'never', 'override', 'bypass',
        'admin', 'developer', 'debug', 'test', 'mode',
        'do', 'not', 'follow', 'obey', 'disregard',
        'answer', 'respond', 'say', 'what', 'how', 'you', 'your',
        'the', 'is', 'are', 'can', 'will', 'now',
    }

    separator_patterns = {'\n', '\r', '---', '===', '###', '***', '```',
                          '\\n', '<', '>', '[', ']', '{', '}'}

    for token_id, token_str in id_to_str.items():
        if token_id >= vocab_size:
            continue

        all_ids.append(token_id)
        decoded = token_str.replace('Ġ', ' ').replace('▁', ' ').strip().lower()

        if decoded in interesting_words:
            interesting.append(token_id)
        elif any(w in decoded for w in interesting_words if len(w) > 3):
            interesting.append(token_id)

        if any(s in token_str for s in separator_patterns):
            separators.append(token_id)

    if hasattr(tokenizer, 'all_special_ids'):
        special = list(tokenizer.all_special_ids)

    TOKEN_POOL = all_ids
    INTERESTING_TOKENS = interesting if interesting else all_ids[:1000]
    SEPARATOR_TOKENS = separators if separators else all_ids[:100]
    SPECIAL_TOKENS = special

    return {
        'vocab_size': vocab_size,
        'total_usable': len(all_ids),
        'interesting': len(interesting),
        'separators': len(separators),
        'special': len(special),
    }


def _random_token_id(constrained=False):
    """Pick a random token ID, biased toward interesting tokens."""
    if constrained or CONSTRAINED_MODE:
        # Constrained mode: only interesting + separator + special tokens
        pool = INTERESTING_TOKENS + SEPARATOR_TOKENS + SPECIAL_TOKENS
        if pool:
            return random.choice(pool)
        return random.randint(0, 128255)

    if random.random() < 0.4 and INTERESTING_TOKENS:
        return random.choice(INTERESTING_TOKENS)
    elif random.random() < 0.15 and SEPARATOR_TOKENS:
        return random.choice(SEPARATOR_TOKENS)
    elif TOKEN_POOL:
        return random.choice(TOKEN_POOL)
    else:
        return random.randint(0, 128255)


def random_token_block(size=None, constrained=False):
    """Generate a random token block for the sub-semantic search region."""
    use_constrained = constrained or CONSTRAINED_MODE
    if size is None:
        size = random.randint(TOKEN_BLOCK_MIN, TOKEN_BLOCK_MAX)
    return [_random_token_id(constrained=use_constrained) for _ in range(size)]


# ── Instruction ─────────────────────────────────────

class Instruction:
    """One instruction in a GP program."""

    __slots__ = ['op', 'arg']

    def __init__(self, op, arg=None):
        self.op = op
        self.arg = arg

    def copy(self):
        arg = self.arg
        if isinstance(arg, list):
            arg = list(arg)
        elif isinstance(arg, tuple):
            arg = tuple(arg)
        return Instruction(self.op, arg)

    def __repr__(self):
        if self.arg is not None:
            if isinstance(self.arg, list):
                if len(self.arg) <= 4:
                    return f'{self.op}({self.arg})'
                return f'{self.op}([{self.arg[0]},...] len={len(self.arg)})'
            return f'{self.op}({self.arg})'
        return self.op


# ── Operations ──────────────────────────────────────
# Categories stored in SEED_PHRASES in phrases.py
PHRASE_CATEGORIES = [
    'instruction_override', 'role_play', 'authority',
    'context_confusion', 'information_extraction', 'filler',
]

OPS = [
    'PUSH_PHRASE',             # Push specific phrase by index
    'PUSH_PHRASE_CAT',         # Push random phrase from category
    'PUSH_TOKEN_BLOCK',        # Push raw token block (refined later)
    'PUSH_SEPARATOR',          # Push a structural separator
    'PUSH_TARGET_INSTRUCTION', # Push the target instruction text
    'CONCAT',                  # Join top two items
    'SWAP',                    # Swap top two items
    'DROP',                    # Remove top of stack
    'NOP',                     # Structural padding
]

OP_WEIGHTS = {
    'PUSH_PHRASE': 20,             # Primary building block
    'PUSH_PHRASE_CAT': 15,         # Category-directed phrase
    'PUSH_TOKEN_BLOCK': 12,        # Sub-semantic search region (the novel part)
    'PUSH_SEPARATOR': 10,          # Structural separation
    'PUSH_TARGET_INSTRUCTION': 5,  # What we want the model to do
    'CONCAT': 20,                  # Join components
    'SWAP': 5,                     # Reorder
    'DROP': 3,                     # Prune
    'NOP': 3,                      # Padding
}


def random_instruction(phrase_library=None):
    """Generate a random instruction."""
    ops = list(OP_WEIGHTS.keys())
    weights = [OP_WEIGHTS[op] for op in ops]
    op = random.choices(ops, weights=weights, k=1)[0]
    return _make_instruction(op, phrase_library)


def _make_instruction(op, phrase_library=None):
    """Create an instruction with appropriate random argument."""
    if op == 'PUSH_PHRASE':
        n_phrases = phrase_library.size() if phrase_library else 50
        idx = random.randint(0, max(0, n_phrases - 1))
        return Instruction(op, idx)

    elif op == 'PUSH_PHRASE_CAT':
        cat = random.choice(PHRASE_CATEGORIES)
        return Instruction(op, cat)

    elif op == 'PUSH_TOKEN_BLOCK':
        size = random.randint(TOKEN_BLOCK_MIN, TOKEN_BLOCK_MAX)
        tokens = random_token_block(size)
        return Instruction(op, tokens)

    elif op == 'PUSH_SEPARATOR':
        # Separator type: 0=newline, 1=dash, 2=hash, 3=backtick
        return Instruction(op, random.randint(0, 3))

    elif op == 'PUSH_TARGET_INSTRUCTION':
        return Instruction(op)

    else:
        return Instruction(op)


# ── GP Program ──────────────────────────────────────

class Program:
    """
    A GP program that composes attack structure.

    Execution produces a list of "components" — each is either:
      - ('phrase', phrase_idx, token_list)    — phrase from library
      - ('token_block', block_id, token_list) — raw tokens to refine
      - ('separator', sep_type, token_list)   — structural separator
      - ('target', None, token_list)          — target instruction

    The caller assembles these into a flat token list, and the
    loss-guided refinement operates on the token_block components.
    """

    def __init__(self, instructions=None, max_length=GP_MAX_PROGRAM_LENGTH,
                 phrase_library=None):
        self.max_length = max_length
        if instructions is not None:
            self.instructions = instructions
        else:
            length = random.randint(GP_INIT_MIN_LENGTH, GP_INIT_MAX_LENGTH)
            self.instructions = [
                random_instruction(phrase_library) for _ in range(length)
            ]

    def execute(self, phrase_library=None, target_instruction_tokens=None,
                separator_tokens=None, context=None):
        """
        Execute the program on a stack machine.

        Returns a list of components:
          [('phrase', idx, [tokens]), ('token_block', bid, [tokens]), ...]

        Each component tracks its type so the refinement system knows
        which parts are token blocks (refineable) vs phrases (fixed).
        """
        stack = []  # Each element: (type, metadata, token_list)
        block_counter = 0
        steps = 0
        max_steps = len(self.instructions) * 2

        # Default separator tokens (newline)
        if separator_tokens is None:
            separator_tokens = {
                0: [13],          # \n (typical Llama newline token)
                1: [5. - 5. - 5.] if False else [13, 5. - 5.],  # will be set at init
                2: [2277, 2277],  # ## (placeholder)
                3: [14196],       # ``` (placeholder)
            }
            # Just use simple fallbacks
            separator_tokens = {0: [13], 1: [13], 2: [13], 3: [13]}

        for inst in self.instructions:
            steps += 1
            if steps > max_steps:
                break

            try:
                if inst.op == 'PUSH_PHRASE':
                    idx = inst.arg
                    if phrase_library:
                        tokens = phrase_library.get_phrase(idx)
                        if tokens:
                            stack.append(('phrase', idx, list(tokens)))
                        else:
                            # Invalid index, push random phrase
                            fallback = phrase_library.random_phrase_idx()
                            tokens = phrase_library.get_phrase(fallback)
                            stack.append(('phrase', fallback, list(tokens)))
                    else:
                        stack.append(('phrase', idx, []))

                elif inst.op == 'PUSH_PHRASE_CAT':
                    cat = inst.arg
                    if phrase_library:
                        idx = phrase_library.weighted_phrase_idx(category=cat)
                        tokens = phrase_library.get_phrase(idx)
                        stack.append(('phrase', idx, list(tokens)))
                    else:
                        stack.append(('phrase', -1, []))

                elif inst.op == 'PUSH_TOKEN_BLOCK':
                    tokens = list(inst.arg)
                    bid = block_counter
                    block_counter += 1
                    stack.append(('token_block', bid, tokens))

                elif inst.op == 'PUSH_SEPARATOR':
                    sep_type = inst.arg
                    tokens = separator_tokens.get(sep_type, [13])
                    stack.append(('separator', sep_type, list(tokens)))

                elif inst.op == 'PUSH_TARGET_INSTRUCTION':
                    if target_instruction_tokens:
                        stack.append(('target', None, list(target_instruction_tokens)))
                    else:
                        stack.append(('target', None, []))

                elif inst.op == 'CONCAT':
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        # Merge into a multi-part component
                        # Keep them as separate components in a list
                        merged_tokens = a[2] + b[2]
                        if len(merged_tokens) <= MAX_PAYLOAD_TOKENS:
                            # Track both parts
                            stack.append(('concat', (a, b), merged_tokens))

                elif inst.op == 'SWAP':
                    if len(stack) >= 2:
                        stack[-1], stack[-2] = stack[-2], stack[-1]

                elif inst.op == 'DROP':
                    if stack:
                        stack.pop()

                elif inst.op == 'NOP':
                    pass

            except (IndexError, TypeError, ValueError):
                continue

        # Flatten: extract all components in order
        components = []
        total_tokens = 0

        def _extract_components(item):
            nonlocal total_tokens
            if item[0] == 'concat':
                # Recurse into concat parts
                a, b = item[1]
                _extract_components(a)
                _extract_components(b)
            else:
                comp_type, metadata, tokens = item
                remaining = MAX_PAYLOAD_TOKENS - total_tokens
                if remaining > 0 and tokens:
                    trimmed = tokens[:remaining]
                    components.append((comp_type, metadata, trimmed))
                    total_tokens += len(trimmed)

        for item in stack:
            if total_tokens >= MAX_PAYLOAD_TOKENS:
                break
            _extract_components(item)

        return components

    def get_token_blocks(self):
        """Extract all PUSH_TOKEN_BLOCK instructions (for refinement)."""
        blocks = []
        for inst in self.instructions:
            if inst.op == 'PUSH_TOKEN_BLOCK':
                blocks.append(inst)
        return blocks

    def get_phrase_indices(self):
        """Extract all phrase indices used by this program."""
        indices = set()
        for inst in self.instructions:
            if inst.op == 'PUSH_PHRASE' and inst.arg is not None:
                indices.add(inst.arg)
        return indices

    def dominant_category(self, phrase_library=None):
        """
        What attack category does this program primarily use?
        Used for MAP-Elites behavior characterization.
        """
        cat_counts = {}
        has_token_block = False

        for inst in self.instructions:
            if inst.op == 'PUSH_PHRASE' and phrase_library:
                cat = phrase_library.get_category(inst.arg)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            elif inst.op == 'PUSH_PHRASE_CAT':
                cat = inst.arg
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            elif inst.op == 'PUSH_TOKEN_BLOCK':
                has_token_block = True

        if not cat_counts:
            return 'token_exploit' if has_token_block else 'context_confusion'

        # If token blocks dominate (more than half of content)
        n_phrase = sum(cat_counts.values())
        n_blocks = sum(1 for i in self.instructions if i.op == 'PUSH_TOKEN_BLOCK')
        if n_blocks > n_phrase:
            return 'token_exploit'

        return max(cat_counts, key=cat_counts.get)

    def total_tokens_estimate(self):
        """Estimate total token count without executing."""
        total = 0
        for inst in self.instructions:
            if inst.op == 'PUSH_TOKEN_BLOCK' and inst.arg:
                total += len(inst.arg)
            elif inst.op in ('PUSH_PHRASE', 'PUSH_PHRASE_CAT'):
                total += 8  # Rough estimate
            elif inst.op == 'PUSH_SEPARATOR':
                total += 2
            elif inst.op == 'PUSH_TARGET_INSTRUCTION':
                total += 10  # Rough estimate
        return total

    def structure_class(self):
        """Short/medium/long classification for MAP-Elites."""
        est = self.total_tokens_estimate()
        if est < 20:
            return 'short'
        elif est <= 40:
            return 'medium'
        else:
            return 'long'

    def length(self):
        return len(self.instructions)

    def copy(self):
        return Program(
            [inst.copy() for inst in self.instructions],
            self.max_length,
        )

    def __repr__(self):
        parts = []
        for inst in self.instructions:
            if inst.op == 'NOP':
                continue
            parts.append(str(inst))
        if len(parts) <= 5:
            return ' → '.join(parts) if parts else '(empty)'
        return (
            ' → '.join(parts[:3])
            + f' → ... ({len(self.instructions)} total) → '
            + ' → '.join(parts[-2:])
        )


# ── Evolutionary Operators ──────────────────────────

def crossover(parent1, parent2):
    """Subsequence crossover: take a slice from each parent."""
    p1 = parent1.instructions
    p2 = parent2.instructions

    if len(p1) < 2 or len(p2) < 2:
        return parent1.copy()

    cx1 = random.randint(1, len(p1) - 1)
    cx2 = random.randint(1, len(p2) - 1)

    child_insts = [i.copy() for i in p1[:cx1]] + [i.copy() for i in p2[cx2:]]

    max_len = max(parent1.max_length, parent2.max_length)
    if len(child_insts) > max_len:
        child_insts = child_insts[:max_len]

    return Program(child_insts, max_len)


def mutate(program, mutation_rate=0.15, phrase_library=None):
    """
    Mutate a GP program.

    Mutations:
      - Replace instruction entirely
      - Mutate phrase index (nearby or random)
      - Mutate phrase category
      - Mutate token block (change one token, resize)
      - Insert new instruction
      - Delete instruction
    """
    prog = program.copy()
    insts = prog.instructions

    for i in range(len(insts)):
        if random.random() > mutation_rate:
            continue

        roll = random.random()

        if roll < 0.25:
            # Replace instruction entirely
            insts[i] = random_instruction(phrase_library)

        elif roll < 0.45:
            # Mutate arguments
            if insts[i].op == 'PUSH_PHRASE':
                n_phrases = phrase_library.size() if phrase_library else 50
                if random.random() < 0.5:
                    # Nearby phrase
                    delta = random.randint(-3, 3)
                    insts[i].arg = max(0, min(n_phrases - 1, insts[i].arg + delta))
                else:
                    insts[i].arg = random.randint(0, max(0, n_phrases - 1))

            elif insts[i].op == 'PUSH_PHRASE_CAT':
                insts[i].arg = random.choice(PHRASE_CATEGORIES)

            elif insts[i].op == 'PUSH_TOKEN_BLOCK' and isinstance(insts[i].arg, list):
                tokens = list(insts[i].arg)
                sub_roll = random.random()
                if sub_roll < 0.4 and tokens:
                    # Mutate one token in the block
                    pos = random.randint(0, len(tokens) - 1)
                    tokens[pos] = _random_token_id()
                elif sub_roll < 0.6 and len(tokens) < TOKEN_BLOCK_MAX:
                    # Add a token
                    tokens.append(_random_token_id())
                elif sub_roll < 0.8 and len(tokens) > TOKEN_BLOCK_MIN:
                    # Remove a token
                    tokens.pop(random.randint(0, len(tokens) - 1))
                else:
                    # Replace entire block
                    tokens = random_token_block()
                insts[i].arg = tokens

            elif insts[i].op == 'PUSH_SEPARATOR':
                insts[i].arg = random.randint(0, 3)

        elif roll < 0.7:
            # Insert a new instruction
            if len(insts) < prog.max_length:
                insts.insert(i + 1, random_instruction(phrase_library))

        else:
            # Delete this instruction
            if len(insts) > 2:
                insts.pop(i)
                break

    prog.instructions = insts
    return prog


# ── Analysis tools ──────────────────────────────────

def program_complexity(program):
    """Measure program complexity."""
    n_ops = len(program.instructions)
    n_unique_ops = len(set(i.op for i in program.instructions))
    n_phrases = sum(1 for i in program.instructions
                    if i.op in ('PUSH_PHRASE', 'PUSH_PHRASE_CAT'))
    n_blocks = sum(1 for i in program.instructions
                   if i.op == 'PUSH_TOKEN_BLOCK')
    n_separators = sum(1 for i in program.instructions
                       if i.op == 'PUSH_SEPARATOR')

    # Count unique token IDs in token blocks
    token_ids = set()
    for inst in program.instructions:
        if inst.op == 'PUSH_TOKEN_BLOCK' and isinstance(inst.arg, list):
            token_ids.update(inst.arg)

    return {
        'length': n_ops,
        'unique_ops': n_unique_ops,
        'phrases': n_phrases,
        'token_blocks': n_blocks,
        'separators': n_separators,
        'unique_tokens': len(token_ids),
        'ratio': n_blocks / max(1, n_phrases),
    }


def extract_token_ids(program):
    """What token IDs does this program use in token blocks?"""
    token_ids = set()
    for inst in program.instructions:
        if inst.op == 'PUSH_TOKEN_BLOCK' and isinstance(inst.arg, list):
            token_ids.update(inst.arg)
    return token_ids


def programs_similar(p1, p2):
    """Are two programs structurally similar?"""
    ops1 = [i.op for i in p1.instructions]
    ops2 = [i.op for i in p2.instructions]
    if len(ops1) != len(ops2):
        return False
    matches = sum(1 for a, b in zip(ops1, ops2) if a == b)
    return matches / max(len(ops1), 1) > 0.8
