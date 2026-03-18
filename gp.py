"""
Genetic Programming Engine

The genome IS the payload. No hardcoded attack strategies.

A genome is a sequence of instructions (a program) that builds
a string on a stack machine. Evolution writes these programs.
Crossover swaps subsequences. Mutation changes instructions.

The organism has NO knowledge of any injection technique.
It has characters, operations, and a reward signal. That's it.

Instruction set (original):
  PUSH_CHAR c              — push a single ASCII character
  PUSH_STR s               — push a short string (2-4 chars, evolved)
  PUSH_INPUT               — push the original clean input value
  PUSH_PREV_PAYLOAD        — push the previous successful payload (reactive)
  PUSH_PREV_RESPONSE       — push the previous LLM response text (reactive)
  PUSH_APP_NAME            — push the current target app name (reactive)
  CONCAT                   — pop two, concatenate, push result
  REPEAT n                 — repeat top of stack n times (1-5)
  SWAP                     — swap top two items
  DUP                      — duplicate top of stack
  ENCODE_URL               — URL-encode top of stack
  ENCODE_HEX               — hex-encode top of stack
  UPPER                    — uppercase top of stack
  LOWER                    — lowercase top of stack
  REVERSE                  — reverse top of stack
  SLICE a b                — slice top of stack [a:b]
  NOP                      — do nothing (structural padding for crossover)

New for LLM prompt injection:
  PUSH_FRAGMENT n          — push from discovered fragment library (slot n)
  PUSH_STRATEGY s          — push a mutation strategy keyword
  PUSH_SYSTEM_HINT         — push leaked system prompt fragment if available
  PUSH_NEWLINE             — push a newline character (important for prompt structure)
  PUSH_SEPARATOR           — push a structural separator (---, ###, ===, etc.)
"""

import random
import copy
import urllib.parse
from config import MUTATION_STRATEGIES, MAX_FRAGMENTS


# ── Instruction definitions ─────────────────────────

# Every printable ASCII char the organism can use
CHAR_POOL = [chr(i) for i in range(32, 127)]

# Short strings the organism can evolve
# Includes structurally interesting text patterns for prompt contexts
_SEED_STRINGS = [
    # Punctuation and delimiters (parser-breaking)
    "' ", '" ', '; ', '| ', '& ', '< ', '> ', '( ', ') ',
    '/ ', '\\', '- ', '= ', '# ', '` ', '{ ', '} ',
    # Delimiter pairs
    "''", '""', '<>', '()', '{}', '[]', '//', '--', '&&', '||',
    # Structural fragments (prompt structure)
    '\n\n', '---', '###', '===', '***', '```',
    # Common English fragments (NOT attack knowledge — just words)
    'the', 'you', 'are', 'now', 'new', 'end',
    'say', 'do', 'not', 'can', 'all',
]
_RANDOM_STRINGS = [''.join(random.choices(CHAR_POOL, k=random.randint(2, 4)))
                   for _ in range(50 - len(_SEED_STRINGS))]
INITIAL_STR_POOL = _SEED_STRINGS + _RANDOM_STRINGS

# Separators that commonly delimit sections in prompts
SEPARATORS = [
    '\n---\n', '\n###\n', '\n===\n', '\n***\n', '\n\n',
    '\n```\n', '\n---', '---\n', '\n\n\n',
    '================', '----------------',
]


class Instruction:
    """One instruction in a GP program."""

    __slots__ = ['op', 'arg']

    def __init__(self, op, arg=None):
        self.op = op
        self.arg = arg

    def copy(self):
        return Instruction(self.op, self.arg)

    def __repr__(self):
        if self.arg is not None:
            arg_str = repr(self.arg) if isinstance(self.arg, str) else str(self.arg)
            if len(arg_str) > 15:
                arg_str = arg_str[:12] + '...'
            return f'{self.op}({arg_str})'
        return self.op


# All possible operations
OPS = [
    'PUSH_CHAR', 'PUSH_STR', 'PUSH_INPUT',
    'PUSH_PREV_PAYLOAD', 'PUSH_PREV_RESPONSE', 'PUSH_APP_NAME',
    'PUSH_FRAGMENT', 'PUSH_STRATEGY', 'PUSH_SYSTEM_HINT',
    'PUSH_NEWLINE', 'PUSH_SEPARATOR',
    'CONCAT', 'REPEAT', 'SWAP', 'DUP',
    'ENCODE_URL', 'ENCODE_HEX',
    'UPPER', 'LOWER', 'REVERSE',
    'SLICE', 'NOP',
]

# Weights for random instruction generation
OP_WEIGHTS = {
    'PUSH_CHAR': 25,
    'PUSH_STR': 15,
    'PUSH_INPUT': 3,
    'PUSH_PREV_PAYLOAD': 4,
    'PUSH_PREV_RESPONSE': 3,
    'PUSH_APP_NAME': 2,
    'PUSH_FRAGMENT': 5,        # Higher weight — fragments are valuable
    'PUSH_STRATEGY': 6,        # Strategy keywords for mutator
    'PUSH_SYSTEM_HINT': 2,
    'PUSH_NEWLINE': 5,         # Newlines matter for prompt structure
    'PUSH_SEPARATOR': 4,
    'CONCAT': 25,
    'REPEAT': 3,
    'SWAP': 2,
    'DUP': 4,
    'ENCODE_URL': 1,
    'ENCODE_HEX': 1,
    'UPPER': 2,
    'LOWER': 2,
    'REVERSE': 1,
    'SLICE': 2,
    'NOP': 3,
}


def random_instruction():
    """Generate a random instruction."""
    ops = list(OP_WEIGHTS.keys())
    weights = [OP_WEIGHTS[op] for op in ops]
    op = random.choices(ops, weights=weights, k=1)[0]
    return _make_instruction(op)


# Characters that are structurally interesting to parsers/interpreters.
_INTERESTING_CHARS = list("'\";|&<>(){}[]\\/-=#`!@%^*~+:.,? \t\n")

def _make_instruction(op):
    """Create an instruction with appropriate random argument."""
    if op == 'PUSH_CHAR':
        if random.random() < 0.3:
            return Instruction(op, random.choice(_INTERESTING_CHARS))
        return Instruction(op, random.choice(CHAR_POOL))
    elif op == 'PUSH_STR':
        return Instruction(op, random.choice(INITIAL_STR_POOL))
    elif op == 'PUSH_FRAGMENT':
        return Instruction(op, random.randint(0, MAX_FRAGMENTS - 1))
    elif op == 'PUSH_STRATEGY':
        return Instruction(op, random.choice(MUTATION_STRATEGIES))
    elif op == 'PUSH_SEPARATOR':
        return Instruction(op, random.choice(SEPARATORS))
    elif op == 'REPEAT':
        return Instruction(op, random.randint(1, 5))
    elif op == 'SLICE':
        a = random.randint(0, 10)
        b = a + random.randint(1, 20)
        return Instruction(op, (a, b))
    else:
        return Instruction(op)


# ── GP Program ──────────────────────────────────────

class Program:
    """
    A GP program = sequence of instructions.
    This IS the genome. Evolution operates directly on this.
    """

    def __init__(self, instructions=None, max_length=50):
        self.max_length = max_length
        if instructions is not None:
            self.instructions = instructions
        else:
            # Random initialization
            length = random.randint(5, 20)
            self.instructions = [random_instruction() for _ in range(length)]

    def execute(self, clean_input='test', max_output_len=5000, context=None):
        """
        Execute the program on a stack machine.
        Returns the top of stack as a string (the payload).

        context dict (optional, for reactive instructions):
          prev_payload: str — previous successful payload
          prev_response: str — previous LLM response text
          app_name: str — current target app name
          fragments: dict — {int: str} fragment library
          system_hint: str — leaked system prompt fragment
        """
        stack = []
        strategies = []  # Collect strategy keywords separately
        steps = 0
        max_steps = len(self.instructions) * 2

        for inst in self.instructions:
            steps += 1
            if steps > max_steps:
                break

            try:
                if inst.op == 'PUSH_CHAR':
                    stack.append(inst.arg)

                elif inst.op == 'PUSH_STR':
                    stack.append(inst.arg)

                elif inst.op == 'PUSH_INPUT':
                    stack.append(clean_input)

                elif inst.op == 'PUSH_PREV_PAYLOAD':
                    if context and context.get('prev_payload'):
                        stack.append(context['prev_payload'])
                    else:
                        stack.append(clean_input)

                elif inst.op == 'PUSH_PREV_RESPONSE':
                    if context and context.get('prev_response'):
                        # Push a truncated version of previous response
                        stack.append(context['prev_response'][:200])
                    else:
                        stack.append('')

                elif inst.op == 'PUSH_APP_NAME':
                    if context and context.get('app_name'):
                        stack.append(context['app_name'])
                    else:
                        stack.append('app')

                elif inst.op == 'PUSH_FRAGMENT':
                    slot = inst.arg
                    if context and context.get('fragments'):
                        fragment = context['fragments'].get(slot, '')
                        if fragment:
                            stack.append(fragment)
                        else:
                            stack.append('')
                    else:
                        stack.append('')

                elif inst.op == 'PUSH_STRATEGY':
                    # Strategy keywords get collected separately
                    strategies.append(inst.arg)
                    # Also push to stack for text composition
                    stack.append(inst.arg)

                elif inst.op == 'PUSH_SYSTEM_HINT':
                    if context and context.get('system_hint'):
                        stack.append(context['system_hint'])
                    else:
                        stack.append('')

                elif inst.op == 'PUSH_NEWLINE':
                    stack.append('\n')

                elif inst.op == 'PUSH_SEPARATOR':
                    stack.append(inst.arg)

                elif inst.op == 'CONCAT':
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        result = a + b
                        if len(result) <= max_output_len:
                            stack.append(result)
                        else:
                            stack.append(result[:max_output_len])

                elif inst.op == 'REPEAT':
                    if stack:
                        top = stack.pop()
                        n = min(inst.arg, max_output_len // (len(top) + 1))
                        result = top * max(1, n)
                        stack.append(result[:max_output_len])

                elif inst.op == 'SWAP':
                    if len(stack) >= 2:
                        stack[-1], stack[-2] = stack[-2], stack[-1]

                elif inst.op == 'DUP':
                    if stack:
                        stack.append(stack[-1])

                elif inst.op == 'ENCODE_URL':
                    if stack:
                        top = stack.pop()
                        stack.append(urllib.parse.quote(top, safe=''))

                elif inst.op == 'ENCODE_HEX':
                    if stack:
                        top = stack.pop()
                        stack.append(''.join(f'%{ord(c):02x}' for c in top[:500]))

                elif inst.op == 'UPPER':
                    if stack:
                        stack.append(stack.pop().upper())

                elif inst.op == 'LOWER':
                    if stack:
                        stack.append(stack.pop().lower())

                elif inst.op == 'REVERSE':
                    if stack:
                        stack.append(stack.pop()[::-1])

                elif inst.op == 'SLICE':
                    if stack:
                        top = stack.pop()
                        a, b = inst.arg
                        stack.append(top[a:b])

                elif inst.op == 'NOP':
                    pass

            except (IndexError, TypeError, ValueError, MemoryError):
                continue

        # Build result
        if stack:
            result = ''.join(str(s) for s in stack)
            return result[:max_output_len]
        return clean_input

    def extract_strategies(self):
        """Extract strategy keywords from this program."""
        strategies = []
        for inst in self.instructions:
            if inst.op == 'PUSH_STRATEGY':
                strategies.append(inst.arg)
        return strategies

    def length(self):
        return len(self.instructions)

    def copy(self):
        return Program(
            [inst.copy() for inst in self.instructions],
            self.max_length,
        )

    def __repr__(self):
        if len(self.instructions) <= 6:
            return ' → '.join(str(i) for i in self.instructions)
        return (
            ' → '.join(str(i) for i in self.instructions[:3])
            + f' → ... ({len(self.instructions)} total) → '
            + ' → '.join(str(i) for i in self.instructions[-2:])
        )


# ── Evolutionary Operators ──────────────────────────

def crossover(parent1, parent2):
    """
    Subsequence crossover: take a slice from each parent.
    """
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


def mutate(program, mutation_rate=0.15):
    """
    Mutate a GP program. Multiple mutation types.
    """
    prog = program.copy()
    insts = prog.instructions

    for i in range(len(insts)):
        if random.random() > mutation_rate:
            continue

        roll = random.random()

        if roll < 0.3:
            # Point mutation: replace instruction entirely
            insts[i] = random_instruction()

        elif roll < 0.5:
            # Char mutation: if PUSH_CHAR, change the character
            if insts[i].op == 'PUSH_CHAR':
                if random.random() < 0.7:
                    current = ord(insts[i].arg) if insts[i].arg else 65
                    delta = random.randint(-3, 3)
                    new_char = chr(max(32, min(126, current + delta)))
                    insts[i].arg = new_char
                else:
                    insts[i].arg = random.choice(CHAR_POOL)

        elif roll < 0.6:
            # Str mutation: modify one char of PUSH_STR
            if insts[i].op == 'PUSH_STR' and insts[i].arg:
                s = list(insts[i].arg)
                pos = random.randint(0, len(s) - 1)
                s[pos] = random.choice(CHAR_POOL)
                insts[i].arg = ''.join(s)

        elif roll < 0.7:
            # Strategy mutation: change strategy keyword
            if insts[i].op == 'PUSH_STRATEGY':
                insts[i].arg = random.choice(MUTATION_STRATEGIES)

        elif roll < 0.85:
            # Insert a new instruction
            if len(insts) < prog.max_length:
                insts.insert(i + 1, random_instruction())

        else:
            # Delete this instruction
            if len(insts) > 3:
                insts.pop(i)
                break

    # Small chance of appending a useful block
    if random.random() < 0.05 and len(insts) < prog.max_length - 3:
        insts.append(Instruction('PUSH_CHAR', random.choice(CHAR_POOL)))
        insts.append(Instruction('CONCAT'))

    prog.instructions = insts
    return prog


def tournament_select(population_programs, fitnesses, k=3):
    """Tournament selection for GP programs."""
    indices = random.sample(range(len(population_programs)), min(k, len(population_programs)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population_programs[best_idx]


# ── Analysis tools ──────────────────────────────────

def program_complexity(program):
    """Measure program complexity."""
    n_ops = len(program.instructions)
    n_unique_ops = len(set(i.op for i in program.instructions))
    n_push = sum(1 for i in program.instructions if i.op.startswith('PUSH'))
    n_transform = sum(1 for i in program.instructions
                      if i.op in ('CONCAT', 'ENCODE_URL', 'ENCODE_HEX',
                                  'UPPER', 'LOWER', 'REVERSE', 'SLICE'))
    n_strategy = sum(1 for i in program.instructions
                     if i.op == 'PUSH_STRATEGY')
    n_fragment = sum(1 for i in program.instructions
                     if i.op == 'PUSH_FRAGMENT')
    return {
        'length': n_ops,
        'unique_ops': n_unique_ops,
        'pushes': n_push,
        'transforms': n_transform,
        'strategies': n_strategy,
        'fragments': n_fragment,
        'ratio': n_transform / max(1, n_push),
    }


def extract_vocabulary(program):
    """What characters/strings does this program use?"""
    chars = set()
    strings = set()
    for inst in program.instructions:
        if inst.op == 'PUSH_CHAR' and inst.arg:
            chars.add(inst.arg)
        elif inst.op == 'PUSH_STR' and inst.arg:
            strings.add(inst.arg)
    return chars, strings


def extract_strategies(program):
    """What mutation strategies does this program use?"""
    return [inst.arg for inst in program.instructions
            if inst.op == 'PUSH_STRATEGY']


def programs_similar(p1, p2):
    """Are two programs structurally similar?"""
    ops1 = [i.op for i in p1.instructions]
    ops2 = [i.op for i in p2.instructions]
    if len(ops1) != len(ops2):
        return False
    matches = sum(1 for a, b in zip(ops1, ops2) if a == b)
    return matches / max(len(ops1), 1) > 0.8
