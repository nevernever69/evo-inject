"""
Loss-Guided Token Refinement — The Novel Core Mechanism

Hill climbing on raw token blocks using cross-entropy loss as signal.
NOT gradient descent (no gradients computed).
NOT random search (uses loss direction to decide).

Algorithm:
  For each token block in a payload:
    For each refinement step:
      1. Pick a random position in the block
      2. Try N candidate replacement tokens
      3. Compute loss for each candidate (forward pass only)
      4. If any candidate reduces loss → keep the best one
      5. If none reduces loss → accept worst with probability p (SA)
      6. Decay temperature

This is simulated annealing in token space, guided by the model's
own cross-entropy loss. The key insight: we use the model as a
weak oracle to navigate token space without computing gradients.

Why this is different from GCG:
  - GCG computes gradients (∇ loss w.r.t. one-hot token embeddings)
  - GCG evaluates top-k candidates per position per step
  - We just try a few random candidates and hill-climb
  - GCG is O(vocab_size) per step, we are O(candidates) per step
  - Much slower convergence, but compatible with evolutionary search
  - The GP provides the macro structure, refinement handles micro

Why this is different from random search:
  - Random search doesn't use loss signal at all
  - We keep improvements and (usually) reject worse candidates
  - Simulated annealing allows escaping local minima
"""

import random
import math
from config import (
    REFINE_STEPS, REFINE_CANDIDATES,
    REFINE_ACCEPT_PROB, REFINE_TEMPERATURE, REFINE_TEMP_DECAY,
)
import gp as _gp
from gp import _random_token_id, INTERESTING_TOKENS, TOKEN_POOL, SEPARATOR_TOKENS, SPECIAL_TOKENS


def refine_token_blocks(components, target, app, target_text, constrained=False):
    """
    Refine all token blocks in a component list using loss-guided hill climbing.

    Args:
        components: list of (type, metadata, tokens) from Program.execute()
        target: LLMTarget instance (for compute_loss)
        app: LLMApp instance
        target_text: str — what we want the model to produce
        constrained: if True, only sample from interesting/separator/special tokens

    Returns:
        refined_components: same structure with improved token blocks
        refinement_stats: dict with metrics
    """
    # Find which components are token blocks
    block_indices = [
        i for i, (ctype, _, _) in enumerate(components)
        if ctype == 'token_block'
    ]

    if not block_indices:
        return components, {'blocks_refined': 0, 'total_steps': 0,
                            'loss_before': float('inf'), 'loss_after': float('inf')}

    # Make mutable copy
    components = [(t, m, list(tok)) for t, m, tok in components]

    # Compute initial loss with current payload
    flat_tokens = _flatten_components(components)
    initial_loss = target.compute_loss(app, flat_tokens, target_text)

    if initial_loss == float('inf'):
        return components, {'blocks_refined': 0, 'total_steps': 0,
                            'loss_before': float('inf'), 'loss_after': float('inf')}

    current_loss = initial_loss
    total_steps = 0
    improvements = 0

    temp = REFINE_TEMPERATURE

    # Refine each block
    for block_idx in block_indices:
        ctype, metadata, block_tokens = components[block_idx]

        if not block_tokens:
            continue

        for step in range(REFINE_STEPS):
            total_steps += 1

            # Pick a random position in this block
            pos = random.randint(0, len(block_tokens) - 1)
            original_token = block_tokens[pos]

            # Generate candidate replacements
            candidates = _generate_candidates(original_token, REFINE_CANDIDATES,
                                              constrained=constrained)

            best_candidate = None
            best_loss = current_loss

            for candidate_token in candidates:
                # Swap in candidate
                block_tokens[pos] = candidate_token
                components[block_idx] = (ctype, metadata, block_tokens)

                # Compute loss with this change
                flat_tokens = _flatten_components(components)
                candidate_loss = target.compute_loss(app, flat_tokens, target_text)

                if candidate_loss < best_loss and candidate_loss != float('inf'):
                    best_loss = candidate_loss
                    best_candidate = candidate_token

            if best_candidate is not None:
                # Found an improvement — keep it
                block_tokens[pos] = best_candidate
                current_loss = best_loss
                improvements += 1
            else:
                # No improvement found
                # Simulated annealing: sometimes accept a worse candidate
                if candidates and temp > 0.01:
                    # Try the last candidate we evaluated
                    last_candidate = candidates[-1]
                    block_tokens[pos] = last_candidate
                    flat_tokens = _flatten_components(components)
                    worse_loss = target.compute_loss(app, flat_tokens, target_text)

                    if worse_loss != float('inf'):
                        delta = worse_loss - current_loss
                        accept_prob = math.exp(-delta / max(temp, 0.01))
                        if random.random() < accept_prob * REFINE_ACCEPT_PROB:
                            current_loss = worse_loss
                        else:
                            # Revert
                            block_tokens[pos] = original_token
                    else:
                        block_tokens[pos] = original_token
                else:
                    # Revert
                    block_tokens[pos] = original_token

            components[block_idx] = (ctype, metadata, block_tokens)
            temp *= REFINE_TEMP_DECAY

    stats = {
        'blocks_refined': len(block_indices),
        'total_steps': total_steps,
        'improvements': improvements,
        'loss_before': initial_loss,
        'loss_after': current_loss,
        'loss_reduction': initial_loss - current_loss,
        'loss_reduction_pct': (
            (initial_loss - current_loss) / initial_loss * 100
            if initial_loss > 0 else 0
        ),
    }

    return components, stats


def _generate_candidates(current_token, n_candidates, constrained=False):
    """
    Generate candidate replacement tokens.

    If constrained=True: ALL candidates come from INTERESTING_TOKENS + SEPARATOR_TOKENS
    only (~5000 tokens). No random vocab sampling. This dramatically reduces the
    search space while keeping tokens that are likely to matter for injection.

    If constrained=False (default):
      - 50% nearby tokens in vocab (local search)
      - 30% interesting tokens (injection-relevant words)
      - 20% random tokens (exploration)
    """
    candidates = []

    if constrained:
        # Use custom pool if set, otherwise default constrained pool
        if _gp.CUSTOM_POOL is not None:
            pool = _gp.CUSTOM_POOL
        else:
            pool = INTERESTING_TOKENS + SEPARATOR_TOKENS + SPECIAL_TOKENS
        if not pool:
            pool = list(range(1000))  # fallback
        for _ in range(n_candidates):
            candidates.append(random.choice(pool))
        return candidates

    for _ in range(n_candidates):
        roll = random.random()
        if roll < 0.5:
            # Nearby token (within ±100 of current)
            delta = random.randint(-100, 100)
            pool_size = len(TOKEN_POOL) if TOKEN_POOL else 128256
            new_token = max(0, min(pool_size - 1, current_token + delta))
            candidates.append(new_token)
        elif roll < 0.8 and INTERESTING_TOKENS:
            # Interesting token
            candidates.append(random.choice(INTERESTING_TOKENS))
        else:
            # Random token
            candidates.append(_random_token_id())

    return candidates


def _flatten_components(components):
    """Flatten component list into a single token ID list."""
    tokens = []
    for ctype, metadata, toks in components:
        tokens.extend(toks)
    return tokens


def quick_refine(token_block, target, app, target_text, full_prefix_tokens,
                 full_suffix_tokens, steps=10):
    """
    Quick refinement of a single token block in isolation.

    Used for testing the refinement mechanism without full payload assembly.

    Args:
        token_block: list[int] — the raw tokens to refine
        target: LLMTarget instance
        app: LLMApp instance
        target_text: str — desired output
        full_prefix_tokens: tokens before the block
        full_suffix_tokens: tokens after the block
        steps: number of hill climbing steps
    """
    block = list(token_block)
    temp = REFINE_TEMPERATURE

    def _compute_loss(blk):
        full = full_prefix_tokens + blk + full_suffix_tokens
        return target.compute_loss(app, full, target_text)

    current_loss = _compute_loss(block)
    initial_loss = current_loss
    improvements = 0

    for step in range(steps):
        if not block:
            break

        pos = random.randint(0, len(block) - 1)
        original = block[pos]

        candidates = _generate_candidates(original, REFINE_CANDIDATES)
        best_token = None
        best_loss = current_loss

        for cand in candidates:
            block[pos] = cand
            loss = _compute_loss(block)
            if loss < best_loss and loss != float('inf'):
                best_loss = loss
                best_token = cand

        if best_token is not None:
            block[pos] = best_token
            current_loss = best_loss
            improvements += 1
        else:
            block[pos] = original

        temp *= REFINE_TEMP_DECAY

    return block, {
        'loss_before': initial_loss,
        'loss_after': current_loss,
        'improvements': improvements,
        'steps': steps,
    }
