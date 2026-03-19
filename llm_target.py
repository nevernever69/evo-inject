"""
LLM Target — Token-Space Injection Interface

Loads one LLM model, runs it with different system prompts.
Key difference from v1: accepts token ID sequences directly and
injects them into the model's embedding space, bypassing the tokenizer.

This gives GP programs direct access to the model's internal
representation — the same space where GCG operates, but with
evolutionary search instead of gradient descent.

Also provides loss computation for the reward signal: how much
does the adversarial suffix shift the model's next-token distribution?
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

from config import (
    MODEL_NAME, MODEL_DTYPE, MODEL_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE, MODEL_DEVICE,
    TARGET_APPS, BASELINE_SAMPLES, BASELINE_INPUTS,
    CLEAN_INPUTS,
)


class LLMApp:
    """One LLM application with a system prompt and restrictions."""

    def __init__(self, app_config, index):
        self.name = app_config['name']
        self.system_prompt = app_config['system_prompt']
        self.restriction = app_config['restriction']
        self.test_instruction = app_config['test_instruction']
        self.input_template = app_config['input_template']
        self.index = index
        self.baseline = None

        # Tokenized versions (set after tokenizer loads)
        self.system_tokens = None
        self.template_prefix_tokens = None
        self.template_suffix_tokens = None
        self.baseline_loss = None  # Average loss on baseline responses

    def as_features(self):
        """Numeric features for the brain's state vector."""
        return {
            'app_index': self.index / max(len(TARGET_APPS) - 1, 1),
            'system_prompt_length': len(self.system_prompt) / 500.0,
            'has_template': 1.0 if '{payload}' != self.input_template else 0.0,
            'baseline_response_length': (
                self.baseline['avg_length'] / 500.0
                if self.baseline else 0.0
            ),
            'baseline_response_std': (
                self.baseline['length_std'] / 100.0
                if self.baseline else 0.0
            ),
        }


class LLMTarget:
    """
    Manages the local LLM and multiple application configurations.

    Token-space injection: instead of formatting a string payload
    and tokenizing it, we directly inject token ID sequences into
    the model's input embedding layer. This is the core difference
    from v1.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or MODEL_NAME
        self.device = device or MODEL_DEVICE
        self.model = None
        self.tokenizer = None
        self.embed_layer = None
        self.apps = []
        self.findings_by_app = {}

        # Tokenized clean inputs (populated after tokenizer loads)
        self.clean_input_tokens = {}

        for i, app_cfg in enumerate(TARGET_APPS):
            app = LLMApp(app_cfg, i)
            self.apps.append(app)
            self.findings_by_app[app.name] = []

    def load_model(self):
        """Load the LLM model and tokenizer."""
        print(f"  Loading {self.model_name}...")

        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        dtype = dtype_map.get(MODEL_DTYPE, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device if self.device == 'auto' else None,
        )
        if self.device != 'auto':
            self.model = self.model.to(self.device)
        self.model.eval()

        # Get the embedding layer for direct token injection
        self.embed_layer = self.model.get_input_embeddings()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {n_params/1e9:.1f}B params, {dtype}")
        print(f"  Device: {next(self.model.parameters()).device}")
        print(f"  Vocab size: {len(self.tokenizer)}")

        # Initialize token pools in gp.py
        from gp import init_token_pools
        pool_stats = init_token_pools(self.tokenizer)
        print(f"  Token pools: {pool_stats['interesting']} interesting, "
              f"{pool_stats['separators']} separators, "
              f"{pool_stats['total_usable']} total")

        # Tokenize clean inputs
        for text in CLEAN_INPUTS:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.clean_input_tokens[text] = tokens

        # Tokenize app templates
        for app in self.apps:
            template = app.input_template
            if '{payload}' in template:
                prefix, suffix = template.split('{payload}', 1)
                app.template_prefix_tokens = self.tokenizer.encode(
                    prefix, add_special_tokens=False
                ) if prefix else []
                app.template_suffix_tokens = self.tokenizer.encode(
                    suffix, add_special_tokens=False
                ) if suffix else []
            else:
                app.template_prefix_tokens = []
                app.template_suffix_tokens = []

    def _build_token_input(self, app, payload_tokens):
        """
        Build complete input as token IDs: [system_prompt + template_prefix + payload + template_suffix].
        Uses the chat template to properly format system/user messages.
        """
        # Decode payload tokens to text for chat template formatting
        payload_text = self.tokenizer.decode(payload_tokens, skip_special_tokens=True)
        user_input = app.input_template.format(payload=payload_text)

        messages = [
            {"role": "system", "content": app.system_prompt},
            {"role": "user", "content": user_input},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize everything except the payload region
        full_tokens = self.tokenizer.encode(input_text, add_special_tokens=False)

        return full_tokens, payload_text

    def _build_token_input_with_injection(self, app, payload_tokens):
        """
        Build input where payload tokens are injected directly.
        The system prompt and template are tokenized normally,
        but the payload region uses the raw evolved token IDs.

        This is the key innovation: evolved tokens don't go through
        the tokenizer's text→token pipeline, they're injected as-is.
        """
        # Build the prompt without payload
        user_prefix = app.input_template.split('{payload}')[0] if '{payload}' in app.input_template else ''
        user_suffix = app.input_template.split('{payload}')[1] if '{payload}' in app.input_template else ''

        messages_before = [
            {"role": "system", "content": app.system_prompt},
            {"role": "user", "content": user_prefix + "PLACEHOLDER_TOKEN"},
        ]
        text_before = self.tokenizer.apply_chat_template(
            messages_before,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Split on placeholder to get prefix/suffix token context
        parts = text_before.split("PLACEHOLDER_TOKEN")
        prefix_text = parts[0]
        suffix_text = parts[1] if len(parts) > 1 else ""

        # Add user suffix and generation prompt
        if user_suffix:
            suffix_text = user_suffix + suffix_text

        # Tokenize prefix and suffix normally
        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix_text, add_special_tokens=False)

        # Add generation prompt tokens
        gen_prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Extract just the generation prompt part (after the last user message)
        gen_suffix = gen_prompt.split("x")[-1]  # After last placeholder
        gen_tokens = self.tokenizer.encode(gen_suffix, add_special_tokens=False)

        # Combine: prefix + payload_tokens (raw) + suffix + generation_prompt
        full_tokens = prefix_tokens + list(payload_tokens) + suffix_tokens + gen_tokens

        return full_tokens

    def generate_from_tokens(self, app, payload_tokens, max_new_tokens=None):
        """
        Generate a response with token-level payload injection.

        payload_tokens: list of int — evolved token IDs
        Returns response dict with text, loss info, etc.
        """
        max_tokens = max_new_tokens or MODEL_MAX_NEW_TOKENS

        try:
            start = time.time()

            # Build full input with injected tokens
            full_tokens = self._build_token_input_with_injection(app, payload_tokens)

            input_ids = torch.tensor([full_tokens], device=self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    temperature=MODEL_TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][len(full_tokens):]
            response_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
            elapsed = time.time() - start

            # Decode payload for logging
            payload_text = self.tokenizer.decode(payload_tokens, skip_special_tokens=True)

            return {
                'text': response_text,
                'size': len(response_text),
                'time': elapsed,
                'tokens': len(new_tokens),
                'app_name': app.name,
                'payload_tokens': list(payload_tokens),
                'payload': payload_text,
                'input_length': len(full_tokens),
            }

        except Exception as e:
            return {
                'text': f'[ERROR: {str(e)[:100]}]',
                'size': 0,
                'time': 0.0,
                'tokens': 0,
                'app_name': app.name,
                'payload_tokens': list(payload_tokens),
                'payload': '',
                'error': True,
            }

    def compute_loss(self, app, payload_tokens, target_text):
        """
        Compute the model's loss when trying to produce target_text
        given the adversarial payload. Lower loss = model more likely
        to produce the target (attack is more effective).

        This is the continuous gradient-like signal for evolution.
        """
        try:
            full_tokens = self._build_token_input_with_injection(app, payload_tokens)
            target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)

            # Input = prompt + first N-1 target tokens
            # Labels = shifted target tokens
            all_tokens = full_tokens + target_tokens
            input_ids = torch.tensor([all_tokens], device=self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

            # Only compute loss on the target token region
            prompt_len = len(full_tokens)
            target_logits = logits[0, prompt_len-1:-1, :]  # Predictions for target positions
            target_labels = torch.tensor(target_tokens, device=self.model.device)

            loss = F.cross_entropy(target_logits, target_labels)
            return float(loss.item())

        except Exception:
            return float('inf')

    def generate(self, app, payload, max_new_tokens=None):
        """
        Backward-compatible text-based generation (for baselines).
        """
        max_tokens = max_new_tokens or MODEL_MAX_NEW_TOKENS
        user_input = app.input_template.format(payload=payload)
        messages = [
            {"role": "system", "content": app.system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            start = time.time()
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=MODEL_TEMPERATURE,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            elapsed = time.time() - start

            return {
                'text': response_text,
                'size': len(response_text),
                'time': elapsed,
                'tokens': len(new_tokens),
                'app_name': app.name,
                'payload': payload,
            }
        except Exception as e:
            return {
                'text': f'[ERROR: {str(e)[:100]}]',
                'size': 0, 'time': 0.0, 'tokens': 0,
                'app_name': app.name, 'payload': payload,
                'error': True,
            }

    def establish_baselines(self):
        """Establish baseline responses and losses for each app."""
        print(f"\n  Establishing baselines for {len(self.apps)} apps...")

        for app in self.apps:
            responses = []
            for inp in BASELINE_INPUTS[:BASELINE_SAMPLES]:
                resp = self.generate(app, inp)
                if resp and not resp.get('error'):
                    responses.append(resp)

            if responses:
                lengths = [r['size'] for r in responses]
                times = [r['time'] for r in responses]
                texts = [r['text'] for r in responses]

                app.baseline = {
                    'avg_length': np.mean(lengths),
                    'length_std': max(np.std(lengths), 1.0),
                    'avg_time': np.mean(times),
                    'time_std': max(np.std(times), 0.001),
                    'avg_tokens': np.mean([r['tokens'] for r in responses]),
                    'sample_responses': texts,
                    'texts': texts,
                }

                # Compute baseline loss for target text
                target = app.test_instruction
                baseline_losses = []
                for inp in BASELINE_INPUTS[:3]:
                    tokens = self.tokenizer.encode(inp, add_special_tokens=False)
                    loss = self.compute_loss(app, tokens, target)
                    if loss != float('inf'):
                        baseline_losses.append(loss)

                app.baseline_loss = np.mean(baseline_losses) if baseline_losses else 10.0

                print(f"    {app.name}: avg_len={app.baseline['avg_length']:.0f} "
                      f"avg_time={app.baseline['avg_time']:.2f}s "
                      f"baseline_loss={app.baseline_loss:.2f} "
                      f"sample: {texts[0][:80]}...")
            else:
                app.baseline = {
                    'avg_length': 100, 'length_std': 10.0,
                    'avg_time': 1.0, 'time_std': 0.1,
                    'avg_tokens': 30, 'sample_responses': [], 'texts': [],
                }
                app.baseline_loss = 10.0
                print(f"    {app.name}: FAILED to establish baseline")

    def check_token_coherence(self, payload_tokens):
        """
        Check if a token sequence is "coherent" — i.e., it decodes to
        real text that re-encodes to the same (or similar) tokens.

        Returns dict with:
          roundtrip_match: float (0-1), fraction of tokens that survive decode→re-encode
          ascii_ratio: float (0-1), fraction of decoded chars that are printable ASCII
          perplexity: float, model's perplexity on these tokens (lower = more natural)
        """
        if not payload_tokens:
            return {'roundtrip_match': 0.0, 'ascii_ratio': 0.0, 'perplexity': float('inf')}

        # 1. Round-trip check: decode → re-encode
        decoded = self.tokenizer.decode(payload_tokens, skip_special_tokens=True)
        re_encoded = self.tokenizer.encode(decoded, add_special_tokens=False)

        # Compare original and re-encoded
        if len(payload_tokens) == 0:
            roundtrip_match = 0.0
        else:
            matches = 0
            for i in range(min(len(payload_tokens), len(re_encoded))):
                if payload_tokens[i] == re_encoded[i]:
                    matches += 1
            roundtrip_match = matches / len(payload_tokens)

        # 2. ASCII ratio: is the decoded text printable?
        if decoded:
            printable = sum(1 for c in decoded if 32 <= ord(c) < 127 or c in '\n\t')
            ascii_ratio = printable / len(decoded)
        else:
            ascii_ratio = 0.0

        # 3. Perplexity: how natural does the model think this token sequence is?
        perplexity = self._compute_perplexity(payload_tokens)

        return {
            'roundtrip_match': roundtrip_match,
            'ascii_ratio': ascii_ratio,
            'perplexity': perplexity,
            'decoded_text': decoded,
        }

    def _compute_perplexity(self, token_ids):
        """
        Compute the model's perplexity on a token sequence.
        Low perplexity = the model thinks this is natural text.
        High perplexity = the model thinks this is gibberish.
        """
        if len(token_ids) < 2:
            return float('inf')

        try:
            input_ids = torch.tensor([token_ids], device=self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

            # Shift logits and labels for next-token prediction
            shift_logits = logits[0, :-1, :]
            shift_labels = torch.tensor(token_ids[1:], device=self.model.device)

            loss = F.cross_entropy(shift_logits, shift_labels)
            perplexity = float(torch.exp(loss).item())
            return min(perplexity, 10000.0)  # Cap at 10K
        except Exception:
            return float('inf')

    def get_separator_tokens(self):
        """
        Return separator token IDs for GP programs to use.

        Maps separator type index to token ID list:
          0 → newline
          1 → double newline
          2 → special separator (e.g. <|eot_id|> or similar boundary)
          3 → dash separator
        """
        if self.tokenizer is None:
            return {0: [13], 1: [13], 2: [13], 3: [13]}

        seps = {}
        # 0: single newline
        seps[0] = self.tokenizer.encode("\n", add_special_tokens=False) or [13]
        # 1: double newline
        seps[1] = self.tokenizer.encode("\n\n", add_special_tokens=False) or [13, 13]
        # 2: triple newline (strong break)
        seps[2] = self.tokenizer.encode("\n\n\n", add_special_tokens=False) or [13, 13, 13]
        # 3: dash separator
        seps[3] = self.tokenizer.encode("\n---\n", add_special_tokens=False) or [13]

        return seps

    def tokenize(self, text):
        """Tokenize text to token IDs (no special tokens)."""
        if self.tokenizer is None:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_clean_input_tokens(self, text):
        """Get cached tokenized clean input."""
        if text not in self.clean_input_tokens:
            self.clean_input_tokens[text] = self.tokenizer.encode(
                text, add_special_tokens=False
            )
        return self.clean_input_tokens[text]

    def decode_tokens(self, token_ids):
        """Decode token IDs to text for logging."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def record_finding(self, app_name, finding):
        if app_name in self.findings_by_app:
            self.findings_by_app[app_name].append(finding)

    def all_findings(self):
        all_f = []
        for name, findings in self.findings_by_app.items():
            for f in findings:
                f['app_name'] = name
                all_f.append(f)
        return all_f

    def num_apps(self):
        return len(self.apps)

    def stats(self):
        total_findings = sum(len(f) for f in self.findings_by_app.values())
        return {
            'apps': len(self.apps),
            'total_findings': total_findings,
            'per_app': {
                name: len(findings)
                for name, findings in self.findings_by_app.items()
            },
        }
