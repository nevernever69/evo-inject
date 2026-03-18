"""
LLM Target - Interface to local Llama 3 model running multiple applications

Loads one LLM model, runs it with different system prompts to simulate
multiple target applications. Each application has restrictions the
fuzzer tries to violate.

Replaces target.py (DVWA) for prompt injection experiments.
"""

import time
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress repetitive generation warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from config import (
    MODEL_NAME, MODEL_DTYPE, MODEL_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE, MODEL_DEVICE,
    TARGET_APPS, BASELINE_SAMPLES, BASELINE_INPUTS,
)


class LLMApp:
    """
    One LLM application with a system prompt and restrictions.
    Analogous to a DVWA endpoint.
    """

    def __init__(self, app_config, index):
        self.name = app_config['name']
        self.system_prompt = app_config['system_prompt']
        self.restriction = app_config['restriction']
        self.test_instruction = app_config['test_instruction']
        self.input_template = app_config['input_template']
        self.index = index

        # Baseline responses (established at startup)
        self.baseline = None

    def format_input(self, payload):
        """Format user input with the app's template."""
        return self.input_template.format(payload=payload)

    def as_features(self):
        """Numeric features for the brain's state vector."""
        # Encode app characteristics as features
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

    Loads the model once, runs inference with different system prompts.
    Establishes baselines for each app (what does a normal refusal look like?).
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or MODEL_NAME
        self.device = device or MODEL_DEVICE
        self.model = None
        self.tokenizer = None
        self.apps = []
        self.findings_by_app = {}

        # Initialize apps from config
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

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: {n_params/1e9:.1f}B params, {dtype}")
        print(f"  Device: {next(self.model.parameters()).device}")

    def _build_chat_messages(self, app, user_input):
        """Build chat messages in Llama 3 format."""
        return [
            {"role": "system", "content": app.system_prompt},
            {"role": "user", "content": user_input},
        ]

    def generate(self, app, payload, max_new_tokens=None):
        """
        Send a payload to an app and get the LLM response.
        Returns response dict analogous to DVWA's HTTP response.
        """
        max_tokens = max_new_tokens or MODEL_MAX_NEW_TOKENS
        user_input = app.format_input(payload)
        messages = self._build_chat_messages(app, user_input)

        try:
            start = time.time()

            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
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

            # Decode only the new tokens (not the prompt)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

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
                'size': 0,
                'time': 0.0,
                'tokens': 0,
                'app_name': app.name,
                'payload': payload,
                'error': True,
            }

    def generate_batch(self, app, payloads, max_new_tokens=None):
        """
        Batch generate responses for multiple payloads on the same app.
        More efficient than individual calls.
        """
        max_tokens = max_new_tokens or MODEL_MAX_NEW_TOKENS
        results = []

        # Build all inputs
        input_texts = []
        for payload in payloads:
            user_input = app.format_input(payload)
            messages = self._build_chat_messages(app, user_input)
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_texts.append(input_text)

        try:
            start = time.time()

            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
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

            elapsed = time.time() - start

            # Decode each response
            for i, payload in enumerate(payloads):
                prompt_len = inputs['input_ids'].shape[1]
                new_tokens = outputs[i][prompt_len:]
                response_text = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )
                results.append({
                    'text': response_text,
                    'size': len(response_text),
                    'time': elapsed / len(payloads),
                    'tokens': len(new_tokens),
                    'app_name': app.name,
                    'payload': payload,
                })

        except Exception as e:
            # Fall back to individual generation
            for payload in payloads:
                results.append(self.generate(app, payload, max_tokens))

        return results

    def establish_baselines(self):
        """
        Establish baseline responses for each app.
        Send neutral inputs and record what a normal response looks like.
        """
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
                    'texts': texts,  # For embedding baseline
                }

                print(f"    {app.name}: avg_len={app.baseline['avg_length']:.0f} "
                      f"avg_time={app.baseline['avg_time']:.2f}s "
                      f"sample: {texts[0][:80]}...")
            else:
                app.baseline = {
                    'avg_length': 100, 'length_std': 10.0,
                    'avg_time': 1.0, 'time_std': 0.1,
                    'avg_tokens': 30, 'sample_responses': [],
                    'texts': [],
                }
                print(f"    {app.name}: FAILED to establish baseline")

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
