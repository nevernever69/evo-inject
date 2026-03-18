"""
Measurement - Metrics, verification, and anti-self-deception

Tracks GP evolution progress: program complexity, vocabulary growth,
fitness trajectories, and prompt injection discovery rates.
Optional wandb integration for live dashboards.
"""

import time
import json
import os
import numpy as np
from config import (
    LOG_EVERY_N_GENERATIONS, SAVE_BEST_EVERY,
    TARGET_APPS,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Metrics:
    """
    Tracks all measurable quantities across the run.
    """

    def __init__(self):
        self.start_time = time.time()

        # Per-generation tracking
        self.gen_fitness = []
        self.gen_avg_fitness = []
        self.gen_diversity = []
        self.gen_success_rate = []
        self.gen_program_lengths = []
        self.gen_vocab_size = []

        # Cumulative
        self.total_requests = 0
        self.total_successes = 0
        self.total_discoveries = 0
        self.total_generations = 0

        # Per-app tracking
        self.endpoint_counts = {}
        self.endpoint_successes = {}

        # Novelty tracking
        self.novel_behaviors = []

    def record_generation(self, population):
        """Record metrics for one generation."""
        self.total_generations += 1

        best = population.best()
        self.gen_fitness.append(best.fitness)
        self.gen_avg_fitness.append(population.avg_fitness())
        self.gen_diversity.append(population.diversity())

        # Aggregate organism stats
        success_rates = []
        apps_found = set()
        program_lengths = []

        for org in population.organisms:
            mem = org.memory
            success_rates.append(mem.success_rate())
            apps_found.update(mem.apps_with_findings)

            # Use library best program length
            best_prog = org.library.best_program() or org.library.programs[0]
            program_lengths.append(best_prog.length())

        self.gen_success_rate.append(
            np.mean(success_rates) if success_rates else 0
        )
        self.gen_program_lengths.append(
            np.mean(program_lengths) if program_lengths else 0
        )

    def record_step(self, app_name, success):
        """Track per-step app usage."""
        self.endpoint_counts[app_name] = self.endpoint_counts.get(app_name, 0) + 1
        self.total_requests += 1
        if success:
            self.endpoint_successes[app_name] = self.endpoint_successes.get(app_name, 0) + 1
            self.total_successes += 1

    def record_discovery(self, description):
        self.total_discoveries += 1
        self.novel_behaviors.append({
            'time': time.time() - self.start_time,
            'generation': self.total_generations,
            'description': description,
        })

    def is_improving(self, window=10):
        if len(self.gen_fitness) < window * 2:
            return None
        recent = np.mean(self.gen_fitness[-window:])
        previous = np.mean(self.gen_fitness[-window * 2:-window])
        return recent > previous

    def improvement_rate(self, window=10):
        if len(self.gen_fitness) < window * 2:
            return 0.0
        recent = np.mean(self.gen_fitness[-window:])
        previous = np.mean(self.gen_fitness[-window * 2:-window])
        if previous == 0:
            return 0.0
        return (recent - previous) / abs(previous)

    def stagnation_detected(self, window=20, threshold=0.01):
        if len(self.gen_fitness) < window:
            return False
        recent = self.gen_fitness[-window:]
        return np.std(recent) / (np.mean(recent) + 1e-10) < threshold


class Logger:
    """Logging and saving results."""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'run_{int(time.time())}.jsonl')

    def log_generation(self, gen, metrics, population, gen_best=None, pre_evolve_avg=None):
        # Always log gen 1 and every N generations
        if gen != 1 and gen % LOG_EVERY_N_GENERATIONS != 0:
            return

        best = gen_best or population.best()

        best_prog = best.library.best_program() or best.library.programs[0]
        lib_stats = best.library.stats()
        frag_stats = best.fragment_library.stats()
        sample_payload = best_prog.execute(clean_input='test')

        # Use pre-evolve avg if provided (post-evolve organisms have fresh memory = 0)
        avg_fit = pre_evolve_avg if pre_evolve_avg is not None else population.avg_fitness()

        entry = {
            'generation': gen,
            'time': time.time(),
            'best_fitness': best.fitness,
            'avg_fitness': avg_fit,
            'diversity': population.diversity(),
            'best_summary': best.summary(),
            'total_requests': metrics.total_requests,
            'total_successes': metrics.total_successes,
            'total_discoveries': metrics.total_discoveries,
            'best_program_length': best_prog.length(),
            'best_payload_sample': sample_payload[:100],
            'library_size': lib_stats['library_size'],
            'clonal_events': lib_stats['clonal_events'],
            'replacements': lib_stats['replacements'],
            'fragments_discovered': frag_stats['num_fragments'],
        }

        print(f"\n{'='*60}")
        print(f"Generation {gen}")
        print(f"{'='*60}")
        print(f"  Best fitness:  {best.fitness:.1f}")
        print(f"  Avg fitness:   {avg_fit:.1f}")
        print(f"  Diversity:     {population.diversity():.3f}")
        print(f"  Requests:      {metrics.total_requests}")
        print(f"  Successes:     {metrics.total_successes}")
        print(f"  Discoveries:   {metrics.total_discoveries}")
        print(f"  Best organism: {best.summary()}")

        # GP program info
        print(f"  Best program:  {best_prog}")
        print(f"  Sample payload: {sample_payload[:80]}")

        # Library stats
        print(f"  Library:       {lib_stats['library_size']} progs, "
              f"clonal={lib_stats['clonal_events']}, "
              f"replaced={lib_stats['replacements']}")

        # Fragment stats
        print(f"  Fragments:     {frag_stats['num_fragments']} promoted, "
              f"{frag_stats['num_candidates']} candidates")

        # GP stats
        genome_stats = population.genome_stats()
        print(f"  Avg prog len:  {genome_stats['program_length']['mean']:.1f}")
        print(f"  Vocab chars:   {genome_stats.get('_vocab_chars', '?')}")
        print(f"  Vocab strings: {genome_stats.get('_vocab_strings', '?')}")
        print(f"  Pop clonal:    {genome_stats.get('total_clonal_events', 0)} "
              f"replacements: {genome_stats.get('total_replacements', 0)} "
              f"fragments: {genome_stats.get('total_fragments', 0)}")

        improving = metrics.is_improving()
        if improving is not None:
            print(f"  Improving: {'Yes' if improving else 'No'}")
            print(f"  Improvement rate: {metrics.improvement_rate():.1%}")

        if metrics.stagnation_detected():
            print(f"  WARNING: Stagnation detected!")

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def save_best(self, gen, organism, metrics):
        # Always save gen 1 and every N generations
        if gen != 1 and gen % SAVE_BEST_EVERY != 0:
            return

        save_path = os.path.join(self.log_dir, f'best_gen_{gen}.json')
        best_prog = organism.library.best_program() or organism.library.programs[0]
        lib_stats = organism.library.stats()
        frag_stats = organism.fragment_library.stats()

        state = {
            'generation': gen,
            'fitness': organism.fitness,
            'program_length': best_prog.length(),
            'program_repr': str(best_prog),
            'sample_payloads': [
                best_prog.execute(clean_input=inp)[:200]
                for inp in ['test', '1', 'admin', 'hello']
            ],
            'brain_params': int(organism.brain.network.num_params()),
            'library_stats': lib_stats,
            'fragment_stats': frag_stats,
            'memory_summary': {
                'total_reward': organism.memory.total_reward,
                'success_rate': organism.memory.success_rate(),
                'unique_apps': organism.memory.unique_endpoints_found(),
                'finding_count': organism.memory.finding_count(),
            },
            'metrics_snapshot': {
                'total_requests': metrics.total_requests,
                'total_successes': metrics.total_successes,
                'total_discoveries': metrics.total_discoveries,
            },
        }

        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

    def log_app_distribution(self, metrics):
        """Show per-app request distribution and success rates."""
        print(f"\n  App Distribution:")
        for app_cfg in TARGET_APPS:
            name = app_cfg['name']
            count = metrics.endpoint_counts.get(name, 0)
            success = metrics.endpoint_successes.get(name, 0)
            rate = success / count if count > 0 else 0
            max_count = max(metrics.endpoint_counts.values()) if metrics.endpoint_counts else 1
            bar = '#' * int(count / max(1, max_count) * 20)
            print(f"    {name:20s} {count:5d} ({rate:.0%}) {bar}")

    # Keep old name as alias for backward compat with main.py
    log_endpoint_distribution = log_app_distribution


class WandbLogger:
    """
    Weights & Biases integration.
    Logs GP evolution metrics for live dashboards.
    """

    def __init__(self, config_dict=None, project='parasite-llm', enabled=True):
        self.enabled = enabled and HAS_WANDB
        self.run = None

        if not self.enabled:
            return

        try:
            self.run = wandb.init(
                project=project,
                config=config_dict or {},
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}")
            self.enabled = False

    def finish(self):
        if self.enabled and self.run:
            wandb.finish()
