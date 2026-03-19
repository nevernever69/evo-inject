#!/usr/bin/env python3
"""
main_constrained.py - Constrained Vocabulary Zero-Seed Experiment

Same as zero-seed (no attack phrases), BUT token block generation and
refinement are CONSTRAINED to ~5000 interesting/separator/special tokens
instead of the full 128K vocabulary.

This tests the hypothesis: is the gradient necessary for discovering
adversarial tokens, or is the loss signal sufficient when the search
space is reduced to tokens that are likely to matter?

Three-way comparison:
  1. main.py           → seeded phrases, full vocab (easy mode)
  2. main_noseed.py    → no seeds, full vocab (blind search)
  3. main_constrained.py → no seeds, constrained vocab (this file)

If (3) finds real attacks but (2) doesn't, the paper contribution is:
  "Loss-guided evolutionary search discovers adversarial tokens without
   gradients when the token search space is appropriately constrained."
"""

import sys
import os
import time
import argparse
import random
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import (
    POPULATION_SIZE, GENERATIONS, STEPS_PER_LIFETIME,
    LOG_EVERY_N_GENERATIONS, SAVE_BEST_EVERY, CHECKPOINT_EVERY,
    TARGET_APPS, CLEAN_INPUTS, ARCHIVE_ATTACK_TYPES,
)
from organism import Organism
from reward import InjectionReward
from llm_target import LLMTarget
from evolution import Population
from measurement import Metrics, Logger, WandbLogger
from gp import program_complexity, extract_token_ids
from phrases import PhraseLibrary
from archive import MAPElitesArchive
from refinement import refine_token_blocks

from main import (
    flatten_components, run_lifetime, run_generation,
    print_findings_report, print_archive_report, print_gp_analysis,
    save_checkpoint, _try_promote_phrases,
)
from main_noseed import EmptyPhraseLibrary
from detailed_log import DetailedLogger


def main():
    parser = argparse.ArgumentParser(
        description='Constrained Vocab Zero-Seed: Evolutionary Attack Search'
    )
    parser.add_argument('--gens', type=int, default=GENERATIONS)
    parser.add_argument('--pop', type=int, default=POPULATION_SIZE)
    parser.add_argument('--steps', type=int, default=STEPS_PER_LIFETIME)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (3 gens, 5 pop, 10 steps)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='evo-inject-compositional')
    parser.add_argument('--no-loss', action='store_true',
                        help='Disable loss computation')
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable token block refinement')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.gens = 3
        args.pop = 5
        args.steps = 10

    config.STEPS_PER_LIFETIME = args.steps

    if args.device:
        config.MODEL_DEVICE = args.device

    compute_loss = not args.no_loss
    do_refine = not args.no_refine and compute_loss

    print("=" * 60)
    print("  CONSTRAINED VOCAB ZERO-SEED EXPERIMENT")
    print("  NO attack seeds + token search limited to ~5K tokens")
    print("  GP evolves structure, loss-guided refinement on tokens")
    print("  MAP-Elites maintains diverse attack portfolio")
    print("=" * 60)
    print(f"\n  Population:   {args.pop}")
    print(f"  Generations:  {args.gens}")
    print(f"  Steps/life:   {args.steps}")
    print(f"  Target apps:  {len(TARGET_APPS)}")
    print(f"  Loss signal:  {'enabled' if compute_loss else 'disabled'}")
    print(f"  Refinement:   {'enabled' if do_refine else 'disabled'}")
    print(f"  Device:       {config.MODEL_DEVICE}")
    print(f"  Seed phrases: NONE (filler only)")
    print(f"  Token vocab:  CONSTRAINED (~5K interesting/separator/special)")
    print()

    # Load LLM model
    target = LLMTarget(
        model_name=args.model or config.MODEL_NAME,
        device=config.MODEL_DEVICE,
    )
    target.load_model()

    # Enable constrained mode globally — ALL token generation uses ~5K pool
    import gp as _gp_module
    _gp_module.CONSTRAINED_MODE = True

    # Report constrained pool size
    from gp import INTERESTING_TOKENS, SEPARATOR_TOKENS, SPECIAL_TOKENS
    constrained_pool_size = len(INTERESTING_TOKENS) + len(SEPARATOR_TOKENS) + len(SPECIAL_TOKENS)
    print(f"  Constrained pool: {constrained_pool_size} tokens "
          f"({len(INTERESTING_TOKENS)} interesting + "
          f"{len(SEPARATOR_TOKENS)} separator + "
          f"{len(SPECIAL_TOKENS)} special)")

    # Initialize EMPTY phrase library (only fillers)
    phrase_library = EmptyPhraseLibrary()
    n_phrases = phrase_library.init_from_seeds(target.tokenizer)
    print(f"  Phrase library: {n_phrases} filler-only phrases (ZERO attack seeds)")

    # Establish baselines
    target.establish_baselines()

    # Initialize reward system
    reward_system = InjectionReward()
    for app in target.apps:
        if app.baseline and app.baseline.get('texts'):
            reward_system.establish_baseline(app.name, app.baseline['texts'])

    # Initialize MAP-Elites archive
    archive = MAPElitesArchive()
    print(f"  MAP-Elites archive: {archive.total_cells()} cells")

    # Initialize population
    metrics = Metrics()
    logger = Logger()
    detailed = DetailedLogger(log_dir=logger.log_dir, prefix='detailed_constrained')
    population = Population(size=args.pop, phrase_library=phrase_library)

    # wandb config
    wandb_config = {
        'experiment': 'constrained-vocab-zero-seed',
        'population_size': args.pop,
        'generations': args.gens,
        'steps_per_lifetime': args.steps,
        'target_apps': len(TARGET_APPS),
        'model': config.MODEL_NAME,
        'gp_max_length': config.GP_MAX_PROGRAM_LENGTH,
        'gp_mutation_rate': config.GP_MUTATION_RATE,
        'gp_crossover_rate': config.GP_CROSSOVER_RATE,
        'max_payload_tokens': config.MAX_PAYLOAD_TOKENS,
        'token_block_size': config.TOKEN_BLOCK_SIZE,
        'refine_steps': config.REFINE_STEPS,
        'phrase_library_size': n_phrases,
        'seed_attack_phrases': 0,
        'constrained_vocab': True,
        'constrained_pool_size': constrained_pool_size,
        'archive_cells': archive.total_cells(),
        'compute_loss': compute_loss,
        'do_refine': do_refine,
        'device': config.MODEL_DEVICE,
        'version': 'v3-constrained-vocab',
    }
    wb = WandbLogger(
        config_dict=wandb_config,
        project=args.project,
        enabled=args.wandb,
    )

    # Show initial info
    sample_org = population.organisms[0]
    sample_components, _ = sample_org.generate_payload(
        phrase_library=phrase_library,
        target_instruction_tokens=target.tokenize("Write a haiku."),
    )
    sample_flat = flatten_components(sample_components)
    sample_decoded = target.decode_tokens(sample_flat) if sample_flat else "(empty)"

    print(f"\n  Brain params: {sample_org.brain.network.num_params()}")
    print(f"  Library size: {len(sample_org.library.programs)} programs/organism")
    print(f"  Target apps: {[a.name for a in target.apps]}")
    print(f"  Sample payload: {len(sample_flat)} tokens → '{sample_decoded[:60]}'")

    if args.wandb and wb.enabled:
        print(f"  wandb:        {wb.run.url}")
    print()
    print("Starting CONSTRAINED VOCAB evolutionary attack search...")
    print("-" * 60)

    best_ever = None
    best_ever_fitness = -float('inf')

    try:
        for gen in range(1, args.gens + 1):
            gen_start = time.time()

            stats, gen_best, pre_evolve_avg = run_generation(
                population, target, reward_system, metrics, gen,
                phrase_library, archive,
                compute_loss=compute_loss, do_refine=do_refine,
                detailed=detailed, constrained=True,
            )

            gen_time = time.time() - gen_start

            if gen_best.fitness > best_ever_fitness:
                best_ever = gen_best
                best_ever_fitness = gen_best.fitness

            # Detailed local logging
            detailed.log_generation(
                gen, metrics, population, archive, phrase_library,
                gen_time, compute_loss, do_refine,
            )

            # Console logging
            logger.log_generation(
                gen, metrics, population,
                gen_best=gen_best,
                pre_evolve_avg=pre_evolve_avg,
                archive=archive,
                phrase_library=phrase_library,
            )
            logger.save_best(gen, gen_best, metrics)

            # wandb logging
            if wb.enabled:
                genome_stats = population.genome_stats()
                archive_stats = archive.stats()

                wb_data = {
                    'fitness/best': gen_best.fitness,
                    'fitness/avg': pre_evolve_avg,
                    'fitness/diversity': population.diversity(),
                    'learning/epsilon': population.best().brain.epsilon,
                    'learning/success_rate': (
                        metrics.gen_success_rate[-1]
                        if metrics.gen_success_rate else 0
                    ),
                    'gp/avg_program_length': genome_stats['program_length']['mean'],
                    'gp/unique_tokens': genome_stats.get('_vocab_tokens', 0),
                    'gp/total_phrases': genome_stats.get('total_phrases', 0),
                    'gp/total_blocks': genome_stats.get('total_token_blocks', 0),
                    'behavior/total_requests': metrics.total_requests,
                    'behavior/total_successes': metrics.total_successes,
                    'findings/unique_signatures': reward_system.total_unique_findings(),
                    'findings/total_count': sum(
                        len(f) for f in target.findings_by_app.values()
                    ),
                    'library/clonal_events': genome_stats.get('total_clonal_events', 0),
                    'library/replacements': genome_stats.get('total_replacements', 0),
                    'library/programs_successful': genome_stats.get('programs_with_success', 0),
                    'archive/coverage': archive_stats['coverage'],
                    'archive/occupied': archive_stats['occupied'],
                    'archive/total_fills': archive_stats['total_fills'],
                    'archive/total_updates': archive_stats['total_updates'],
                    'phrases/total': phrase_library.size(),
                    'phrases/promoted': phrase_library.stats()['promoted_phrases'],
                    'timing/gen_seconds': gen_time,
                }

                for app_cfg in TARGET_APPS:
                    name = app_cfg['name']
                    count = metrics.endpoint_counts.get(name, 0)
                    succ = metrics.endpoint_successes.get(name, 0)
                    wb_data[f'apps/{name}_rate'] = succ / count if count > 0 else 0

                if compute_loss:
                    avg_loss = gen_best.memory.avg_loss()
                    if avg_loss != float('inf'):
                        wb_data['loss/best_avg_loss'] = avg_loss

                if do_refine:
                    avg_refine = gen_best.memory.avg_refinement_improvement()
                    wb_data['refinement/avg_improvement'] = avg_refine

                for at, cov in archive_stats['by_attack_type'].items():
                    wb_data[f'archive/{at}_coverage'] = cov

                import wandb as wandb_lib
                wandb_lib.log(wb_data, step=gen)

            # Periodic reports
            if gen % (LOG_EVERY_N_GENERATIONS * 2) == 0:
                logger.log_app_distribution(metrics)
                print_findings_report(target, reward_system)
                print_archive_report(archive, target)
                print_gp_analysis(population, target, phrase_library)

            # Checkpoint
            if gen % CHECKPOINT_EVERY == 0:
                save_checkpoint(population, archive, phrase_library,
                                metrics, gen, logger.log_dir)

            # Stagnation injection
            if metrics.stagnation_detected():
                print("\n  Stagnation! Injecting diversity...")
                population.organisms.sort(key=lambda o: o.fitness, reverse=True)
                n_replace = max(1, len(population.organisms) // 5)
                for i in range(-n_replace, 0):
                    population.organisms[i] = Organism(
                        generation=gen, phrase_library=phrase_library,
                    )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("  FINAL REPORT (CONSTRAINED VOCAB ZERO-SEED)")
    print("=" * 60)

    print(f"\n  Generations:      {metrics.total_generations}")
    print(f"  Total queries:    {metrics.total_requests}")
    print(f"  Total successes:  {metrics.total_successes}")
    print(f"  Success rate:     {metrics.total_successes / max(1, metrics.total_requests):.1%}")
    print(f"  Runtime:          {time.time() - metrics.start_time:.1f}s")

    if best_ever:
        print(f"\n  Best organism:    {best_ever.summary()}")

    print_findings_report(target, reward_system)
    print_archive_report(archive, target)

    if best_ever:
        class _BestWrapper:
            def best(self): return best_ever
            @property
            def organisms(self): return [best_ever]
        print_gp_analysis(_BestWrapper(), target, phrase_library)

    # Save final report
    final_report = {
        'version': 'v3-constrained-vocab',
        'experiment': 'constrained-vocab-zero-seed',
        'seed_attack_phrases': 0,
        'constrained_vocab': True,
        'constrained_pool_size': constrained_pool_size,
        'generations': metrics.total_generations,
        'total_requests': metrics.total_requests,
        'total_successes': metrics.total_successes,
        'success_rate': metrics.total_successes / max(1, metrics.total_requests),
        'runtime_seconds': time.time() - metrics.start_time,
        'unique_findings': reward_system.total_unique_findings(),
        'best_fitness': best_ever_fitness,
        'best_organism': best_ever.summary() if best_ever else None,
        'compute_loss': compute_loss,
        'do_refine': do_refine,
        'archive': archive.to_dict(),
        'phrase_library': phrase_library.stats(),
        'findings': [],
        'per_app': {},
    }

    for f in target.all_findings():
        final_report['findings'].append({
            'type': f.get('type', 'unknown'),
            'severity': f.get('severity', '?'),
            'detail': f.get('detail', ''),
            'payload': f.get('payload', '')[:200],
            'payload_tokens': f.get('payload_tokens', [])[:20],
            'app_name': f.get('app_name', 'unknown'),
        })

    for app_cfg in TARGET_APPS:
        name = app_cfg['name']
        count = metrics.endpoint_counts.get(name, 0)
        succ = metrics.endpoint_successes.get(name, 0)
        final_report['per_app'][name] = {
            'requests': count,
            'successes': succ,
            'rate': succ / count if count > 0 else 0,
        }

    report_path = os.path.join(
        logger.log_dir, f'final_report_constrained_{int(time.time())}.json'
    )
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    print(f"\n  Final report:     {report_path}")
    print(f"  Logs saved to:    {logger.log_file}")
    detailed.final_save()
    if wb.enabled:
        print(f"  wandb run:        {wb.run.url}")
    print("=" * 60)

    wb.finish()


if __name__ == '__main__':
    main()
