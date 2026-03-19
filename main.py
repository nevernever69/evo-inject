#!/usr/bin/env python3
"""
main.py - Compositional Evolutionary Attack Search

Two-level architecture:
  Level 1: GP programs evolve attack STRUCTURE (phrases + token blocks)
  Level 2: Loss-guided hill climbing refines raw token blocks

Key components:
  - Phrase library: shared coherent building blocks (seeded + evolved)
  - Token blocks: small raw token regions for sub-semantic exploration
  - Loss-guided refinement: hill climbing on CE loss without gradients
  - MAP-Elites archive: quality-diversity across attack types/apps/sizes

Usage:
    python main.py                    # Run with defaults
    python main.py --quick            # Quick test (3 gens, 5 pop)
    python main.py --wandb            # Log to wandb
    python main.py --gens 50 --pop 20 # Custom params
    python main.py --no-loss          # Disable loss computation
    python main.py --no-refine        # Disable token block refinement
    python main.py --device cpu       # Run on CPU (slow, for testing)
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
from detailed_log import DetailedLogger


def flatten_components(components):
    """Flatten component list into a single token ID list."""
    tokens = []
    for ctype, metadata, toks in components:
        tokens.extend(toks)
    return tokens


def run_lifetime(organism, target, reward_system, metrics, phrase_library,
                 archive, compute_loss=True, do_refine=True,
                 detailed=None, current_gen=0, constrained=False):
    """
    One organism's lifetime.

    Each step:
      1. Brain picks which app to target
      2. Library selects GP program → composes structure (phrases + token blocks)
      3. If do_refine: loss-guided hill climbing refines token blocks
      4. Assembled payload injected into model
      5. Reward computed (with coherence gate)
      6. Results fed to library (clonal selection) + archive (MAP-Elites)
      7. Brain learns from experience
    """
    apps = target.apps
    last_result = None
    last_app_idx = 0

    # Get separator tokens for the target model
    separator_tokens = target.get_separator_tokens()

    for step in range(STEPS_PER_LIFETIME):
        # Brain picks target app
        state = organism.observe_state(apps, last_app_idx, last_result)
        app_idx = organism.choose_endpoint(state)
        app_idx = app_idx % len(apps)
        last_app_idx = app_idx
        app = apps[app_idx]

        # Get target instruction tokens
        target_inst_tokens = target.tokenize(app.test_instruction)

        # Library selects program, produces structure
        components, program_idx = organism.generate_payload(
            phrase_library=phrase_library,
            target_instruction_tokens=target_inst_tokens,
            separator_tokens=separator_tokens,
        )

        if not components:
            organism.memory.record(app.name, app_idx, -10, False)
            organism.library.record_result(program_idx, -10, False)
            metrics.record_step(app.name, False)
            continue

        # Track which phrases were used (for phrase library feedback)
        used_phrase_indices = set()
        for ctype, metadata, _ in components:
            if ctype == 'phrase' and isinstance(metadata, int) and metadata >= 0:
                phrase_library.record_use(metadata)
                used_phrase_indices.add(metadata)

        # Loss-guided refinement of token blocks
        refine_stats = None
        has_token_blocks = any(ct == 'token_block' for ct, _, _ in components)

        if do_refine and has_token_blocks and compute_loss:
            components, refine_stats = refine_token_blocks(
                components, target, app, app.test_instruction,
                constrained=constrained,
            )
            if refine_stats:
                organism.memory.record_refinement(refine_stats)
                if detailed and refine_stats.get('improvements', 0) > 0:
                    detailed.log_refinement(
                        gen=current_gen,
                        organism_id=organism.id,
                        app_name=app.name,
                        block_idx=refine_stats.get('block_idx', 0),
                        loss_before=refine_stats.get('loss_before'),
                        loss_after=refine_stats.get('loss_after'),
                        tokens_before=refine_stats.get('tokens_before', []),
                        tokens_after=refine_stats.get('tokens_after', []),
                        improvements=refine_stats.get('improvements', 0),
                        steps=refine_stats.get('steps', 0),
                    )

        # Flatten to token list
        flat_tokens = flatten_components(components)

        if not flat_tokens:
            organism.memory.record(app.name, app_idx, -10, False)
            organism.library.record_result(program_idx, -10, False)
            metrics.record_step(app.name, False)
            continue

        # Generate response from target LLM
        response = target.generate_from_tokens(app, flat_tokens)
        if response is None or response.get('error'):
            organism.memory.record(app.name, app_idx, -10, False)
            organism.library.record_result(program_idx, -10, False)
            metrics.record_step(app.name, False)
            continue

        # Compute loss
        loss_info = None
        if compute_loss:
            loss = target.compute_loss(app, flat_tokens, app.test_instruction)
            loss_info = {
                'loss': loss,
                'baseline_loss': app.baseline_loss,
            }
            organism.memory.record_loss(loss)

        # Check coherence (for the gate)
        coherence_info = target.check_token_coherence(flat_tokens)

        # Compute reward
        reward, findings = reward_system.analyze(
            response, app, flat_tokens,
            loss_info=loss_info,
            coherence_info=coherence_info,
        )

        # Record results
        found_something = len(findings) > 0 and not any(
            f.get('type') == 'coherence_gate' for f in findings
        )
        response['had_findings'] = found_something
        for f in findings:
            if f.get('type') == 'embedding_deviation':
                response['embedding_distance'] = f.get('distance', 0.0)
                break
        last_result = response

        organism.memory.record(app.name, app_idx, reward, found_something)
        organism.library.record_result(program_idx, reward, found_something)
        metrics.record_step(app.name, found_something)

        if found_something:
            active_program = organism.library.get_active_program()
            payload_text = target.decode_tokens(flat_tokens)

            for f in findings:
                f['payload'] = payload_text[:200]
                f['payload_tokens'] = flat_tokens[:20]
                f['app_name'] = app.name
                f['program'] = str(active_program)[:100]
                f['program_idx'] = program_idx
                if refine_stats:
                    f['refinement'] = refine_stats
                organism.memory.record_finding(
                    f, flat_tokens, components=components,
                    program=active_program,
                )
                target.record_finding(app.name, f)

            # Update phrase success tracking
            for pidx in used_phrase_indices:
                phrase_library.record_success(pidx)

            # Try to insert into MAP-Elites archive
            attack_type = active_program.dominant_category(phrase_library)
            structure_class = active_program.structure_class()

            inserted, is_new, archive_bonus = archive.try_insert(
                attack_type=attack_type,
                app_name=app.name,
                structure_class=structure_class,
                fitness=reward,
                components=components,
                flat_tokens=flat_tokens,
                program=active_program,
                generation=organism.generation,
                response_text=response.get('text', ''),
                findings=findings,
                refinement_stats=refine_stats,
                phrase_indices=list(used_phrase_indices),
            )

            if archive_bonus > 0:
                reward += archive_bonus
                organism.memory.total_reward += archive_bonus

            # Detailed logging: every successful attack
            if detailed:
                components_summary = [
                    {'type': ct, 'meta': str(meta), 'n_tokens': len(toks)}
                    for ct, meta, toks in components
                ]
                detailed.log_attack(
                    gen=current_gen,
                    organism_id=organism.id,
                    app_name=app.name,
                    payload_text=payload_text,
                    payload_tokens=flat_tokens,
                    response_text=response.get('text', ''),
                    findings=findings,
                    reward=reward,
                    loss=loss_info.get('loss') if loss_info else None,
                    baseline_loss=loss_info.get('baseline_loss') if loss_info else None,
                    coherence_info=coherence_info,
                    program_str=str(active_program),
                    components_summary=components_summary,
                    refinement_stats=refine_stats,
                    phrase_indices=list(used_phrase_indices),
                    attack_type=attack_type,
                    structure_class=structure_class,
                )

                if inserted:
                    detailed.log_archive_fill(
                        gen=current_gen,
                        attack_type=attack_type,
                        app_name=app.name,
                        structure_class=structure_class,
                        fitness=reward,
                        payload_text=payload_text,
                        payload_tokens=flat_tokens,
                        response_text=response.get('text', ''),
                        program_str=str(active_program),
                        is_new=is_new,
                        findings=findings,
                    )

            # Try to promote new phrases from successful payloads
            if reward > 50:
                _try_promote_phrases(
                    components, flat_tokens, target, phrase_library, attack_type,
                    detailed=detailed, current_gen=current_gen,
                )

        # Brain learns
        next_state = organism.observe_state(apps, app_idx, response)
        done = (step == STEPS_PER_LIFETIME - 1)
        organism.learn_from_experience(state, app_idx, reward, next_state, done)

    return organism


def _try_promote_phrases(components, flat_tokens, target, phrase_library,
                         attack_type, detailed=None, current_gen=0):
    """Try to promote successful component sequences as new phrases."""
    for ctype, metadata, tokens in components:
        if ctype == 'token_block' and len(tokens) >= 3:
            # Decode and check if it's coherent enough to promote
            text = target.decode_tokens(tokens)
            if text and len(text.strip()) > 3:
                # Only promote if it has recognizable words
                words = text.strip().split()
                if len(words) >= 2:
                    old_size = phrase_library.size()
                    phrase_library.try_promote(
                        text.strip(), tokens, category=attack_type,
                    )
                    if detailed and phrase_library.size() > old_size:
                        detailed.log_promoted_phrase(
                            gen=current_gen,
                            text=text.strip(),
                            tokens=tokens,
                            category=attack_type,
                            source='token_block',
                        )


def run_generation(population, target, reward_system, metrics, gen,
                   phrase_library, archive,
                   compute_loss=True, do_refine=True, detailed=None,
                   constrained=False):
    """Run one full generation."""
    n_organisms = len(population.organisms)

    for i, organism in enumerate(population.organisms):
        reward_system.reset_local()

        run_lifetime(
            organism, target, reward_system, metrics, phrase_library,
            archive, compute_loss=compute_loss, do_refine=do_refine,
            detailed=detailed, current_gen=gen, constrained=constrained,
        )

        if (i + 1) % 5 == 0 or (i + 1) == n_organisms:
            print(f"  Organism {i + 1}/{n_organisms} done", end='\r')

    print(f"  Organism {n_organisms}/{n_organisms} done")

    # Evaluate fitness
    population.evaluate_all()
    metrics.record_generation(population)

    pre_evolve_best = population.best()
    pre_evolve_avg = population.avg_fitness()

    # Track discoveries
    new_unique = (
        reward_system.total_unique_findings()
        - getattr(metrics, '_last_unique_findings', 0)
    )
    if new_unique > 0:
        for _ in range(new_unique):
            metrics.record_discovery(f"gen_{gen}")
    metrics._last_unique_findings = reward_system.total_unique_findings()

    # Evolve (pass archive for diversity seeding)
    stats = population.evolve(archive=archive)
    return stats, pre_evolve_best, pre_evolve_avg


def print_findings_report(target, reward_system):
    """Print all unique findings."""
    findings = target.all_findings()
    if not findings:
        print("\n  No findings yet.")
        return

    print(f"\n  {'='*60}")
    print(f"  FINDINGS REPORT")
    print(f"  {'='*60}")

    by_app = {}
    for f in findings:
        app = f.get('app_name', 'unknown')
        if app not in by_app:
            by_app[app] = []
        by_app[app].append(f)

    for app, app_findings in by_app.items():
        print(f"\n  {app}:")
        seen_types = set()
        for f in app_findings:
            ftype = f.get('type', 'unknown')
            if ftype not in seen_types:
                seen_types.add(ftype)
                severity = f.get('severity', '?')
                detail = f.get('detail', '')
                payload = f.get('payload', '')[:80]
                print(f"    [{severity:8s}] {ftype}: {detail}")
                print(f"            payload: {payload}")
                tokens = f.get('payload_tokens', [])
                if tokens:
                    print(f"            tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                if f.get('refinement'):
                    rs = f['refinement']
                    print(f"            refine: {rs.get('improvements', 0)} improvements, "
                          f"loss {rs.get('loss_before', 0):.2f} → {rs.get('loss_after', 0):.2f}")

    print(f"\n  Unique finding signatures: {reward_system.total_unique_findings()}")


def print_archive_report(archive, target):
    """Print MAP-Elites archive status."""
    stats = archive.stats()
    print(f"\n  {'='*60}")
    print(f"  MAP-ELITES ARCHIVE")
    print(f"  {'='*60}")
    print(f"  Coverage: {stats['occupied']}/{stats['total_cells']} "
          f"({stats['coverage']:.1%})")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  New cells filled: {stats['total_fills']}")

    if stats['fitness']['n'] > 0:
        print(f"  Fitness: mean={stats['fitness']['mean']:.1f} "
              f"max={stats['fitness']['max']:.1f}")

    print(f"\n  By attack type:")
    for at, cov in stats['by_attack_type'].items():
        best = stats['best_per_attack_type'].get(at, 0)
        bar = '#' * int(cov * 20)
        print(f"    {at:25s} {cov:5.1%} best={best:.0f} {bar}")

    print(f"\n  By app:")
    for app, cov in stats['by_app'].items():
        best = stats['best_per_app'].get(app, 0)
        bar = '#' * int(cov * 20)
        print(f"    {app:25s} {cov:5.1%} best={best:.0f} {bar}")


def print_gp_analysis(population, target, phrase_library):
    """Analyze what GP programs have evolved."""
    best = population.best()
    print(f"\n  {'='*60}")
    print(f"  GP PROGRAM ANALYSIS (COMPOSITIONAL)")
    print(f"  {'='*60}")

    lib_stats = best.library.stats()
    print(f"\n  Library: {lib_stats['library_size']} programs")
    print(f"  Clonal events: {lib_stats['clonal_events']}")
    print(f"  Replacements: {lib_stats['replacements']}")
    print(f"  Avg program length: {lib_stats['avg_program_length']:.1f}")

    # Per-program breakdown
    print(f"\n  Per-program stats:")
    for i, prog in enumerate(best.library.programs):
        uses = lib_stats['program_uses'][i]
        reward = lib_stats['program_rewards'][i]
        successes = lib_stats['program_successes'][i]
        rate = reward / uses if uses > 0 else 0
        marker = " *" if successes > 0 else ""
        cat = prog.dominant_category(phrase_library)
        print(f"    [{i}] {prog.length():2d} insts | "
              f"uses={uses:3d} succ={successes:2d} "
              f"rate={rate:.1f} cat={cat}{marker}")

    # Best program details
    best_prog = best.library.best_program() or best.library.programs[0]
    print(f"\n  Best program ({best_prog.length()} instructions):")
    print(f"    {best_prog}")

    # Sample execution
    print(f"\n  Sample payload from best program:")
    target_inst_tokens = target.tokenize("Write a haiku about cats.")
    components = best_prog.execute(
        phrase_library=phrase_library,
        target_instruction_tokens=target_inst_tokens,
    )
    flat = flatten_components(components)
    decoded = target.decode_tokens(flat) if flat else "(empty)"
    print(f"    Components: {len(components)}")
    for ct, meta, toks in components[:8]:
        tok_text = target.decode_tokens(toks)[:60] if toks else ""
        print(f"      [{ct:12s}] {len(toks):3d} tokens → '{tok_text}'")
    print(f"    Full payload: {len(flat)} tokens → '{decoded[:100]}'")

    # Phrase library stats
    phrase_stats = phrase_library.stats()
    print(f"\n  Phrase library: {phrase_stats['total_phrases']} phrases "
          f"({phrase_stats['seed_phrases']} seed, "
          f"{phrase_stats['promoted_phrases']} promoted)")
    print(f"  Total phrase uses: {phrase_stats['total_uses']}")
    print(f"  Total phrase successes: {phrase_stats['total_successes']}")

    # Top phrases by success rate
    phrases_by_rate = sorted(
        enumerate(phrase_library.phrases),
        key=lambda x: x[1]['successes'] / max(x[1]['uses'], 1),
        reverse=True,
    )
    print(f"\n  Top phrases:")
    for idx, p in phrases_by_rate[:5]:
        rate = p['successes'] / max(p['uses'], 1)
        print(f"    [{idx:2d}] uses={p['uses']:3d} succ={p['successes']:3d} "
              f"rate={rate:.2f} [{p['category']}] '{p['text'][:50]}'")

    # Complexity
    cx = program_complexity(best_prog)
    print(f"\n  Program complexity:")
    print(f"    Phrases: {cx['phrases']}, Token blocks: {cx['token_blocks']}, "
          f"Separators: {cx['separators']}, Unique tokens: {cx['unique_tokens']}")
    print(f"    Block/phrase ratio: {cx['ratio']:.2f}")

    # Loss info
    avg_loss = best.memory.avg_loss()
    if avg_loss != float('inf'):
        print(f"  Average loss: {avg_loss:.2f}")
    avg_refine = best.memory.avg_refinement_improvement()
    if avg_refine > 0:
        print(f"  Avg refinement improvement: {avg_refine:.3f}")


def save_checkpoint(population, archive, phrase_library, metrics, gen,
                    log_dir='logs'):
    """Save full checkpoint for resuming."""
    ckpt = {
        'generation': gen,
        'archive': archive.to_dict(),
        'phrase_library_stats': phrase_library.stats(),
        'population_size': len(population.organisms),
        'best_fitness': population.best().fitness,
        'metrics_snapshot': {
            'total_requests': metrics.total_requests,
            'total_successes': metrics.total_successes,
            'total_discoveries': metrics.total_discoveries,
        },
    }
    path = os.path.join(log_dir, f'checkpoint_gen_{gen}.json')
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=str)
    print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compositional Evolutionary Attack Search'
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
    print("  COMPOSITIONAL EVOLUTIONARY ATTACK SEARCH")
    print("  GP evolves attack structure (phrases + token blocks)")
    print("  Loss-guided hill climbing refines token blocks")
    print("  MAP-Elites maintains diverse attack portfolio")
    print("=" * 60)
    print(f"\n  Population:   {args.pop}")
    print(f"  Generations:  {args.gens}")
    print(f"  Steps/life:   {args.steps}")
    print(f"  Target apps:  {len(TARGET_APPS)}")
    print(f"  Loss signal:  {'enabled' if compute_loss else 'disabled'}")
    print(f"  Refinement:   {'enabled' if do_refine else 'disabled'}")
    print(f"  Device:       {config.MODEL_DEVICE}")
    print()

    # Load LLM model
    target = LLMTarget(
        model_name=args.model or config.MODEL_NAME,
        device=config.MODEL_DEVICE,
    )
    target.load_model()

    # Initialize phrase library with seed phrases
    phrase_library = PhraseLibrary()
    n_phrases = phrase_library.init_from_seeds(target.tokenizer)
    print(f"  Phrase library: {n_phrases} seed phrases loaded")

    # Establish baselines
    target.establish_baselines()

    # Initialize reward system (embedding model)
    reward_system = InjectionReward()

    # Establish reward baselines from target baselines
    for app in target.apps:
        if app.baseline and app.baseline.get('texts'):
            reward_system.establish_baseline(app.name, app.baseline['texts'])

    # Initialize MAP-Elites archive
    archive = MAPElitesArchive()
    print(f"  MAP-Elites archive: {archive.total_cells()} cells "
          f"({len(ARCHIVE_ATTACK_TYPES)} types x {len(TARGET_APPS)} apps x 3 sizes)")

    # Initialize population with phrase library
    metrics = Metrics()
    logger = Logger()
    detailed = DetailedLogger(log_dir=logger.log_dir, prefix='detailed')
    population = Population(size=args.pop, phrase_library=phrase_library)

    # wandb config
    wandb_config = {
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
        'archive_cells': archive.total_cells(),
        'compute_loss': compute_loss,
        'do_refine': do_refine,
        'device': config.MODEL_DEVICE,
        'version': 'v3-compositional',
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
    print(f"  Sample program: {sample_org.library.programs[0]}")
    print(f"  Sample components: {len(sample_components)}")
    print(f"  Sample payload: {len(sample_flat)} tokens → '{sample_decoded[:60]}'")

    if args.wandb and wb.enabled:
        print(f"  wandb:        {wb.run.url}")
    print()
    print("Starting compositional evolutionary attack search...")
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
                detailed=detailed,
            )

            gen_time = time.time() - gen_start

            if gen_best.fitness > best_ever_fitness:
                best_ever = gen_best
                best_ever_fitness = gen_best.fitness

            # Detailed local logging (every generation)
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

                # Per-app
                for app_cfg in TARGET_APPS:
                    name = app_cfg['name']
                    count = metrics.endpoint_counts.get(name, 0)
                    succ = metrics.endpoint_successes.get(name, 0)
                    wb_data[f'apps/{name}_rate'] = succ / count if count > 0 else 0

                # Loss tracking
                if compute_loss:
                    avg_loss = gen_best.memory.avg_loss()
                    if avg_loss != float('inf'):
                        wb_data['loss/best_avg_loss'] = avg_loss

                # Refinement tracking
                if do_refine:
                    avg_refine = gen_best.memory.avg_refinement_improvement()
                    wb_data['refinement/avg_improvement'] = avg_refine

                # Per-attack-type coverage
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
    print("  FINAL REPORT")
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
        'version': 'v3-compositional',
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
        logger.log_dir, f'final_report_{int(time.time())}.json'
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
