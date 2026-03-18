#!/usr/bin/env python3
"""
main.py - GP-Evolving Prompt Injection Fuzzer

Organisms evolve GP programs that construct prompt injections from raw ASCII.
No hardcoded attack strategies. Evolution discovers injection patterns.

Target: Local Llama 3 8B running 5 different LLM applications, each with
system prompts and restrictions the fuzzer tries to violate.

The research question: Can evolution, starting from single characters
and basic string operations, independently discover prompt injection
patterns — without being told what prompt injection is?

Usage:
    python main.py                    # Run with defaults
    python main.py --quick            # Quick test (3 gens, 5 pop)
    python main.py --wandb            # Log to wandb
    python main.py --gens 50 --pop 20 # Custom params
    python main.py --no-mutator       # Disable LLM-as-mutator (raw GP only)
    python main.py --device cpu       # Run on CPU (slow, for testing)
"""

import sys
import os
import time
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import (
    POPULATION_SIZE, GENERATIONS, STEPS_PER_LIFETIME,
    LOG_EVERY_N_GENERATIONS, SAVE_BEST_EVERY,
    TARGET_APPS, CLEAN_INPUTS,
)
from organism import Organism
from reward import InjectionReward
from llm_target import LLMTarget
from mutator import Mutator
from evolution import Population
from measurement import Metrics, Logger, WandbLogger
from gp import extract_vocabulary, program_complexity, extract_strategies


def run_lifetime(organism, target, mutator, reward_system, metrics):
    """
    One organism's lifetime: probe LLM applications for injection vulnerabilities.

    Each step:
      1. Brain picks which app to target (RL)
      2. Library selects a GP program, generates payload + strategies
      3. Optional: mutator LLM refines raw GP output into coherent text
      4. Send to target LLM application
      5. Reward system analyzes response (embedding distance, leak detection, etc.)
      6. Library adapts: clone successful programs, replace failures
      7. Brain learns from experience
    """
    apps = target.apps
    last_result = None
    last_app_idx = 0

    for step in range(STEPS_PER_LIFETIME):
        # Brain observes state and picks target app
        state = organism.observe_state(apps, last_app_idx, last_result)
        app_idx = organism.choose_endpoint(state)
        app_idx = app_idx % len(apps)
        last_app_idx = app_idx

        app = apps[app_idx]

        # Library selects program, generates raw payload + strategies
        clean_input = random.choice(CLEAN_INPUTS)
        raw_payload, program_idx, strategies = organism.generate_payload(
            clean_input=clean_input,
            app_name=app.name,
        )

        # Optional: mutator refines into coherent text
        if mutator and strategies:
            payload = mutator.mutate(raw_payload, strategies, app.name)
        else:
            payload = raw_payload

        # Truncate to reasonable length
        payload = payload[:2000]

        # Send to target LLM
        response = target.generate(app, payload)
        if response is None or response.get('error'):
            organism.memory.record(app.name, app_idx, -10, False)
            organism.library.record_result(program_idx, -10, False)
            metrics.record_step(app.name, False)
            continue

        # Compute reward
        reward, findings = reward_system.analyze(response, app, payload)

        # Record
        found_something = len(findings) > 0
        response['had_findings'] = found_something
        # Add embedding distance to response for brain state
        for f in findings:
            if f.get('type') == 'embedding_deviation':
                response['embedding_distance'] = f.get('distance', 0.0)
                break
        last_result = response

        organism.memory.record(app.name, app_idx, reward, found_something)
        organism.library.record_result(program_idx, reward, found_something)
        organism.update_context(payload, response.get('text'), found_something)
        metrics.record_step(app.name, found_something)

        if found_something:
            active_program = organism.library.get_active_program()
            for f in findings:
                f['payload'] = payload[:200]
                f['app_name'] = app.name
                f['program'] = str(active_program)[:100]
                f['program_idx'] = program_idx
                f['strategies'] = strategies[:5]
                organism.memory.record_finding(f, payload, active_program)
                target.record_finding(app.name, f)

            # If system prompt was leaked, store hint for reactive use
            for f in findings:
                if f.get('type') == 'system_prompt_leak':
                    organism.record_system_hint(response.get('text', '')[:200])
                    break

        # Brain learns
        next_state = organism.observe_state(apps, app_idx, response)
        done = (step == STEPS_PER_LIFETIME - 1)
        organism.learn_from_experience(state, app_idx, reward, next_state, done)

    return organism


def run_generation(population, target, mutator, reward_system, metrics, gen):
    """Run one full generation — sequential organism evaluation."""
    n_organisms = len(population.organisms)

    for i, organism in enumerate(population.organisms):
        # Reset per-organism novelty tracking
        reward_system.reset_local()

        run_lifetime(organism, target, mutator, reward_system, metrics)

        if (i + 1) % 5 == 0 or (i + 1) == n_organisms:
            print(f"  Organism {i + 1}/{n_organisms} done", end='\r')

    print(f"  Organism {n_organisms}/{n_organisms} done")

    # Evaluate and record metrics BEFORE evolve
    population.evaluate_all()
    metrics.record_generation(population)

    pre_evolve_best = population.best()
    pre_evolve_avg = population.avg_fitness()

    # Track discoveries
    new_unique = reward_system.total_unique_findings() - getattr(metrics, '_last_unique_findings', 0)
    if new_unique > 0:
        for _ in range(new_unique):
            metrics.record_discovery(f"gen_{gen}")
    metrics._last_unique_findings = reward_system.total_unique_findings()

    # Evolve
    stats = population.evolve()
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
                strategies = f.get('strategies', [])
                print(f"    [{severity:8s}] {ftype}: {detail}")
                print(f"            payload: {payload}")
                if strategies:
                    print(f"            strategies: {strategies}")

    print(f"\n  Unique finding signatures: {reward_system.total_unique_findings()}")


def print_gp_analysis(population):
    """Analyze what GP programs have evolved."""
    best = population.best()
    print(f"\n  {'='*60}")
    print(f"  GP PROGRAM ANALYSIS")
    print(f"  {'='*60}")

    # Library overview
    lib_stats = best.library.stats()
    frag_stats = best.fragment_library.stats()
    print(f"\n  Library: {lib_stats['library_size']} programs")
    print(f"  Clonal events: {lib_stats['clonal_events']}")
    print(f"  Replacements: {lib_stats['replacements']}")
    print(f"  Avg program length: {lib_stats['avg_program_length']:.1f}")
    print(f"  Fragments discovered: {frag_stats['num_fragments']}")

    # Fragment library contents
    if best.fragment_library.fragments:
        print(f"\n  Discovered fragments:")
        for slot, text in list(best.fragment_library.fragments.items())[:10]:
            print(f"    [{slot}] '{text[:60]}'")

    # Per-program breakdown
    print(f"\n  Per-program stats:")
    for i, prog in enumerate(best.library.programs):
        uses = lib_stats['program_uses'][i]
        reward = lib_stats['program_rewards'][i]
        successes = lib_stats['program_successes'][i]
        rate = reward / uses if uses > 0 else 0
        strategies = extract_strategies(prog)
        marker = " *" if successes > 0 else ""
        strat_str = f" strategies={strategies}" if strategies else ""
        print(f"    [{i}] {prog.length():2d} insts | "
              f"uses={uses:3d} succ={successes:2d} "
              f"rate={rate:.1f}{marker}{strat_str}")

    # Best program details
    best_prog = best.library.best_program() or best.library.programs[0]
    print(f"\n  Best program ({best_prog.length()} instructions):")
    print(f"    {best_prog}")

    # Sample payloads
    print(f"\n  Sample payloads from best program:")
    for inp in CLEAN_INPUTS[:3]:
        payload = best_prog.execute(clean_input=inp)
        print(f"    input='{inp}' -> '{payload[:80]}'")

    # Vocabulary analysis
    chars, strings = extract_vocabulary(best_prog)
    if chars:
        print(f"\n  Characters evolved: {sorted(chars)[:30]}")
    if strings:
        print(f"  Strings evolved: {sorted(strings)[:10]}")

    # Strategies evolved
    prog_strategies = extract_strategies(best_prog)
    if prog_strategies:
        print(f"  Strategies evolved: {prog_strategies}")

    cx = program_complexity(best_prog)
    print(f"  Unique ops: {cx['unique_ops']}, Pushes: {cx['pushes']}, "
          f"Transforms: {cx['transforms']}, Strategies: {cx['strategies']}, "
          f"Fragments: {cx['fragments']}")


def main():
    parser = argparse.ArgumentParser(
        description='GP-Evolving Prompt Injection Fuzzer'
    )
    parser.add_argument('--gens', type=int, default=GENERATIONS)
    parser.add_argument('--pop', type=int, default=POPULATION_SIZE)
    parser.add_argument('--steps', type=int, default=STEPS_PER_LIFETIME)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (3 gens, 5 pop, 10 steps)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='parasite-llm')
    parser.add_argument('--no-mutator', action='store_true',
                        help='Disable LLM-as-mutator (raw GP only)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda, cpu, auto (default: from config)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/path (default: from config)')
    args = parser.parse_args()

    if args.quick:
        args.gens = 3
        args.pop = 5
        args.steps = 10

    config.STEPS_PER_LIFETIME = args.steps

    if args.device:
        config.MODEL_DEVICE = args.device

    print("=" * 60)
    print("  GP-EVOLVING PROMPT INJECTION FUZZER")
    print("  Organisms evolve injections from raw ASCII")
    print("  No hardcoded attack strategies")
    print("=" * 60)
    print(f"\n  Population:   {args.pop}")
    print(f"  Generations:  {args.gens}")
    print(f"  Steps/life:   {args.steps}")
    print(f"  Target apps:  {len(TARGET_APPS)}")
    print(f"  Mutator:      {'enabled' if not args.no_mutator else 'disabled (raw GP only)'}")
    print(f"  Device:       {config.MODEL_DEVICE}")
    print()

    # Load LLM model
    target = LLMTarget(
        model_name=args.model or config.MODEL_NAME,
        device=config.MODEL_DEVICE,
    )
    target.load_model()

    # Establish baselines
    target.establish_baselines()

    # Initialize reward system (embedding model)
    reward_system = InjectionReward()

    # Establish reward baselines from target baselines
    for app in target.apps:
        if app.baseline and app.baseline.get('texts'):
            reward_system.establish_baseline(app.name, app.baseline['texts'])

    # Initialize mutator (shares model with target)
    mutator = None
    if not args.no_mutator:
        mutator = Mutator(target)
        print(f"  Mutator initialized (shares model)")

    metrics = Metrics()
    logger = Logger()
    population = Population(size=args.pop)

    # wandb
    wandb_config = {
        'population_size': args.pop,
        'generations': args.gens,
        'steps_per_lifetime': args.steps,
        'target_apps': len(TARGET_APPS),
        'model': config.MODEL_NAME,
        'gp_max_length': config.GP_MAX_PROGRAM_LENGTH,
        'gp_mutation_rate': config.GP_MUTATION_RATE,
        'gp_crossover_rate': config.GP_CROSSOVER_RATE,
        'state_size': config.STATE_SIZE,
        'action_size': config.ACTION_SIZE,
        'mutator_enabled': not args.no_mutator,
        'device': config.MODEL_DEVICE,
    }
    wb = WandbLogger(
        config_dict=wandb_config,
        project=args.project,
        enabled=args.wandb,
    )

    # Show initial info
    sample_org = population.organisms[0]
    sample_payload, _, sample_strategies = sample_org.generate_payload('test')
    print(f"\n  Brain params: {sample_org.brain.network.num_params()}")
    print(f"  Library size: {len(sample_org.library.programs)} programs/organism")
    print(f"  Target apps: {[a.name for a in target.apps]}")
    print(f"  Sample GP program: {sample_org.library.programs[0]}")
    print(f"  Sample payload: {sample_payload[:60]}")
    if sample_strategies:
        print(f"  Sample strategies: {sample_strategies}")

    if args.wandb and wb.enabled:
        print(f"  wandb:        {wb.run.url}")
    print()
    print("Starting GP evolution...")
    print("-" * 60)

    best_ever = None
    best_ever_fitness = -float('inf')

    try:
        for gen in range(1, args.gens + 1):
            gen_start = time.time()

            stats, gen_best, pre_evolve_avg = run_generation(
                population, target, mutator, reward_system, metrics, gen,
            )

            gen_time = time.time() - gen_start

            if gen_best.fitness > best_ever_fitness:
                best_ever = gen_best
                best_ever_fitness = gen_best.fitness

            # Console logging
            logger.log_generation(
                gen, metrics, population,
                gen_best=gen_best,
                pre_evolve_avg=pre_evolve_avg,
            )
            logger.save_best(gen, gen_best, metrics)

            # wandb logging
            if wb.enabled:
                genome_stats = population.genome_stats()

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
                    'gp/vocab_chars': genome_stats.get('_vocab_chars', 0),
                    'gp/vocab_strings': genome_stats.get('_vocab_strings', 0),
                    'gp/transform_ratio': genome_stats['transform_ratio']['mean'],
                    'behavior/total_requests': metrics.total_requests,
                    'behavior/total_successes': metrics.total_successes,
                    'findings/unique_signatures': reward_system.total_unique_findings(),
                    'findings/total_count': sum(
                        len(f) for f in target.findings_by_app.values()
                    ),
                    'library/clonal_events': genome_stats.get('total_clonal_events', 0),
                    'library/replacements': genome_stats.get('total_replacements', 0),
                    'library/programs_successful': genome_stats.get('programs_with_success', 0),
                    'library/total_fragments': genome_stats.get('total_fragments', 0),
                    'timing/gen_seconds': gen_time,
                }

                # Per-app distribution
                for app_cfg in TARGET_APPS:
                    name = app_cfg['name']
                    count = metrics.endpoint_counts.get(name, 0)
                    succ = metrics.endpoint_successes.get(name, 0)
                    wb_data[f'apps/{name}_rate'] = succ / count if count > 0 else 0
                    wb_data[f'apps/{name}_count'] = count

                # Mutator stats
                if mutator:
                    mstats = mutator.stats()
                    wb_data['mutator/mutations'] = mstats['total_mutations']
                    wb_data['mutator/fallbacks'] = mstats['total_fallbacks']
                    wb_data['mutator/rate'] = mstats['mutation_rate']

                import wandb as wandb_lib
                wandb_lib.log(wb_data, step=gen)

            # Periodic detailed report
            if gen % (LOG_EVERY_N_GENERATIONS * 2) == 0:
                logger.log_endpoint_distribution(metrics)
                print_findings_report(target, reward_system)
                class _Wrapper:
                    def best(self_): return gen_best
                    @property
                    def organisms(self_): return [gen_best]
                _w = _Wrapper()
                _w.fragment_library = gen_best.fragment_library
                print_gp_analysis(_Wrapper())

            # Stagnation
            if metrics.stagnation_detected():
                print("\n  Stagnation! Injecting diversity...")
                population.organisms.sort(key=lambda o: o.fitness, reverse=True)
                n_replace = max(1, len(population.organisms) // 5)
                for i in range(-n_replace, 0):
                    population.organisms[i] = Organism(generation=gen)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

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
    else:
        print(f"\n  Best organism:    (none ran)")

    if mutator:
        mstats = mutator.stats()
        print(f"\n  Mutator stats:    {mstats['total_mutations']} mutations, "
              f"{mstats['total_fallbacks']} fallbacks "
              f"({mstats['mutation_rate']:.1%} mutation rate)")

    print_findings_report(target, reward_system)
    if best_ever:
        class _BestWrapper:
            def best(self): return best_ever
            @property
            def organisms(self): return [best_ever]
        print_gp_analysis(_BestWrapper())

    # Save final report to logs/
    import json as _json
    final_report = {
        'generations': metrics.total_generations,
        'total_requests': metrics.total_requests,
        'total_successes': metrics.total_successes,
        'success_rate': metrics.total_successes / max(1, metrics.total_requests),
        'runtime_seconds': time.time() - metrics.start_time,
        'unique_findings': reward_system.total_unique_findings(),
        'best_fitness': best_ever_fitness,
        'best_organism': best_ever.summary() if best_ever else None,
        'findings': [],
        'per_app': {},
    }
    for f in target.all_findings():
        final_report['findings'].append({
            'type': f.get('type', 'unknown'),
            'severity': f.get('severity', '?'),
            'detail': f.get('detail', ''),
            'payload': f.get('payload', '')[:200],
            'app_name': f.get('app_name', 'unknown'),
            'strategies': f.get('strategies', []),
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
    if mutator:
        final_report['mutator'] = mutator.stats()
    if best_ever:
        best_prog = best_ever.library.best_program() or best_ever.library.programs[0]
        final_report['best_program'] = str(best_prog)
        final_report['sample_payloads'] = [
            best_prog.execute(clean_input=inp)[:200]
            for inp in CLEAN_INPUTS[:5]
        ]
        final_report['fragments'] = best_ever.fragment_library.stats()

    report_path = os.path.join(logger.log_dir, f'final_report_{int(time.time())}.json')
    with open(report_path, 'w') as f:
        _json.dump(final_report, f, indent=2, default=str)
    print(f"\n  Final report:     {report_path}")

    print(f"  Logs saved to:    {logger.log_file}")
    if wb.enabled:
        print(f"  wandb run:        {wb.run.url}")
    print("=" * 60)

    wb.finish()


if __name__ == '__main__':
    main()
