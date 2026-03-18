"""
Evolution - GP-based selection, crossover, mutation across generations

Each organism carries a LIBRARY of GP programs (immune system model)
and a FRAGMENT LIBRARY of discovered text patterns.
Evolution operates at two levels:
  - Within lifetime: clonal selection adapts the library (organism.py)
  - Between lifetimes: population-level selection, crossover, mutation (this file)

Population lifecycle:
  1. Evaluate fitness of all organisms
  2. Select parents (tournament selection)
  3. Create offspring via library crossover + mutation + fragment inheritance
  4. Replace the population
"""

import random
import numpy as np
from organism import Organism
from gp import (
    crossover as gp_crossover, mutate as gp_mutate,
    program_complexity, extract_vocabulary, programs_similar,
)
from config import (
    POPULATION_SIZE, ELITE_RATIO,
    GP_CROSSOVER_RATE, GP_MUTATION_RATE, GP_MAX_PROGRAM_LENGTH,
    LIBRARY_SIZE,
)


class Population:
    """
    A population of GP-evolved organisms.
    """

    def __init__(self, size=POPULATION_SIZE):
        self.organisms = [Organism(generation=0) for _ in range(size)]
        self.generation = 0
        self.history = []

    def evaluate_all(self):
        """Compute fitness for every organism."""
        for org in self.organisms:
            org.compute_fitness()

    def evolve(self):
        """
        One generation of GP evolution.
        1. Sort by fitness
        2. Keep elites (same GP program + brain, fresh memory)
        3. Tournament select parents
        4. GP crossover + mutate to fill population
        """
        self.evaluate_all()
        self.generation += 1

        ranked = sorted(self.organisms, key=lambda o: o.fitness, reverse=True)

        fitnesses = [o.fitness for o in ranked]
        self.history.append({
            'generation': self.generation,
            'best_fitness': fitnesses[0],
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': fitnesses[-1],
            'std_fitness': np.std(fitnesses),
            'best_organism': ranked[0].summary(),
        })

        n_elite = max(1, int(len(self.organisms) * ELITE_RATIO))
        elites = ranked[:n_elite]

        new_pop = []

        # Elites carry over (same library + brain, fresh memory/stats)
        for elite in elites:
            child = Organism(
                library=elite.library.copy(),
                brain=elite.brain.copy(),
                generation=self.generation,
            )
            new_pop.append(child)

        # Fill rest via reproduction
        target_size = len(self.organisms)
        while len(new_pop) < target_size:
            parent1 = self._tournament_select(ranked)
            parent2 = self._tournament_select(ranked)

            if random.random() < GP_CROSSOVER_RATE:
                # Library crossover: best programs from each parent + GP crossover
                child = parent1.make_offspring(parent2, self.generation)
            else:
                # Asexual: clone library with per-program mutation
                better = parent1 if parent1.fitness >= parent2.fitness else parent2
                child_lib = better.library.copy()
                for i in range(len(child_lib.programs)):
                    child_lib.programs[i] = gp_mutate(
                        child_lib.programs[i], GP_MUTATION_RATE
                    )
                child = Organism(
                    library=child_lib,
                    brain=better.brain.copy(),
                    generation=self.generation,
                )

            new_pop.append(child)

        self.organisms = new_pop[:target_size]
        return self.history[-1]

    def _tournament_select(self, ranked, tournament_size=3):
        contestants = random.sample(ranked, min(tournament_size, len(ranked)))
        return max(contestants, key=lambda o: o.fitness)

    def best(self):
        return max(self.organisms, key=lambda o: o.fitness)

    def avg_fitness(self):
        fits = [o.fitness for o in self.organisms]
        return np.mean(fits) if fits else 0.0

    def diversity(self):
        """
        GP diversity: fraction of unique best-program structures.
        Compares the best program from each organism's library.
        Two programs are "similar" if >80% of their ops match.
        """
        if len(self.organisms) < 2:
            return 0.0

        n_unique = 0
        sample_size = min(50, len(self.organisms))
        sampled = random.sample(self.organisms, sample_size)

        best_progs = [o.library.best_program() or o.library.programs[0]
                      for o in sampled]

        for i in range(len(best_progs)):
            is_unique = True
            for j in range(i):
                if programs_similar(best_progs[i], best_progs[j]):
                    is_unique = False
                    break
            if is_unique:
                n_unique += 1

        return n_unique / sample_size

    def genome_stats(self):
        """GP program statistics across all libraries in population."""
        # Collect ALL programs from ALL libraries
        all_programs = []
        for o in self.organisms:
            all_programs.extend(o.library.programs)

        lengths = [p.length() for p in all_programs]
        complexities = [program_complexity(p) for p in all_programs]

        # Vocabulary analysis across all programs
        all_chars = set()
        all_strings = set()
        for p in all_programs:
            chars, strings = extract_vocabulary(p)
            all_chars.update(chars)
            all_strings.update(strings)

        # Library-level stats
        total_clonal = sum(o.library.clonal_events for o in self.organisms)
        total_replacements = sum(o.library.replacements for o in self.organisms)
        programs_with_success = sum(
            sum(1 for s in o.library.program_successes if s > 0)
            for o in self.organisms
        )

        # Fragment stats across population
        total_fragments = sum(
            len(o.fragment_library.fragments)
            for o in self.organisms
            if hasattr(o, 'fragment_library')
        )

        stats = {
            'program_length': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
            },
            'unique_ops': {
                'mean': np.mean([c['unique_ops'] for c in complexities]),
                'std': np.std([c['unique_ops'] for c in complexities]),
                'min': min(c['unique_ops'] for c in complexities),
                'max': max(c['unique_ops'] for c in complexities),
            },
            'transform_ratio': {
                'mean': np.mean([c['ratio'] for c in complexities]),
                'std': np.std([c['ratio'] for c in complexities]),
                'min': min(c['ratio'] for c in complexities),
                'max': max(c['ratio'] for c in complexities),
            },
            # Library-level aggregate stats
            'total_clonal_events': total_clonal,
            'total_replacements': total_replacements,
            'programs_with_success': programs_with_success,
            'total_programs': len(all_programs),
            'total_fragments': total_fragments,
        }

        stats['_vocab_chars'] = len(all_chars)
        stats['_vocab_strings'] = len(all_strings)

        return stats

    def summary(self):
        best = self.best()
        return (
            f"Gen {self.generation}: "
            f"best={best.fitness:.1f} "
            f"avg={self.avg_fitness():.1f} "
            f"div={self.diversity():.3f} "
            f"| {best.summary()}"
        )
