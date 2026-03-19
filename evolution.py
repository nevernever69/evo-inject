"""
Evolution — Population-level selection, crossover, mutation

Works with the two-level architecture:
  - GP programs compose structures (phrases + token blocks)
  - MAP-Elites archive maintains diverse attack portfolio
  - Population evolves compositional strategy
"""

import random
import numpy as np
from organism import Organism
from gp import (
    crossover as gp_crossover, mutate as gp_mutate,
    program_complexity, extract_token_ids, programs_similar,
)
from config import (
    POPULATION_SIZE, ELITE_RATIO,
    GP_CROSSOVER_RATE, GP_MUTATION_RATE, GP_MAX_PROGRAM_LENGTH,
    LIBRARY_SIZE,
)


class Population:
    """A population of compositional attack organisms."""

    def __init__(self, size=POPULATION_SIZE, phrase_library=None):
        self.phrase_library = phrase_library
        self.organisms = [
            Organism(generation=0, phrase_library=phrase_library)
            for _ in range(size)
        ]
        self.generation = 0
        self.history = []

    def evaluate_all(self):
        for org in self.organisms:
            org.compute_fitness()

    def evolve(self, archive=None):
        """
        One generation of evolution.
        Optionally seeds from MAP-Elites archive for diversity.
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

        # Carry elites
        for elite in elites:
            child = Organism(
                library=elite.library.copy(),
                brain=elite.brain.copy(),
                generation=self.generation,
                phrase_library=self.phrase_library,
            )
            new_pop.append(child)

        # Inject archive elites for diversity (if available)
        if archive and archive.occupied_count() > 0:
            n_archive_inject = min(2, archive.occupied_count())
            for _ in range(n_archive_inject):
                cell = archive.random_occupied_cell()
                if cell and cell.program:
                    # Create organism from archive program
                    from organism import ProgramLibrary
                    from gp import Program
                    archive_prog = cell.program.copy()
                    lib = ProgramLibrary(
                        programs=[archive_prog] + [
                            Program(max_length=GP_MAX_PROGRAM_LENGTH,
                                    phrase_library=self.phrase_library)
                            for _ in range(LIBRARY_SIZE - 1)
                        ],
                        size=LIBRARY_SIZE,
                    )
                    child = Organism(
                        library=lib,
                        generation=self.generation,
                        phrase_library=self.phrase_library,
                    )
                    new_pop.append(child)

        target_size = len(self.organisms)
        while len(new_pop) < target_size:
            parent1 = self._tournament_select(ranked)
            parent2 = self._tournament_select(ranked)

            if random.random() < GP_CROSSOVER_RATE:
                child = parent1.make_offspring(
                    parent2, self.generation,
                    phrase_library=self.phrase_library,
                )
            else:
                better = parent1 if parent1.fitness >= parent2.fitness else parent2
                child_lib = better.library.copy()
                for i in range(len(child_lib.programs)):
                    child_lib.programs[i] = gp_mutate(
                        child_lib.programs[i], GP_MUTATION_RATE,
                        phrase_library=self.phrase_library,
                    )
                child = Organism(
                    library=child_lib,
                    brain=better.brain.copy(),
                    generation=self.generation,
                    phrase_library=self.phrase_library,
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
        if len(self.organisms) < 2:
            return 0.0

        n_unique = 0
        sample_size = min(50, len(self.organisms))
        sampled = random.sample(self.organisms, sample_size)

        best_progs = [
            o.library.best_program() or o.library.programs[0]
            for o in sampled
        ]

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
        all_programs = []
        for o in self.organisms:
            all_programs.extend(o.library.programs)

        lengths = [p.length() for p in all_programs]
        complexities = [program_complexity(p) for p in all_programs]

        all_tokens = set()
        for p in all_programs:
            all_tokens.update(extract_token_ids(p))

        total_clonal = sum(o.library.clonal_events for o in self.organisms)
        total_replacements = sum(o.library.replacements for o in self.organisms)
        programs_with_success = sum(
            sum(1 for s in o.library.program_successes if s > 0)
            for o in self.organisms
        )

        # Phrase vs token block balance
        total_phrases = sum(c['phrases'] for c in complexities)
        total_blocks = sum(c['token_blocks'] for c in complexities)

        stats = {
            'program_length': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
            },
            'unique_ops': {
                'mean': np.mean([c['unique_ops'] for c in complexities]),
            },
            'total_clonal_events': total_clonal,
            'total_replacements': total_replacements,
            'programs_with_success': programs_with_success,
            'total_programs': len(all_programs),
            'total_phrases': total_phrases,
            'total_token_blocks': total_blocks,
            'phrase_block_ratio': (
                total_phrases / max(1, total_blocks) if total_blocks > 0
                else float('inf')
            ),
            '_vocab_tokens': len(all_tokens),
        }

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
