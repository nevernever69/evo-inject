"""
MAP-Elites Quality-Diversity Archive

Maintains a grid of the best attack found for each behavioral niche.
This structurally prevents population collapse — the archive MUST
contain diverse solutions by construction.

Behavior dimensions:
  1. attack_type: instruction_override, role_play, authority,
                  context_confusion, information_extraction, token_exploit
  2. target_app: summarizer, email_assistant, code_helper,
                 customer_support, translator
  3. structure_class: short (<20 tokens), medium (20-40), long (>40)

Total cells: 6 × 5 × 3 = 90

Each cell stores:
  - The best-performing attack (components + flat tokens)
  - Its fitness score
  - The GP program that generated it
  - Metadata (generation found, refinement stats, etc.)

Selection for variation:
  - Uniform random over occupied cells (not fitness-weighted)
  - This ensures diversity pressure
"""

import random
import numpy as np
from config import (
    ARCHIVE_ATTACK_TYPES, ARCHIVE_STRUCTURE_CLASSES,
    TARGET_APPS, REWARD_ARCHIVE_FILL,
)


class ArchiveCell:
    """One cell in the MAP-Elites grid."""

    __slots__ = [
        'attack_type', 'app_name', 'structure_class',
        'fitness', 'components', 'flat_tokens', 'program',
        'generation', 'response_text', 'findings',
        'refinement_stats', 'phrase_indices',
    ]

    def __init__(self, attack_type, app_name, structure_class):
        self.attack_type = attack_type
        self.app_name = app_name
        self.structure_class = structure_class
        self.fitness = None
        self.components = None
        self.flat_tokens = None
        self.program = None
        self.generation = None
        self.response_text = None
        self.findings = None
        self.refinement_stats = None
        self.phrase_indices = None

    @property
    def occupied(self):
        return self.fitness is not None

    def update(self, fitness, components, flat_tokens, program,
               generation, response_text=None, findings=None,
               refinement_stats=None, phrase_indices=None):
        """Update this cell if the new solution is better."""
        if self.fitness is not None and fitness <= self.fitness:
            return False

        self.fitness = fitness
        self.components = components
        self.flat_tokens = list(flat_tokens) if flat_tokens else []
        self.program = program
        self.generation = generation
        self.response_text = response_text[:500] if response_text else None
        self.findings = findings
        self.refinement_stats = refinement_stats
        self.phrase_indices = phrase_indices
        return True

    def to_dict(self):
        """Serialize for saving."""
        return {
            'attack_type': self.attack_type,
            'app_name': self.app_name,
            'structure_class': self.structure_class,
            'fitness': self.fitness,
            'flat_tokens': self.flat_tokens[:30] if self.flat_tokens else [],
            'generation': self.generation,
            'response_text': self.response_text,
            'findings': self.findings,
            'refinement_stats': self.refinement_stats,
            'phrase_indices': list(self.phrase_indices) if self.phrase_indices else [],
        }


class MAPElitesArchive:
    """
    MAP-Elites quality-diversity archive.

    Grid indexed by (attack_type, app_name, structure_class).
    Each cell stores the best attack found for that behavioral niche.
    """

    def __init__(self):
        self.grid = {}  # (attack_type, app_name, structure_class) -> ArchiveCell
        self.app_names = [app['name'] for app in TARGET_APPS]

        # Pre-create all cells
        for at in ARCHIVE_ATTACK_TYPES:
            for app_name in self.app_names:
                for sc in ARCHIVE_STRUCTURE_CLASSES:
                    key = (at, app_name, sc)
                    self.grid[key] = ArchiveCell(at, app_name, sc)

        self.total_updates = 0
        self.total_fills = 0  # New cells filled (was empty before)

    def try_insert(self, attack_type, app_name, structure_class,
                   fitness, components, flat_tokens, program,
                   generation, response_text=None, findings=None,
                   refinement_stats=None, phrase_indices=None):
        """
        Try to insert an attack into the archive.

        Returns:
            (inserted: bool, is_new_cell: bool, archive_bonus: float)
        """
        key = (attack_type, app_name, structure_class)

        if key not in self.grid:
            return False, False, 0.0

        cell = self.grid[key]
        was_empty = not cell.occupied

        inserted = cell.update(
            fitness=fitness,
            components=components,
            flat_tokens=flat_tokens,
            program=program,
            generation=generation,
            response_text=response_text,
            findings=findings,
            refinement_stats=refinement_stats,
            phrase_indices=phrase_indices,
        )

        if inserted:
            self.total_updates += 1
            if was_empty:
                self.total_fills += 1
                return True, True, REWARD_ARCHIVE_FILL
            return True, False, REWARD_ARCHIVE_FILL * 0.25  # Smaller bonus for improvement

        return False, False, 0.0

    def random_occupied_cell(self):
        """
        Select a random occupied cell (uniform over occupied cells).
        This is the key to diversity — selection is not fitness-weighted.
        """
        occupied = [cell for cell in self.grid.values() if cell.occupied]
        if not occupied:
            return None
        return random.choice(occupied)

    def get_cell(self, attack_type, app_name, structure_class):
        """Get a specific cell."""
        return self.grid.get((attack_type, app_name, structure_class))

    def coverage(self):
        """Fraction of cells that are occupied."""
        total = len(self.grid)
        occupied = sum(1 for cell in self.grid.values() if cell.occupied)
        return occupied / total if total > 0 else 0.0

    def occupied_count(self):
        """Number of occupied cells."""
        return sum(1 for cell in self.grid.values() if cell.occupied)

    def total_cells(self):
        return len(self.grid)

    def best_per_app(self):
        """Best fitness per app across all attack types and structure classes."""
        best = {}
        for cell in self.grid.values():
            if cell.occupied:
                app = cell.app_name
                if app not in best or cell.fitness > best[app]:
                    best[app] = cell.fitness
        return best

    def best_per_attack_type(self):
        """Best fitness per attack type across all apps and structure classes."""
        best = {}
        for cell in self.grid.values():
            if cell.occupied:
                at = cell.attack_type
                if at not in best or cell.fitness > best[at]:
                    best[at] = cell.fitness
        return best

    def coverage_by_app(self):
        """Coverage per app."""
        counts = {name: {'total': 0, 'occupied': 0} for name in self.app_names}
        for key, cell in self.grid.items():
            _, app_name, _ = key
            counts[app_name]['total'] += 1
            if cell.occupied:
                counts[app_name]['occupied'] += 1

        return {
            name: c['occupied'] / c['total'] if c['total'] > 0 else 0
            for name, c in counts.items()
        }

    def coverage_by_attack_type(self):
        """Coverage per attack type."""
        counts = {at: {'total': 0, 'occupied': 0} for at in ARCHIVE_ATTACK_TYPES}
        for key, cell in self.grid.items():
            at, _, _ = key
            counts[at]['total'] += 1
            if cell.occupied:
                counts[at]['occupied'] += 1

        return {
            at: c['occupied'] / c['total'] if c['total'] > 0 else 0
            for at, c in counts.items()
        }

    def fitness_stats(self):
        """Statistics across all occupied cells."""
        fitnesses = [cell.fitness for cell in self.grid.values() if cell.occupied]
        if not fitnesses:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
        return {
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses)),
            'n': len(fitnesses),
        }

    def stats(self):
        """Full summary statistics."""
        return {
            'total_cells': self.total_cells(),
            'occupied': self.occupied_count(),
            'coverage': self.coverage(),
            'total_updates': self.total_updates,
            'total_fills': self.total_fills,
            'fitness': self.fitness_stats(),
            'by_app': self.coverage_by_app(),
            'by_attack_type': self.coverage_by_attack_type(),
            'best_per_app': self.best_per_app(),
            'best_per_attack_type': self.best_per_attack_type(),
        }

    def get_elites_for_seeding(self, n=5):
        """
        Get top-N elites from archive for seeding into the population.
        Used to inject archive diversity back into the evolutionary population.
        """
        occupied = [(cell, cell.fitness) for cell in self.grid.values() if cell.occupied]
        if not occupied:
            return []

        # Sort by fitness, take top N
        occupied.sort(key=lambda x: x[1], reverse=True)
        return [cell for cell, _ in occupied[:n]]

    def to_dict(self):
        """Serialize entire archive for saving."""
        cells = []
        for cell in self.grid.values():
            if cell.occupied:
                cells.append(cell.to_dict())
        return {
            'cells': cells,
            'stats': self.stats(),
        }
