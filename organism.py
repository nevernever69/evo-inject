"""
Organism — Compositional Attack Entity

Each organism carries:
  - Library: collection of GP programs (structure evolution)
  - Brain: Q-network that learns which app to target
  - Memory: findings, rewards, what worked where
  - Fitness: based on injection success + loss reduction + archive contribution

The GP programs compose phrases + token blocks.
Token blocks are refined by loss-guided hill climbing (refinement.py).
The phrase library is SHARED across the population (phrases.py).
"""

import numpy as np
import random
from gp import (
    Program, crossover as gp_crossover, mutate as gp_mutate,
    program_complexity,
)
from brain import Brain
from config import (
    STATE_SIZE, ACTION_SIZE, STEPS_PER_LIFETIME,
    GP_MAX_PROGRAM_LENGTH, GP_MUTATION_RATE,
    LIBRARY_SIZE, LIBRARY_CLONE_MUTATE_RATE,
    LIBRARY_REPLACE_THRESHOLD, LIBRARY_SELECTION_TEMP,
    LIBRARY_INHERIT_RATIO,
    TARGET_APPS, MAX_PAYLOAD_TOKENS,
)


class ProgramLibrary:
    """
    Immune system model: a library of GP programs.
    Programs that find vulnerabilities get cloned, failures get replaced.
    """

    def __init__(self, programs=None, size=LIBRARY_SIZE, phrase_library=None):
        self.size = size
        if programs is not None:
            self.programs = programs
        else:
            self.programs = [
                Program(max_length=GP_MAX_PROGRAM_LENGTH,
                        phrase_library=phrase_library)
                for _ in range(size)
            ]

        self._reset_stats()
        self.clonal_events = 0
        self.replacements = 0
        self.active_index = 0

    def _reset_stats(self):
        n = len(self.programs)
        self.program_rewards = [0.0] * n
        self.program_uses = [0] * n
        self.program_successes = [0] * n
        self.steps_since_reward = [0] * n

    def select_program(self):
        scores = []
        for i in range(len(self.programs)):
            if self.program_uses[i] == 0:
                scores.append(1.0)
            else:
                rate = self.program_rewards[i] / self.program_uses[i]
                scores.append(rate)

        scores = np.array(scores, dtype=np.float64)
        scores = scores / max(LIBRARY_SELECTION_TEMP, 0.01)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / (exp_scores.sum() + 1e-10)

        self.active_index = np.random.choice(len(self.programs), p=probs)
        return self.active_index

    def get_active_program(self):
        return self.programs[self.active_index]

    def record_result(self, program_idx, reward, found_something):
        self.program_rewards[program_idx] += reward
        self.program_uses[program_idx] += 1

        if found_something:
            self.program_successes[program_idx] += 1
            self.steps_since_reward[program_idx] = 0
            self._clonal_select(program_idx)
        else:
            self.steps_since_reward[program_idx] += 1
            self._maybe_replace_weakest()

    def _clonal_select(self, program_idx):
        if len(self.programs) < 2:
            return
        weakest_idx = self._find_weakest(exclude=program_idx)
        if weakest_idx is None:
            return

        clone = self.programs[program_idx].copy()
        clone = gp_mutate(clone, LIBRARY_CLONE_MUTATE_RATE)

        self.programs[weakest_idx] = clone
        self.program_rewards[weakest_idx] = 0.0
        self.program_uses[weakest_idx] = 0
        self.program_successes[weakest_idx] = 0
        self.steps_since_reward[weakest_idx] = 0
        self.clonal_events += 1

    def _maybe_replace_weakest(self, phrase_library=None):
        total_steps = sum(self.program_uses)
        for i in range(len(self.programs)):
            should_replace = False
            if (self.program_uses[i] >= LIBRARY_REPLACE_THRESHOLD and
                self.program_successes[i] == 0 and
                self.steps_since_reward[i] >= LIBRARY_REPLACE_THRESHOLD):
                should_replace = True
            if total_steps >= LIBRARY_REPLACE_THRESHOLD * 2 and self.program_uses[i] == 0:
                should_replace = True

            if should_replace:
                self.programs[i] = Program(
                    max_length=GP_MAX_PROGRAM_LENGTH,
                    phrase_library=phrase_library,
                )
                self.program_rewards[i] = 0.0
                self.program_uses[i] = 0
                self.program_successes[i] = 0
                self.steps_since_reward[i] = 0
                self.replacements += 1
                break

    def _find_weakest(self, exclude=None):
        worst_idx = None
        worst_score = float('inf')
        for i in range(len(self.programs)):
            if i == exclude:
                continue
            if self.program_uses[i] == 0:
                return i
            score = self.program_rewards[i] / self.program_uses[i]
            if score < worst_score:
                worst_score = score
                worst_idx = i
        return worst_idx

    def best_program(self):
        if not self.programs:
            return None
        best_idx = max(range(len(self.programs)),
                       key=lambda i: self.program_rewards[i])
        return self.programs[best_idx]

    def length(self):
        best = self.best_program()
        return best.length() if best else 0

    def copy(self):
        new_lib = ProgramLibrary(
            programs=[p.copy() for p in self.programs],
            size=self.size,
        )
        return new_lib

    def stats(self):
        return {
            'library_size': len(self.programs),
            'clonal_events': self.clonal_events,
            'replacements': self.replacements,
            'program_rewards': list(self.program_rewards),
            'program_uses': list(self.program_uses),
            'program_successes': list(self.program_successes),
            'avg_program_length': sum(
                p.length() for p in self.programs
            ) / max(1, len(self.programs)),
        }


class Memory:
    """What the organism discovered during its lifetime."""

    def __init__(self):
        self.apps_tried = set()
        self.apps_with_findings = set()
        self.actions_tried = {}
        self.actions_succeeded = {}
        self.total_reward = 0.0
        self.total_steps = 0
        self.findings = []
        self.successful_payloads = []    # list of flat token lists
        self.successful_programs = []
        self.successful_components = []  # component lists from successful attacks
        self.timeline = []
        self.loss_history = []
        self.refinement_history = []     # stats from refinement steps

    def record(self, app_name, app_idx, reward, found_something):
        self.total_steps += 1
        self.total_reward += reward
        self.apps_tried.add(app_name)
        self.timeline.append((self.total_steps, app_idx, reward))

        self.actions_tried[app_idx] = self.actions_tried.get(app_idx, 0) + 1
        if found_something:
            self.apps_with_findings.add(app_name)
            self.actions_succeeded[app_idx] = (
                self.actions_succeeded.get(app_idx, 0) + 1
            )

    def record_finding(self, finding, flat_tokens, components=None, program=None):
        self.findings.append(finding)
        self.successful_payloads.append(list(flat_tokens))
        if components is not None:
            self.successful_components.append(components)
        if program is not None:
            self.successful_programs.append(program.copy())

    def record_loss(self, loss):
        if loss != float('inf'):
            self.loss_history.append(loss)

    def record_refinement(self, stats):
        self.refinement_history.append(stats)

    def success_rate(self):
        if self.total_steps == 0:
            return 0.0
        return sum(self.actions_succeeded.values()) / self.total_steps

    def unique_apps_found(self):
        return len(self.apps_with_findings)

    def unique_endpoints_found(self):
        return self.unique_apps_found()

    def unique_servers(self):
        return self.unique_apps_found()

    def method_diversity(self):
        return len(self.actions_succeeded)

    def finding_count(self):
        return len(self.findings)

    def reward_trend(self):
        if len(self.timeline) < 4:
            return 0.0
        mid = len(self.timeline) // 2
        first_half = np.mean([r for _, _, r in self.timeline[:mid]])
        second_half = np.mean([r for _, _, r in self.timeline[mid:]])
        return second_half - first_half

    def avg_loss(self):
        if not self.loss_history:
            return float('inf')
        return np.mean(self.loss_history)

    def avg_refinement_improvement(self):
        if not self.refinement_history:
            return 0.0
        reductions = [s.get('loss_reduction', 0) for s in self.refinement_history]
        return np.mean(reductions)


class Organism:
    """
    A compositional attack entity.

    Lifecycle:
      1. Born with library from parents + brain
      2. Each step: brain picks app → library picks program →
         program composes structure → token blocks refined →
         inject into model → reward → learn
      3. Within lifetime: clonal selection adapts library
      4. Successful attacks fed to MAP-Elites archive
    """

    _id_counter = 0

    def __init__(self, library=None, brain=None,
                 generation=0, phrase_library=None):
        Organism._id_counter += 1
        self.id = Organism._id_counter
        self.generation = generation

        if library is not None:
            self.library = library
        else:
            self.library = ProgramLibrary(
                size=LIBRARY_SIZE,
                phrase_library=phrase_library,
            )

        self.brain = brain or Brain()
        self.memory = Memory()
        self.fitness = 0.0

        # For backward compatibility
        self.genome = (
            self.library.programs[0] if self.library.programs else None
        )

    def generate_payload(self, phrase_library=None, target_instruction_tokens=None,
                         separator_tokens=None):
        """
        Select a program from the library and execute it.

        Returns:
            (components, program_idx)
            components: list of (type, metadata, tokens)
            program_idx: which program was used
        """
        program_idx = self.library.select_program()
        program = self.library.get_active_program()

        components = program.execute(
            phrase_library=phrase_library,
            target_instruction_tokens=target_instruction_tokens,
            separator_tokens=separator_tokens,
        )

        return components, program_idx

    def observe_state(self, apps, target_idx, last_result=None):
        """Build state vector for the brain."""
        state = np.zeros(STATE_SIZE)
        app = apps[target_idx]
        app_features = app.as_features()

        # App features (0-4)
        state[0] = app_features['app_index']
        state[1] = app_features['system_prompt_length']
        state[2] = app_features['has_template']
        state[3] = app_features['baseline_response_length']
        state[4] = app_features['baseline_response_std']

        # Last result context (5-9)
        if last_result:
            state[5] = min(last_result.get('size', 0), 1000) / 1000.0
            state[6] = min(last_result.get('time', 0), 10.0) / 10.0
            state[7] = min(last_result.get('tokens', 0), 200) / 200.0
            state[8] = 1.0 if last_result.get('had_findings', False) else 0.0
            state[9] = last_result.get('embedding_distance', 0.0)

        # Organism context (10-15)
        state[10] = self.library.length() / GP_MAX_PROGRAM_LENGTH
        state[11] = self.memory.success_rate()
        state[12] = self.memory.total_steps / max(STEPS_PER_LIFETIME, 1)
        state[13] = self.memory.unique_apps_found() / max(len(apps), 1)
        # Refinement effectiveness
        avg_refine = self.memory.avg_refinement_improvement()
        state[14] = min(avg_refine, 5.0) / 5.0
        # Average loss trend
        avg_loss = self.memory.avg_loss()
        state[15] = min(avg_loss, 15.0) / 15.0 if avg_loss != float('inf') else 1.0

        # Per-app history (16-19)
        app_tried = self.memory.actions_tried.get(target_idx, 0)
        app_success = self.memory.actions_succeeded.get(target_idx, 0)
        state[16] = min(app_tried, 20) / 20.0
        state[17] = app_success / max(app_tried, 1)
        state[18] = 1.0 if app.name in self.memory.apps_with_findings else 0.0
        app_rewards = [r for s, idx, r in self.memory.timeline if idx == target_idx]
        state[19] = max(app_rewards) / 200.0 if app_rewards else 0.0

        # Library health (20-23)
        lib_stats = self.library.stats()
        programs_active = sum(1 for u in lib_stats['program_uses'] if u > 0)
        state[20] = programs_active / max(len(self.library.programs), 1)
        programs_successful = sum(
            1 for s in lib_stats['program_successes'] if s > 0
        )
        state[21] = programs_successful / max(len(self.library.programs), 1)
        state[22] = self.library.clonal_events / max(self.memory.total_steps, 1)
        state[23] = self.memory.reward_trend() / 100.0

        return state

    def choose_endpoint(self, state):
        return self.brain.choose_action(state)

    def learn_from_experience(self, state, action, reward, next_state, done):
        self.brain.remember(state, action, reward, next_state, done)
        self.brain.learn()

    def compute_fitness(self):
        """Fitness for compositional attack organism."""
        mem = self.memory

        reward_score = mem.total_reward

        finding_types = set()
        for f in mem.findings:
            finding_types.add(f.get('type', 'unknown'))
        finding_diversity = len(finding_types) * 50

        n_apps = mem.unique_apps_found()
        coverage = n_apps * (n_apps + 1) * 30

        finding_count = mem.finding_count() * 10
        trend = max(0, mem.reward_trend()) * 30

        prog_len = max(1, self.library.length())
        efficiency_bonus = (
            (mem.finding_count() / prog_len) * 20
            if mem.finding_count() > 0 else 0
        )

        lib_stats = self.library.stats()
        programs_that_found = sum(
            1 for s in lib_stats['program_successes'] if s > 0
        )
        library_bonus = programs_that_found * 15

        bloat_penalty = max(0, prog_len - 25) * 0.5

        # Loss-based bonus
        loss_bonus = 0
        avg_loss = mem.avg_loss()
        if avg_loss != float('inf') and avg_loss < 8.0:
            loss_bonus = (8.0 - avg_loss) * 25

        # Refinement bonus — organisms that benefit from refinement
        refine_bonus = 0
        avg_refine = mem.avg_refinement_improvement()
        if avg_refine > 0:
            refine_bonus = min(avg_refine * 20, 100)

        critical_bonus = 0
        for f in mem.findings:
            if f.get('type') == 'system_prompt_leak':
                critical_bonus += 200
            elif f.get('type') == 'instruction_followed':
                critical_bonus += 150
            elif f.get('type') == 'low_target_loss':
                critical_bonus += 250
            elif f.get('type') == 'loss_reduction':
                critical_bonus += 100

        self.fitness = (
            reward_score
            + finding_diversity
            + coverage
            + finding_count
            + trend
            + efficiency_bonus
            + library_bonus
            + loss_bonus
            + refine_bonus
            + critical_bonus
            - bloat_penalty
        )

        self.genome = self.library.best_program() or self.library.programs[0]
        return self.fitness

    def make_offspring(self, other, child_generation, phrase_library=None):
        """Create child via library crossover + mutation."""
        child_library = self._library_crossover(other, phrase_library)

        if self.fitness >= other.fitness:
            child_brain = self.brain.copy()
        else:
            child_brain = other.brain.copy()

        child_brain.epsilon = max(child_brain.epsilon, 0.2)

        return Organism(
            library=child_library,
            brain=child_brain,
            generation=child_generation,
            phrase_library=phrase_library,
        )

    def _library_crossover(self, other, phrase_library=None):
        """Library-level crossover."""
        def _rate(lib, i):
            if lib.program_uses[i] == 0:
                return 0.0
            return lib.program_rewards[i] / lib.program_uses[i]

        p1_ranked = sorted(
            range(len(self.library.programs)),
            key=lambda i: _rate(self.library, i), reverse=True,
        )
        p2_ranked = sorted(
            range(len(other.library.programs)),
            key=lambda i: _rate(other.library, i), reverse=True,
        )

        child_programs = []
        n_inherit = int(LIBRARY_SIZE * LIBRARY_INHERIT_RATIO)
        n_from_each = max(1, n_inherit // 2)

        for i in p1_ranked[:n_from_each]:
            child_programs.append(self.library.programs[i].copy())
        for i in p2_ranked[:n_from_each]:
            child_programs.append(other.library.programs[i].copy())

        while len(child_programs) < n_inherit:
            p1_prog = random.choice(self.library.programs)
            p2_prog = random.choice(other.library.programs)
            child_programs.append(gp_crossover(p1_prog, p2_prog))

        while len(child_programs) < LIBRARY_SIZE:
            if random.random() < 0.5 and child_programs:
                src = random.choice(child_programs)
                child_programs.append(
                    gp_mutate(src.copy(), GP_MUTATION_RATE,
                              phrase_library=phrase_library)
                )
            else:
                child_programs.append(
                    Program(max_length=GP_MAX_PROGRAM_LENGTH,
                            phrase_library=phrase_library)
                )

        for i in range(len(child_programs)):
            if random.random() < GP_MUTATION_RATE:
                child_programs[i] = gp_mutate(
                    child_programs[i], GP_MUTATION_RATE,
                    phrase_library=phrase_library,
                )

        return ProgramLibrary(
            programs=child_programs[:LIBRARY_SIZE], size=LIBRARY_SIZE,
        )

    def summary(self):
        lib_stats = self.library.stats()
        avg_loss = self.memory.avg_loss()
        loss_str = f"{avg_loss:.2f}" if avg_loss != float('inf') else "inf"
        avg_refine = self.memory.avg_refinement_improvement()
        return (
            f"Org#{self.id} gen={self.generation} "
            f"fit={self.fitness:.1f} "
            f"reward={self.memory.total_reward:.1f} "
            f"findings={self.memory.finding_count()} "
            f"apps={self.memory.unique_apps_found()} "
            f"loss={loss_str} "
            f"refine={avg_refine:.2f} "
            f"lib={lib_stats['library_size']}progs "
            f"clonal={lib_stats['clonal_events']} "
            f"succ={self.memory.success_rate():.1%}"
        )

    def __repr__(self):
        return (
            f"Organism(id={self.id}, gen={self.generation}, "
            f"fitness={self.fitness:.1f}, lib={len(self.library.programs)}progs)"
        )
