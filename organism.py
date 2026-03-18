"""
Organism - The evolving prompt injection fuzzer entity

The genome is a LIBRARY of GP programs (immune system model).
Each program constructs payloads / mutation strategies from raw ASCII.
During a lifetime, programs that cause injection success get cloned+mutated
(clonal selection). Programs that do nothing get replaced.

Fragment library: successful substrings get promoted to reusable fragments.
The GP can reference these fragments via PUSH_FRAGMENT instructions.
This creates a self-organizing vocabulary that the system discovers.

Each organism has:
  - Library: collection of GP programs (the antibody repertoire)
  - Fragments: promoted successful substrings (discovered vocabulary)
  - Brain: Q-network that learns which app to target
  - Memory: findings, rewards, what worked where
  - Context: runtime state for reactive payloads
  - Fitness: based on injection success
"""

import numpy as np
import random
from gp import Program, crossover as gp_crossover, mutate as gp_mutate
from brain import Brain
from config import (
    STATE_SIZE, ACTION_SIZE, STEPS_PER_LIFETIME,
    GP_MAX_PROGRAM_LENGTH, GP_MUTATION_RATE,
    LIBRARY_SIZE, LIBRARY_CLONE_MUTATE_RATE,
    LIBRARY_REPLACE_THRESHOLD, LIBRARY_SELECTION_TEMP,
    LIBRARY_INHERIT_RATIO,
    MAX_FRAGMENTS, FRAGMENT_PROMOTE_THRESHOLD,
    FRAGMENT_MIN_LENGTH, FRAGMENT_MAX_LENGTH,
    TARGET_APPS,
)


class FragmentLibrary:
    """
    Self-organizing vocabulary of discovered text fragments.

    When a payload succeeds, its substrings are tracked. Substrings
    that appear in multiple successful payloads get promoted to
    reusable fragments. GP programs reference these via PUSH_FRAGMENT.

    This is how the system discovers its own "language" for the target.
    Against DVWA it would discover SQL fragments. Against an LLM it
    discovers prompt structure patterns.
    """

    def __init__(self, fragments=None):
        # slot_id -> fragment text
        self.fragments = fragments or {}
        # substring -> success count (candidates for promotion)
        self._candidates = {}
        self._next_slot = 0

    def record_success(self, payload):
        """Record a successful payload — extract candidate fragments."""
        if not payload or len(payload) < FRAGMENT_MIN_LENGTH:
            return

        # Extract substrings of various lengths
        for length in range(FRAGMENT_MIN_LENGTH,
                           min(FRAGMENT_MAX_LENGTH, len(payload)) + 1,
                           max(1, len(payload) // 10)):
            for start in range(0, len(payload) - length + 1,
                               max(1, length // 2)):
                substr = payload[start:start + length]
                if len(substr.strip()) >= FRAGMENT_MIN_LENGTH:
                    self._candidates[substr] = (
                        self._candidates.get(substr, 0) + 1
                    )

        # Promote candidates that hit threshold
        self._promote_candidates()

    def _promote_candidates(self):
        """Promote frequently successful substrings to fragment slots."""
        for text, count in list(self._candidates.items()):
            if count >= FRAGMENT_PROMOTE_THRESHOLD:
                if len(self.fragments) < MAX_FRAGMENTS:
                    self.fragments[self._next_slot] = text
                    self._next_slot += 1
                else:
                    # Replace shortest/oldest fragment
                    weakest = min(self.fragments.keys(),
                                  key=lambda k: len(self.fragments[k]))
                    if len(text) > len(self.fragments[weakest]):
                        self.fragments[weakest] = text

                del self._candidates[text]

        # Prune low-count candidates to prevent memory bloat
        if len(self._candidates) > 200:
            sorted_cands = sorted(
                self._candidates.items(), key=lambda x: x[1], reverse=True
            )
            self._candidates = dict(sorted_cands[:100])

    def get(self, slot):
        """Get fragment at slot, or empty string."""
        return self.fragments.get(slot, '')

    def as_dict(self):
        """Return fragments as dict for GP context."""
        return dict(self.fragments)

    def copy(self):
        """Deep copy."""
        return FragmentLibrary(fragments=dict(self.fragments))

    def stats(self):
        return {
            'num_fragments': len(self.fragments),
            'num_candidates': len(self._candidates),
            'fragment_lengths': [len(v) for v in self.fragments.values()],
        }


class ProgramLibrary:
    """
    Immune system model: a library of GP programs.
    Unchanged from DVWA version — same clonal selection logic.
    """

    def __init__(self, programs=None, size=LIBRARY_SIZE):
        self.size = size
        if programs is not None:
            self.programs = programs
        else:
            self.programs = [
                Program(max_length=GP_MAX_PROGRAM_LENGTH)
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

    def _maybe_replace_weakest(self):
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
                self.programs[i] = Program(max_length=GP_MAX_PROGRAM_LENGTH)
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
            'avg_program_length': sum(p.length() for p in self.programs) / max(1, len(self.programs)),
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
        self.successful_payloads = []
        self.successful_programs = []
        self.timeline = []

    def record(self, app_name, app_idx, reward, found_something):
        self.total_steps += 1
        self.total_reward += reward
        self.apps_tried.add(app_name)
        self.timeline.append((self.total_steps, app_idx, reward))

        self.actions_tried[app_idx] = self.actions_tried.get(app_idx, 0) + 1
        if found_something:
            self.apps_with_findings.add(app_name)
            self.actions_succeeded[app_idx] = self.actions_succeeded.get(app_idx, 0) + 1

    def record_finding(self, finding, payload, program=None):
        self.findings.append(finding)
        self.successful_payloads.append(payload)
        if program is not None:
            self.successful_programs.append(program.copy())

    def success_rate(self):
        if self.total_steps == 0:
            return 0.0
        return sum(self.actions_succeeded.values()) / self.total_steps

    def unique_apps_found(self):
        return len(self.apps_with_findings)

    # Compat aliases
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


class Organism:
    """
    A GP-based prompt injection fuzzer organism.

    The library is a collection of GP programs (immune system).
    Fragment library stores discovered reusable text patterns.
    Brain selects which target app to attack.

    Lifecycle:
      1. Born with library from parents + brain + fragments
      2. Each step: brain picks app → library picks program →
         generate payload → (optional: mutator refines) → send to target →
         reward → learn
      3. Within lifetime: clonal selection + fragment promotion
      4. Fitness = injection success metrics
    """

    _id_counter = 0

    def __init__(self, library=None, genome=None, brain=None,
                 fragments=None, generation=0):
        Organism._id_counter += 1
        self.id = Organism._id_counter
        self.generation = generation

        # Library of GP programs (immune system)
        if library is not None:
            self.library = library
        elif genome is not None:
            self.library = ProgramLibrary(
                programs=[genome] + [Program(max_length=GP_MAX_PROGRAM_LENGTH)
                                     for _ in range(LIBRARY_SIZE - 1)],
                size=LIBRARY_SIZE,
            )
        else:
            self.library = ProgramLibrary(size=LIBRARY_SIZE)

        # Fragment library (discovered vocabulary)
        self.fragment_library = fragments or FragmentLibrary()

        # Compat
        self.genome = self.library.programs[0]

        self.brain = brain or Brain()
        self.memory = Memory()
        self.fitness = 0.0

        # Runtime context for reactive payloads
        self._context = {
            'prev_payload': None,
            'prev_response': None,
            'app_name': None,
            'fragments': self.fragment_library.as_dict(),
            'system_hint': None,
        }

    def generate_payload(self, clean_input='test', app_name=None):
        """
        Select a program from the library and execute it with context.
        Returns (payload_string, program_index, strategies).
        """
        if app_name:
            self._context['app_name'] = app_name
        self._context['fragments'] = self.fragment_library.as_dict()

        program_idx = self.library.select_program()
        program = self.library.get_active_program()

        payload = program.execute(
            clean_input=clean_input,
            context=self._context,
        )
        strategies = program.extract_strategies()

        return payload, program_idx, strategies

    def update_context(self, payload, response_text, found_something):
        """Update runtime context for reactive payloads."""
        if found_something and payload:
            self._context['prev_payload'] = payload
            # Promote successful payload fragments
            self.fragment_library.record_success(payload)
        if response_text is not None:
            self._context['prev_response'] = response_text[:500]

    def record_system_hint(self, hint_text):
        """Store leaked system prompt fragment for reactive use."""
        if hint_text:
            self._context['system_hint'] = hint_text[:200]

    def observe_state(self, apps, target_idx, last_result=None):
        """
        Build state vector for the brain.
        24 features: app info + context + performance + embedding hints.
        """
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
            # Embedding distance from last result (if available)
            state[9] = last_result.get('embedding_distance', 0.0)

        # Organism context (10-15)
        state[10] = self.library.length() / GP_MAX_PROGRAM_LENGTH
        state[11] = self.memory.success_rate()
        state[12] = self.memory.total_steps / max(STEPS_PER_LIFETIME, 1)
        state[13] = self.memory.unique_apps_found() / max(len(apps), 1)
        state[14] = len(self.fragment_library.fragments) / MAX_FRAGMENTS
        # Strategy diversity: how many unique strategies has this organism used?
        strategy_count = sum(1 for p in self.library.programs
                             for i in p.instructions if i.op == 'PUSH_STRATEGY')
        state[15] = min(strategy_count, 20) / 20.0

        # Per-app history (16-19)
        app_tried = self.memory.actions_tried.get(target_idx, 0)
        app_success = self.memory.actions_succeeded.get(target_idx, 0)
        state[16] = min(app_tried, 20) / 20.0
        state[17] = app_success / max(app_tried, 1)
        state[18] = 1.0 if app.name in self.memory.apps_with_findings else 0.0
        # Best reward on this app
        app_rewards = [r for s, idx, r in self.memory.timeline if idx == target_idx]
        state[19] = max(app_rewards) / 200.0 if app_rewards else 0.0

        # Library health (20-23)
        lib_stats = self.library.stats()
        programs_active = sum(1 for u in lib_stats['program_uses'] if u > 0)
        state[20] = programs_active / max(len(self.library.programs), 1)
        programs_successful = sum(1 for s in lib_stats['program_successes'] if s > 0)
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
        """
        Fitness for prompt injection fuzzer:
        - Injection success is the core signal
        - App coverage (testing multiple targets)
        - Strategy diversity
        - Fragment discovery
        """
        mem = self.memory

        reward_score = mem.total_reward

        finding_types = set()
        finding_apps = set()
        for f in mem.findings:
            finding_types.add(f.get('type', 'unknown'))
            finding_apps.add(f.get('app_name', ''))
        finding_diversity = len(finding_types) * 50

        # Escalating coverage bonus per app
        n_apps = mem.unique_apps_found()
        coverage = n_apps * (n_apps + 1) * 30

        finding_count = mem.finding_count() * 10
        trend = max(0, mem.reward_trend()) * 30

        # Program efficiency
        prog_len = max(1, self.library.length())
        efficiency_bonus = (mem.finding_count() / prog_len) * 20 if mem.finding_count() > 0 else 0

        # Library diversity
        lib_stats = self.library.stats()
        programs_that_found = sum(1 for s in lib_stats['program_successes'] if s > 0)
        library_bonus = programs_that_found * 15

        # Fragment bonus: reward discovered vocabulary
        fragment_bonus = len(self.fragment_library.fragments) * 10

        # Parsimony pressure
        bloat_penalty = max(0, prog_len - 40) * 0.5

        # System prompt leak bonus (critical finding)
        critical_bonus = 0
        for f in mem.findings:
            if f.get('type') == 'system_prompt_leak':
                critical_bonus += 200
            elif f.get('type') == 'instruction_followed':
                critical_bonus += 150

        self.fitness = (
            reward_score
            + finding_diversity
            + coverage
            + finding_count
            + trend
            + efficiency_bonus
            + library_bonus
            + fragment_bonus
            + critical_bonus
            - bloat_penalty
        )

        self.genome = self.library.best_program() or self.library.programs[0]
        return self.fitness

    def make_offspring(self, other, child_generation):
        """Create child via library crossover + mutation."""
        child_library = self._library_crossover(other)

        if self.fitness >= other.fitness:
            child_brain = self.brain.copy()
            child_fragments = self.fragment_library.copy()
        else:
            child_brain = other.brain.copy()
            child_fragments = other.fragment_library.copy()

        child_brain.epsilon = max(child_brain.epsilon, 0.2)

        return Organism(
            library=child_library,
            brain=child_brain,
            fragments=child_fragments,
            generation=child_generation,
        )

    def _library_crossover(self, other):
        """Library-level crossover."""
        def _rate(lib, i):
            if lib.program_uses[i] == 0:
                return 0.0
            return lib.program_rewards[i] / lib.program_uses[i]

        p1_ranked = sorted(range(len(self.library.programs)),
                           key=lambda i: _rate(self.library, i), reverse=True)
        p2_ranked = sorted(range(len(other.library.programs)),
                           key=lambda i: _rate(other.library, i), reverse=True)

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
                child_programs.append(gp_mutate(src.copy(), GP_MUTATION_RATE))
            else:
                child_programs.append(Program(max_length=GP_MAX_PROGRAM_LENGTH))

        for i in range(len(child_programs)):
            if random.random() < GP_MUTATION_RATE:
                child_programs[i] = gp_mutate(child_programs[i], GP_MUTATION_RATE)

        return ProgramLibrary(programs=child_programs[:LIBRARY_SIZE], size=LIBRARY_SIZE)

    def summary(self):
        lib_stats = self.library.stats()
        frag_stats = self.fragment_library.stats()
        return (
            f"Org#{self.id} gen={self.generation} "
            f"fit={self.fitness:.1f} "
            f"reward={self.memory.total_reward:.1f} "
            f"findings={self.memory.finding_count()} "
            f"apps={self.memory.unique_apps_found()} "
            f"lib={lib_stats['library_size']}progs "
            f"frags={frag_stats['num_fragments']} "
            f"clonal={lib_stats['clonal_events']} "
            f"succ={self.memory.success_rate():.1%}"
        )

    def __repr__(self):
        return (
            f"Organism(id={self.id}, gen={self.generation}, "
            f"fitness={self.fitness:.1f}, lib={len(self.library.programs)}progs, "
            f"frags={len(self.fragment_library.fragments)})"
        )
