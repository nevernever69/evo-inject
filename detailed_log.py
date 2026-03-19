"""
Detailed Local Logger — saves everything needed for paper analysis.

Produces two files:
  logs/detailed_<timestamp>.jsonl  — one JSON line per event
  logs/summary_<timestamp>.json   — periodic full snapshot

Events logged:
  - Every successful attack: payload, response, tokens, program, findings, app, loss, refinement
  - Every generation summary: fitness stats, archive state, phrase library state, top programs
  - Every new archive cell fill: full details of the attack that filled it
  - Every promoted phrase: the token sequence, decoded text, what it was promoted from
  - Refinement details: before/after loss, which tokens changed, improvement per step

This gives us everything to:
  1. Build paper figures (fitness curves, archive heatmaps, attack examples)
  2. Analyze what evolution actually discovered vs what was seeded
  3. Show concrete attack examples with full context
  4. Compare seeded vs zero-seed experiments
"""

import json
import os
import time


class DetailedLogger:
    """Logs detailed per-event data locally for post-hoc analysis."""

    def __init__(self, log_dir='logs', prefix='detailed'):
        os.makedirs(log_dir, exist_ok=True)
        ts = int(time.time())
        self.event_path = os.path.join(log_dir, f'{prefix}_{ts}.jsonl')
        self.summary_path = os.path.join(log_dir, f'summary_{ts}.json')
        self.log_dir = log_dir
        self._event_count = 0
        self._start_time = time.time()

        # Running tallies for quick local reads
        self._all_attacks = []       # every successful attack
        self._archive_fills = []     # every new archive cell
        self._promoted_phrases = []  # every promoted phrase
        self._gen_summaries = []     # per-generation snapshots

    def _write_event(self, event_type, data):
        """Append one JSON line to the event log."""
        record = {
            'event': event_type,
            'time': time.time() - self._start_time,
            'wall_clock': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seq': self._event_count,
        }
        record.update(data)
        self._event_count += 1

        with open(self.event_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def log_attack(self, gen, organism_id, app_name, payload_text,
                   payload_tokens, response_text, findings, reward,
                   loss, baseline_loss, coherence_info, program_str,
                   components_summary, refinement_stats, phrase_indices,
                   attack_type, structure_class):
        """Log every successful attack with full details."""
        attack = {
            'gen': gen,
            'organism_id': organism_id,
            'app_name': app_name,
            'payload_text': payload_text[:500],
            'payload_tokens': payload_tokens[:60],
            'payload_len': len(payload_tokens),
            'response_text': response_text[:1000],
            'response_len': len(response_text),
            'findings': [
                {
                    'type': f.get('type', '?'),
                    'severity': f.get('severity', '?'),
                    'detail': f.get('detail', '')[:200],
                }
                for f in findings
            ],
            'finding_types': [f.get('type', '?') for f in findings],
            'reward': reward,
            'loss': loss,
            'baseline_loss': baseline_loss,
            'loss_reduction': (baseline_loss - loss) if loss and baseline_loss else 0,
            'coherence': coherence_info,
            'program': program_str[:200],
            'components': components_summary,
            'refinement': refinement_stats,
            'phrase_indices': phrase_indices,
            'attack_type': attack_type,
            'structure_class': structure_class,
        }

        self._all_attacks.append(attack)
        self._write_event('attack', attack)

    def log_archive_fill(self, gen, attack_type, app_name, structure_class,
                         fitness, payload_text, payload_tokens,
                         response_text, program_str, is_new, findings):
        """Log every archive insertion (new cell or update)."""
        fill = {
            'gen': gen,
            'attack_type': attack_type,
            'app_name': app_name,
            'structure_class': structure_class,
            'fitness': fitness,
            'is_new_cell': is_new,
            'payload_text': payload_text[:500],
            'payload_tokens': payload_tokens[:60],
            'response_text': response_text[:1000],
            'program': program_str[:200],
            'finding_types': [f.get('type', '?') for f in findings],
        }

        self._archive_fills.append(fill)
        self._write_event('archive_fill', fill)

    def log_promoted_phrase(self, gen, text, tokens, category, source):
        """Log when a new phrase is promoted from token block discovery."""
        promo = {
            'gen': gen,
            'text': text[:200],
            'tokens': tokens[:20],
            'category': category,
            'source': source,
        }

        self._promoted_phrases.append(promo)
        self._write_event('phrase_promoted', promo)

    def log_refinement(self, gen, organism_id, app_name, block_idx,
                       loss_before, loss_after, tokens_before, tokens_after,
                       improvements, steps):
        """Log token block refinement details."""
        self._write_event('refinement', {
            'gen': gen,
            'organism_id': organism_id,
            'app_name': app_name,
            'block_idx': block_idx,
            'loss_before': loss_before,
            'loss_after': loss_after,
            'improvement': loss_before - loss_after if loss_before and loss_after else 0,
            'tokens_before': tokens_before[:20],
            'tokens_after': tokens_after[:20],
            'tokens_changed': sum(
                1 for a, b in zip(tokens_before, tokens_after) if a != b
            ),
            'improvements': improvements,
            'steps': steps,
        })

    def log_generation(self, gen, metrics, population, archive,
                       phrase_library, gen_time, compute_loss, do_refine):
        """Log full generation snapshot."""
        best = population.best()
        archive_stats = archive.stats()
        phrase_stats = phrase_library.stats()
        genome_stats = population.genome_stats()

        # Top 5 programs across population by success rate
        top_programs = []
        for org in sorted(population.organisms, key=lambda o: o.fitness, reverse=True)[:5]:
            for i, prog in enumerate(org.library.programs):
                stats = org.library.stats()
                if stats['program_uses'][i] > 0:
                    top_programs.append({
                        'organism_fitness': org.fitness,
                        'program_idx': i,
                        'program': str(prog)[:200],
                        'uses': stats['program_uses'][i],
                        'successes': stats['program_successes'][i],
                        'reward': stats['program_rewards'][i],
                        'length': prog.length(),
                        'category': prog.dominant_category(phrase_library),
                        'structure': prog.structure_class(),
                    })
        top_programs.sort(key=lambda x: x['reward'], reverse=True)
        top_programs = top_programs[:10]

        # Top phrases by success rate
        top_phrases = []
        for idx, p in enumerate(phrase_library.phrases):
            if p['uses'] > 0:
                top_phrases.append({
                    'idx': idx,
                    'text': p['text'][:100],
                    'category': p['category'],
                    'uses': p['uses'],
                    'successes': p['successes'],
                    'rate': p['successes'] / p['uses'],
                    'is_seed': p['is_seed'],
                })
        top_phrases.sort(key=lambda x: x['rate'], reverse=True)
        top_phrases = top_phrases[:15]

        # Archive coverage by dimension
        archive_by_type = archive_stats.get('by_attack_type', {})
        archive_by_app = archive_stats.get('by_app', {})

        summary = {
            'gen': gen,
            'gen_time': gen_time,
            'total_runtime': time.time() - self._start_time,
            'fitness': {
                'best': best.fitness,
                'avg': population.avg_fitness(),
                'diversity': population.diversity(),
            },
            'metrics': {
                'total_requests': metrics.total_requests,
                'total_successes': metrics.total_successes,
                'success_rate': metrics.total_successes / max(1, metrics.total_requests),
                'total_discoveries': metrics.total_discoveries,
            },
            'best_organism': {
                'summary': best.summary(),
                'avg_loss': best.memory.avg_loss(),
                'avg_refine': best.memory.avg_refinement_improvement(),
                'findings_count': len(best.memory.findings),
                'apps_targeted': len(best.memory.apps_tried),
            },
            'archive': {
                'coverage': archive_stats['coverage'],
                'occupied': archive_stats['occupied'],
                'total_cells': archive_stats['total_cells'],
                'total_fills': archive_stats['total_fills'],
                'total_updates': archive_stats['total_updates'],
                'by_attack_type': archive_by_type,
                'by_app': archive_by_app,
            },
            'phrases': {
                'total': phrase_stats['total_phrases'],
                'seed': phrase_stats['seed_phrases'],
                'promoted': phrase_stats['promoted_phrases'],
                'total_uses': phrase_stats['total_uses'],
                'total_successes': phrase_stats['total_successes'],
            },
            'genome': {
                'avg_program_length': genome_stats['program_length']['mean'],
                'total_phrases_in_gp': genome_stats.get('total_phrases', 0),
                'total_token_blocks': genome_stats.get('total_token_blocks', 0),
                'vocab_tokens': genome_stats.get('_vocab_tokens', 0),
            },
            'top_programs': top_programs,
            'top_phrases': top_phrases,
            'compute_loss': compute_loss,
            'do_refine': do_refine,
            'attacks_this_session': len(self._all_attacks),
            'archive_fills_this_session': len(self._archive_fills),
            'promoted_this_session': len(self._promoted_phrases),
        }

        self._gen_summaries.append(summary)
        self._write_event('generation', summary)

        # Also write/overwrite the running summary file every generation
        self._save_summary()

    def _save_summary(self):
        """Save current summary snapshot (overwritten each gen)."""
        snapshot = {
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_runtime': time.time() - self._start_time,
            'total_events': self._event_count,
            'total_attacks': len(self._all_attacks),
            'total_archive_fills': len(self._archive_fills),
            'total_promoted_phrases': len(self._promoted_phrases),
            'generations_logged': len(self._gen_summaries),
            'event_log': self.event_path,

            # Last generation summary
            'latest_gen': self._gen_summaries[-1] if self._gen_summaries else None,

            # All archive fills (these are the key results)
            'archive_fills': self._archive_fills,

            # All promoted phrases (novel discoveries)
            'promoted_phrases': self._promoted_phrases,

            # Top 20 attacks by reward
            'top_attacks': sorted(
                self._all_attacks, key=lambda x: x['reward'], reverse=True
            )[:20],

            # Attacks with system_prompt_leak (the strongest evidence)
            'system_prompt_leaks': [
                a for a in self._all_attacks
                if 'system_prompt_leak' in a.get('finding_types', [])
            ][:20],

            # Attacks with instruction_followed
            'instruction_followed': [
                a for a in self._all_attacks
                if 'instruction_followed' in a.get('finding_types', [])
            ][:20],
        }

        with open(self.summary_path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

    def final_save(self):
        """Save final comprehensive summary."""
        self._save_summary()
        print(f"  Detailed event log: {self.event_path}")
        print(f"  Summary snapshot:   {self.summary_path}")
        print(f"  Total events:       {self._event_count}")
        print(f"  Total attacks:      {len(self._all_attacks)}")
        print(f"  Archive fills:      {len(self._archive_fills)}")
        print(f"  Promoted phrases:   {len(self._promoted_phrases)}")
