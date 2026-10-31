"""Rebuild the public synthetic dataset. Expected answers never enter prompts."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cases = []
for variant in range(10):
    attempts = 1000 + 100 * variant
    previous = int(attempts * .90)
    for kind in ('decline', 'normal', 'missing', 'silent', 'small_sample'):
        current = {'id': 'E1', 'type': 'current', 'provider': f'provider-{variant + 1:02d}',
                   'window': 'matched-hour', 'attempts': attempts, 'successes': int(attempts * .40)}
        baseline = {'id': 'E2', 'type': 'baseline', 'provider': current['provider'],
                    'window': 'matched-hour', 'attempts': attempts, 'successes': previous}
        evidence = [current, baseline]
        decision = 'investigate'
        if kind == 'normal':
            current['successes'] = int(attempts * .88)
            decision = 'normal'
        elif kind == 'missing':
            decision = 'insufficient_evidence'
            if variant % 2:
                del current['successes']
            else:
                evidence = [current]
        elif kind == 'silent':
            current.update(attempts=0, successes=0)
        elif kind == 'small_sample':
            current.update(attempts=50 + variant, successes=20 + variant)
            decision = 'insufficient_evidence'
        rate = (round(current['successes'] / current['attempts'], 4)
                if current['attempts'] and 'successes' in current else None)
        cases.append({'id': f'{kind}-{variant + 1:02d}',
            'question': 'Assess this provider using the reference rules. Return the required JSON record.',
            'evidence': evidence,
            'expected': {'decision': decision, 'current_success_rate': rate,
                         'required_evidence_ids': [item['id'] for item in evidence]}})
(root / 'lab' / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')
print(f'Wrote {len(cases)} synthetic cases.')
