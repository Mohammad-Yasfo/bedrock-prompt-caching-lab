"""A small Converse benchmark. Offline commands never initialize an AWS client."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'us.anthropic.claude-sonnet-4-6'
COUNT_MODEL = 'anthropic.claude-sonnet-4-6'
REGION = 'us-east-1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_text(experiment, revision):
    # The same experiment and revision produce identical text in both arms.
    return (f'Experiment partition: {experiment}. Reference revision: {revision}. '
            'These labels are bookkeeping only and have no bearing on the answer.\n\n'
            + (ROOT / 'lab' / 'reference.txt').read_text(encoding='utf-8'))


def request_for(case, prefix, model, mode, max_tokens):
    system = [{'text': prefix}]
    if mode == 'explicit':
        system.append({'cachePoint': {'type': 'default'}})
    # Expected answers stay local. Only question and evidence are sent.
    prompt = {'question': case['question'], 'evidence': case['evidence']}
    return {'modelId': model, 'system': system,
            'messages': [{'role': 'user', 'content': [{'text': json.dumps(prompt, sort_keys=True)}]}],
            'inferenceConfig': {'maxTokens': max_tokens, 'temperature': 0}}


def token_usage(response):
    raw = response['usage']
    if 'inputTokens' not in raw or 'outputTokens' not in raw:
        raise ValueError('Missing required usage counters')
    values = {name: raw.get(name, 0) for name in
              ('inputTokens', 'outputTokens', 'cacheReadInputTokens', 'cacheWriteInputTokens')}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError('Invalid usage counters in model response')
    # This lab uses a single default five-minute checkpoint.
    for detail in raw.get('cacheDetails', []):
        if detail.get('cacheWriteInputTokens', 0) and detail.get('ttl') not in (None, '5m'):
            raise ValueError('Unexpected cache TTL. A different write price is required.')
    return values


def price(usage, rates):
    return (usage['inputTokens'] * rates['input']
            + usage['cacheReadInputTokens'] * rates['cache_read']
            + usage['cacheWriteInputTokens'] * rates['cache_write_5m']
            + usage['outputTokens'] * rates['output']) / 1_000_000


def uncached_equivalent(usage, rates):
    total_input = usage['inputTokens'] + usage['cacheReadInputTokens'] + usage['cacheWriteInputTokens']
    return (total_input * rates['input'] + usage['outputTokens'] * rates['output']) / 1_000_000


def cache_state(usage):
    read, write = usage['cacheReadInputTokens'], usage['cacheWriteInputTokens']
    return 'read_and_write' if read and write else 'read' if read else 'write' if write else 'none'


def evaluate(text, case, stop_reason='end_turn'):
    """Deterministic checks on a narrow schema, not a complete quality evaluation."""
    result = {key: False for key in ('schema_ok', 'decision_ok', 'number_ok', 'citations_ok', 'passed')}
    try:
        answer = json.loads(text)
    except (ValueError, TypeError):
        return result
    fields = {'decision', 'current_success_rate', 'evidence_ids', 'explanation'}
    if not isinstance(answer, dict) or set(answer) != fields:
        return result
    number = answer['current_success_rate']
    ids = answer['evidence_ids']
    if (answer['decision'] not in ('investigate', 'normal', 'insufficient_evidence')
        or not (number is None or (type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 1))
        or not isinstance(ids, list) or not all(isinstance(item, str) for item in ids)
        or not isinstance(answer['explanation'], str) or not answer['explanation'].strip()):
        return result
    expected = case['expected']
    result['schema_ok'] = True
    result['decision_ok'] = answer['decision'] == expected['decision']
    target = expected['current_success_rate']
    result['number_ok'] = number is None if target is None else (number is not None and abs(number - target) <= 0.00005)
    available = {item['id'] for item in case['evidence']}
    result['citations_ok'] = (len(ids) == len(set(ids)) and set(ids) <= available
                              and set(expected['required_evidence_ids']) <= set(ids))
    result['passed'] = all(result[key] for key in ('schema_ok', 'decision_ok', 'number_ok', 'citations_ok')) and stop_reason == 'end_turn'
    return result


def read_rates(path, live=False, model=None, region=None):
    config = load(path)
    rates = config['usd_per_million_tokens']
    if set(rates) != {'input', 'output', 'cache_read', 'cache_write_5m'}:
        raise ValueError('Prices require input, output, cache_read and cache_write_5m rates.')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError('All prices must be finite positive numbers.')
    if live:
        if config.get('verified') is not True or not config.get('verified_on') or not config.get('source_url'):
            raise ValueError('Copy example-prices.json to prices.local.json, check the actual prices, and set verified, verified_on and source_url.')
        if config.get('model_id') != model or config.get('region') != region:
            raise ValueError('The verified price configuration must match the requested model and source Region.')
        checked = datetime.strptime(config['verified_on'], '%Y-%m-%d').date()
        age = (datetime.now(timezone.utc).date() - checked).days
        if age < 0 or age > 7:
            raise ValueError('Use prices verified within the last seven days.')
    return config, rates


def economics(rates, prefix=10000, dynamic=2000, output=500, requests=10):
    plain = requests * (prefix * rates['input'] + dynamic * rates['input'] + output * rates['output']) / 1e6
    cached = (prefix * rates['cache_write_5m'] + (requests - 1) * prefix * rates['cache_read']
              + requests * (dynamic * rates['input'] + output * rates['output'])) / 1e6
    return {'requests': requests, 'uncached_usd': plain, 'one_write_then_hits_usd': cached,
            'saving_percent': 100 * (1 - cached / plain)}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def report(folder):
    folder = Path(folder)
    path = folder / 'requests.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    completed = [row for row in rows if row.get('status') == 'ok']
    summary = {'completed_requests': len(completed), 'failed_requests': len(rows) - len(completed), 'arms': {}}
    for mode in sorted({row['mode'] for row in completed}):
        group = [row for row in completed if row['mode'] == mode]
        good = sum(row['checks']['passed'] for row in group)
        cost = sum(row['cost_usd'] for row in group)
        read = sum(row['usage']['cacheReadInputTokens'] for row in group)
        inputs = sum(row['usage']['inputTokens'] + row['usage']['cacheReadInputTokens'] + row['usage']['cacheWriteInputTokens'] for row in group)
        summary['arms'][mode] = {'requests': len(group), 'passed': good, 'pass_rate': good / len(group),
            'cost_usd': cost, 'cost_per_passing_case_usd': cost / good if good else None,
            'priced_uncached_equivalent_usd': sum(row['uncached_equivalent_usd'] for row in group),
            'cache_read_fraction_of_input': read / inputs if inputs else 0,
            'cache_read_requests': sum(row['usage']['cacheReadInputTokens'] > 0 for row in group),
            'latency_p50_seconds': statistics.median(row['latency_seconds'] for row in group),
            'latency_p95_seconds': percentile([row['latency_seconds'] for row in group], .95)}
    summary['limitations'] = ['Unmarked requests may receive implicit cache hits.',
        'Uncached-equivalent cost is arithmetic using the same observed tokens, not an uncached latency measurement.',
        'Automated checks do not assess all claims in the explanation. Complete manual review.',
        'Failed calls may have incurred charges without returning token usage; listed costs exclude unknown charges.',
        'Latency measures total synchronous request duration, not time to first token.']
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    review_path = folder / ('review-template.csv' if (folder / 'review.csv').exists() else 'review.csv')
    with review_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'repeat', 'mode', 'answer_file', 'automated_pass', 'factual_correctness_0_to_2',
                         'evidence_support_0_to_2', 'useful_next_step_0_to_2', 'unsupported_claim', 'review_notes'])
        for row in completed:
            writer.writerow([row['case_id'], row['repeat'], row['mode'], row['answer_file'], row['checks']['passed'], '', '', '', '', ''])
    print(json.dumps(summary, indent=2))
    return summary


def run(args):
    if args.model != MODEL:
        raise ValueError('This lab targets ' + MODEL + '. Adapt model-specific controls and tests before using another model.')
    config, rates = read_rates(args.prices, live=True, model=args.model, region=args.region)
    cases = load(ROOT / 'lab' / 'cases.json')[:args.limit]
    prefix = stable_text(args.experiment, args.revision)
    modes = ['unmarked', 'explicit'] if args.mode == 'paired' else [args.mode]
    plan = []
    for repetition in range(args.repeats):
        # Alternate block order to reduce a fixed ordering effect. Shared implicit
        # cache state still prevents interpreting the unmarked arm as cache-off.
        for mode in (modes if repetition % 2 == 0 else list(reversed(modes))):
            for case in cases:
                plan.append((mode, repetition + 1, case))
    # Imports are deliberately below all offline command paths.
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name=args.profile, region_name=args.region).client(
        'bedrock-runtime', config=Config(retries={'total_max_attempts': 1, 'mode': 'standard'},
                                        connect_timeout=10, read_timeout=180))
    # Count each unique full input without the checkpoint for an upper estimate.
    counts = {}
    for case in cases:
        request = request_for(case, prefix, args.model, 'unmarked', args.max_tokens)
        result = client.count_tokens(modelId=COUNT_MODEL, input={'converse': {'system': request['system'], 'messages': request['messages']}})
        counts[case['id']] = result['inputTokens']
    # Count the cached prefix separately using a short, deliberately conservative
    # message allowance. The service usage response is the final cache check.
    prefix_result = client.count_tokens(modelId=COUNT_MODEL, input={'converse': {
        'system': [{'text': prefix}], 'messages': [{'role': 'user', 'content': [{'text': 'OK'}]}]}})
    if prefix_result['inputTokens'] < 1200:
        raise ValueError('Reference may be below the 1,024-token prefix minimum; expand meaningful reference content.')
    worst = sum(counts[case['id']] * max(rates['input'], rates['cache_write_5m']) + args.max_tokens * rates['output']
                for _, _, case in plan) / 1e6
    reserved = worst * 1.10
    if reserved > args.max_usd:
        raise ValueError(f'Estimated upper token cost with 10% allowance is ${reserved:.4f}, above --max-usd {args.max_usd:.2f}. Reduce cases/repeats or explicitly raise the limit.')
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'model': args.model, 'region': args.region,
        'experiment': args.experiment, 'revision': args.revision, 'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
        'cases_sha256': hashlib.sha256((ROOT / 'lab' / 'cases.json').read_bytes()).hexdigest(),
        'mode': args.mode, 'repeats': args.repeats, 'limit': args.limit, 'gap_seconds': args.gap_seconds,
        'max_tokens': args.max_tokens, 'temperature': 0, 'planned_requests': len(plan),
        'estimated_upper_token_cost_with_allowance_usd': reserved, 'price_config': config,
        'boto3_version': boto3.__version__, 'count_model': COUNT_MODEL, 'automatic_retries': False,
        'note': 'Token estimate only. Not an AWS billing cap. No paid repair calls are made.'}
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'{len(plan)} calls planned. Estimated upper token cost with allowance: ${reserved:.4f}', flush=True)
    with (folder / 'requests.jsonl').open('w', encoding='utf-8') as log:
        for index, (mode, repetition, case) in enumerate(plan):
            if index and args.gap_seconds:
                print(f'Waiting {args.gap_seconds} seconds before the next call.', flush=True)
                time.sleep(args.gap_seconds)
            started = time.perf_counter()
            try:
                response = client.converse(**request_for(case, prefix, args.model, mode, args.max_tokens))
                latency = time.perf_counter() - started
                usage = token_usage(response)
                answer = ''.join(block.get('text', '') for block in response['output']['message']['content'])
                filename = f'{index + 1:04d}-{mode}-{case["id"]}.txt'
                (folder / filename).write_text(answer, encoding='utf-8')
                row = {'status': 'ok', 'case_id': case['id'], 'repeat': repetition, 'mode': mode,
                    'latency_seconds': latency, 'usage': usage, 'cache_state': cache_state(usage),
                    'cost_usd': price(usage, rates), 'uncached_equivalent_usd': uncached_equivalent(usage, rates),
                    'checks': evaluate(answer, case, response.get('stopReason')), 'answer_file': filename,
                    'request_id': response.get('ResponseMetadata', {}).get('RequestId'), 'stop_reason': response.get('stopReason')}
            except Exception as error:
                code = getattr(error, 'response', {}).get('Error', {}).get('Code', type(error).__name__)
                log.write(json.dumps({'status': 'error', 'case_id': case['id'], 'mode': mode, 'error_code': code}) + '\n')
                log.flush()
                print('Stopped after request failure; unknown charges may exist. Error code: ' + code, flush=True)
                report(folder)
                raise
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f'{index + 1}/{len(plan)} {mode} {case["id"]}: {row["cache_state"]}, ${row["cost_usd"]:.6f}, checks={row["checks"]["passed"]}', flush=True)
    report(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    offline = sub.add_parser('offline', help='Show request structure and illustrative cost calculations; no AWS calls')
    offline.add_argument('--out', default='results/offline')
    live = sub.add_parser('run', help='Make paid Bedrock requests after validating prices and estimated token cost')
    live.add_argument('--model', default=MODEL)
    live.add_argument('--region', default=REGION)
    live.add_argument('--profile')
    live.add_argument('--prices', default='prices.local.json')
    live.add_argument('--experiment', required=True, help='Use a new ASCII ID per experiment and the same ID for paired arms')
    live.add_argument('--revision', default='1')
    live.add_argument('--mode', choices=['paired', 'unmarked', 'explicit'], default='paired')
    live.add_argument('--limit', type=int, default=5)
    live.add_argument('--repeats', type=int, default=2)
    live.add_argument('--max-tokens', type=int, default=600)
    live.add_argument('--max-usd', type=float, default=1.0)
    live.add_argument('--gap-seconds', type=float, default=0)
    live.add_argument('--out', required=True)
    summary = sub.add_parser('report', help='Rebuild aggregate metrics and a blank human review CSV')
    summary.add_argument('folder')
    args = parser.parse_args()
    if args.command == 'offline':
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        case = load(ROOT / 'lab' / 'cases.json')[0]
        request = request_for(case, stable_text('offline-example', '1'), MODEL, 'explicit', 600)
        (folder / 'request-preview.json').write_text(json.dumps(request, indent=2) + '\n', encoding='utf-8')
        _, rates = read_rates(ROOT / 'example-prices.json')
        result = {'kind': 'ILLUSTRATIVE ARITHMETIC, NOT BEDROCK MEASUREMENTS',
            'assumptions': {'prefix_tokens': 10000, 'dynamic_tokens': 2000, 'output_tokens': 500,
                            'one_write_then_all_hits': True, 'rates': rates},
            'rows': [economics(rates, requests=count) for count in (1, 2, 5, 10, 20)]}
        (folder / 'economics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        print('No AWS client was created. No latency or quality results were generated.')
    elif args.command == 'report':
        report(args.folder)
    else:
        if not 1 <= args.limit <= 50 or not 1 <= args.repeats <= 20:
            parser.error('Use 1 to 50 cases and 1 to 20 repeats.')
        if not 100 <= args.max_tokens <= 4096 or not math.isfinite(args.max_usd) or args.max_usd <= 0:
            parser.error('Use 100 to 4096 max tokens and a finite positive spending estimate limit.')
        if not math.isfinite(args.gap_seconds) or not 0 <= args.gap_seconds <= 7200:
            parser.error('Use a finite gap between 0 and 7200 seconds.')
        run(args)


if __name__ == '__main__':
    main()
