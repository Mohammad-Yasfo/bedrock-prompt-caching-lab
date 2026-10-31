"""Render a Markdown results section from a results folder.

Every number is read from metadata.json, requests.jsonl and summary.json.
Nothing is typed by hand, and the output is plain ASCII Markdown.
"""
import argparse
import json
from pathlib import Path

START = '<!-- live-run:start -->'
END = '<!-- live-run:end -->'
CHECKS = ('schema_ok', 'decision_ok', 'number_ok', 'citations_ok')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def usd(value):
    return f'${value:.4f}'


def rate(value):
    text = f'{value:.4f}'.rstrip('0')
    whole, _, fraction = text.partition('.')
    return f'${whole}.{fraction.ljust(2, "0")}'


def ascii_only(text):
    return text.encode('ascii', 'backslashreplace').decode('ascii')


def md_table(header, align, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def checks_text(checks):
    if checks.get('passed'):
        return 'passed'
    failed = [name.replace('_ok', '') for name in CHECKS if not checks.get(name)]
    return 'failed: ' + (', '.join(failed) if failed else 'stop reason')


def answer_block(folder, row):
    text = (folder / row['answer_file']).read_text(encoding='utf-8')
    try:
        body = json.dumps(json.loads(text), indent=2, ensure_ascii=True)
        return 're-indented for reading', f'```json\n{body}\n```'
    except ValueError:
        return 'verbatim, with any non-ASCII characters escaped', f'```text\n{ascii_only(text)}\n```'


def render(folder, case_id):
    folder = Path(folder)
    metadata = load(folder / 'metadata.json')
    summary = load(folder / 'summary.json')
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    completed = [row for row in rows if row.get('status') == 'ok']
    failed = [row for row in rows if row.get('status') != 'ok']
    prices = metadata['price_config']
    rates = prices['usd_per_million_tokens']
    arms = [arm for arm in ('unmarked', 'explicit') if arm in summary['arms']]
    arms += [arm for arm in sorted(summary['arms']) if arm not in arms]
    parts = []

    parts.append(
        f"This section was produced by `tools/render_run.py` from the raw files of one run. "
        f"The run started on {metadata['started_at_utc'][:10]} (UTC) and sent {summary['completed_requests']} completed "
        f"requests to `{metadata['model']}` from `{metadata['region']}` using boto3 {metadata['boto3_version']}. "
        f"Prices were taken from a local price file verified on {prices.get('verified_on')} against the "
        f"[Bedrock pricing page]({prices.get('source_url')}).")

    arm_count = '2 arms' if metadata['mode'] == 'paired' else f"1 arm ({metadata['mode']})"
    settings = [
        ('Experiment and reference revision', f"`{metadata['experiment']}`, revision `{metadata['revision']}`"),
        ('Cases, arms and repeats', f"{metadata['limit']} case{'s' if metadata['limit'] != 1 else ''}, {arm_count}, "
                                    f"{metadata['repeats']} repeat{'s' if metadata['repeats'] != 1 else ''}: {metadata['planned_requests']} planned calls"),
        ('Completed and failed calls', f"{summary['completed_requests']} completed, {summary['failed_requests']} failed"),
        ('Generation settings', f"max {metadata['max_tokens']} output tokens, temperature {metadata['temperature']}"),
        ('Prefix SHA-256 (first 12 characters)', f"`{metadata['prefix_sha256'][:12]}`"),
        ('Estimated upper token cost with 10% allowance', usd(metadata['estimated_upper_token_cost_with_allowance_usd'])),
        ('Rates per million tokens', f"input {rate(rates['input'])}, five-minute write {rate(rates['cache_write_5m'])}, "
                                     f"cache read {rate(rates['cache_read'])}, output {rate(rates['output'])}"),
    ]
    parts.append(md_table(('Setting', 'Value'), ('---', '---'), settings))

    parts.append('Every request, in the order sent. Token columns are the counters returned by Converse.')
    request_rows = []
    for index, row in enumerate(rows, 1):
        if row.get('status') != 'ok':
            request_rows.append((index, row.get('mode', ''), row.get('case_id', ''), '', 'error: ' + str(row.get('error_code', 'unknown')),
                                 '', '', '', '', '', '', 'no usage returned'))
            continue
        usage = row['usage']
        request_rows.append((index, row['mode'], row['case_id'], row['repeat'], row['cache_state'], usage['inputTokens'],
                             usage['cacheReadInputTokens'], usage['cacheWriteInputTokens'], usage['outputTokens'],
                             usd(row['cost_usd']), f"{row['latency_seconds']:.2f}", checks_text(row['checks'])))
    parts.append(md_table(('Call', 'Arm', 'Case', 'Repeat', 'Cache state', 'Input', 'Cache read', 'Cache write', 'Output', 'Cost', 'Seconds', 'Checks'),
                          ('---:', '---', '---', '---:', '---', '---:', '---:', '---:', '---:', '---:', '---:', '---'), request_rows))

    parts.append('Per-arm totals from `summary.json`:')
    metrics = [
        ('Completed requests', lambda arm: arm['requests']),
        ('Passed automated checks', lambda arm: f"{arm['passed']} of {arm['requests']}"),
        ('Requests with cache reads', lambda arm: arm['cache_read_requests']),
        ('Share of input tokens read from cache', lambda arm: f"{100 * arm['cache_read_fraction_of_input']:.1f}%"),
        ('Token cost', lambda arm: usd(arm['cost_usd'])),
        ('Cost per passing case', lambda arm: usd(arm['cost_per_passing_case_usd']) if arm['cost_per_passing_case_usd'] is not None else 'no passing case'),
        ('Priced uncached equivalent', lambda arm: usd(arm['priced_uncached_equivalent_usd'])),
        ('Duration p50 (seconds)', lambda arm: f"{arm['latency_p50_seconds']:.2f}"),
        ('Duration p95 (seconds)', lambda arm: f"{arm['latency_p95_seconds']:.2f}"),
    ]
    parts.append(md_table(('Metric',) + tuple(arms), ('---',) + ('---:',) * len(arms),
                          [(name,) + tuple(value(summary['arms'][arm]) for arm in arms) for name, value in metrics]))

    if {'unmarked', 'explicit'} <= set(summary['arms']):
        unmarked, explicit = summary['arms']['unmarked'], summary['arms']['explicit']
        if unmarked['cost_usd'] > 0:
            change = 100 * (1 - explicit['cost_usd'] / unmarked['cost_usd'])
            direction = 'lower' if change >= 0 else 'higher'
            parts.append(f"Over the same cases, the explicit arm's token cost was {abs(change):.1f}% {direction} than the unmarked arm's. "
                         f"Automated checks passed on {explicit['passed']} of {explicit['requests']} explicit requests and "
                         f"{unmarked['passed']} of {unmarked['requests']} unmarked requests.")

    if failed:
        codes = sorted({str(row.get('error_code', 'unknown')) for row in failed})
        parts.append(f"{len(failed)} request(s) failed with error code(s) {', '.join(codes)}. Any charges for them are not in the totals; limit 4 applies.")

    parts.append(f'### Returned answers for `{case_id}`')
    for arm in arms:
        match = next((row for row in completed if row['case_id'] == case_id and row['mode'] == arm), None)
        if match is None:
            parts.append(f'No completed `{arm}` request for `{case_id}` is in this run.')
            continue
        note, block = answer_block(folder, match)
        parts.append(f"The first `{arm}` request for this case (repeat {match['repeat']}) reported cache state `{match['cache_state']}`, "
                     f"cost {usd(match['cost_usd'])}, duration {match['latency_seconds']:.2f} seconds, stop reason `{match['stop_reason']}` "
                     f"and automated checks {checks_text(match['checks'])}. The saved answer, {note}:")
        parts.append(block)

    parts.append('Read these numbers with the four limits in mind. Cache reads in the unmarked arm are implicit reuse, not a flaw in the run, '
                 'and the priced uncached equivalent is arithmetic on the same observed tokens rather than a separate measurement.')
    return ascii_only('\n\n'.join(parts))


def insert(article, section):
    article = Path(article)
    text = article.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f'{article} must contain exactly one {START} and one {END}')
    head, rest = text.split(START)
    _, tail = rest.split(END)
    article.write_text(head + START + '\n' + section.rstrip('\n') + '\n' + END + tail, encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', help='Results folder written by lab.benchmark run')
    parser.add_argument('--case', default='decline-01', help='Case whose saved answers are shown in full')
    parser.add_argument('--insert', metavar='ARTICLE', help='Replace the text between <!-- live-run:start --> and <!-- live-run:end --> in this Markdown file')
    parser.add_argument('--out', help='Write the section to this file (default: <folder>/run-section.md)')
    args = parser.parse_args()
    section = render(args.folder, args.case)
    out = Path(args.out) if args.out else Path(args.folder) / 'run-section.md'
    out.write_text(section + '\n', encoding='utf-8', newline='\n')
    if args.insert:
        insert(args.insert, section)
        print(f'Inserted the run section into {args.insert}')
    print(section)


if __name__ == '__main__':
    main()
