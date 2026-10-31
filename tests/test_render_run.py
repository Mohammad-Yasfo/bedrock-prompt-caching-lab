import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime, timezone
from lab import benchmark as b

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('render_run', ROOT / 'tools' / 'render_run.py')
render_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_run)


class RenderRunTests(unittest.TestCase):
    """The renderer is exercised on a folder written by the real runner with a mocked service."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = b.load(ROOT / 'example-prices.json')
        config.update(verified=True, verified_on=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        prices = self.root / 'prices.json'
        prices.write_text(json.dumps(config), encoding='utf-8')
        self.folder = self.root / 'run'
        args = SimpleNamespace(model=b.MODEL, region=b.REGION, profile=None, prices=prices,
            experiment='unit-test', revision='1', mode='paired', limit=1, repeats=1,
            max_tokens=600, max_usd=1, gap_seconds=0, out=str(self.folder))
        answer = {'decision': 'investigate', 'current_success_rate': .4,
                  'evidence_ids': ['E1', 'E2'], 'explanation': 'The approval rate fell. Check recent error codes.'}
        response = {'output': {'message': {'content': [{'text': json.dumps(answer)}]}},
                    'usage': {'inputTokens': 500, 'outputTokens': 100, 'cacheReadInputTokens': 2000, 'cacheWriteInputTokens': 0},
                    'stopReason': 'end_turn'}
        with patch('boto3.Session') as session:
            client = session.return_value.client.return_value
            client.count_tokens.return_value = {'inputTokens': 2500}
            client.converse.return_value = response
            b.run(args)

    def tearDown(self):
        self.temp.cleanup()

    def test_section_is_ascii_and_built_from_run_files(self):
        section = render_run.render(self.folder, 'decline-01')
        self.assertTrue(section.isascii())
        for expected in ('| unmarked |', '| explicit |', 'Cost per passing case', '"decision": "investigate"',
                         '`unit-test`', '2 completed, 0 failed', 'Returned answers for `decline-01`'):
            self.assertIn(expected, section)

    def test_missing_case_is_reported_not_invented(self):
        section = render_run.render(self.folder, 'silent-01')
        self.assertIn('No completed `unmarked` request for `silent-01`', section)
        self.assertNotIn('```json', section.split('### Returned answers')[1])

    def test_non_json_answer_is_escaped_not_dropped(self):
        row = json.loads((self.folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()[0])
        (self.folder / row['answer_file']).write_text('not JSON \u2014 with a dash', encoding='utf-8')
        section = render_run.render(self.folder, 'decline-01')
        self.assertTrue(section.isascii())
        self.assertIn('\\u2014', section)
        self.assertIn('verbatim', section)

    def test_insert_replaces_only_marked_region_and_is_idempotent(self):
        article = self.root / 'article.md'
        article.write_text('# Title\n\nbefore\n\n<!-- live-run:start -->\nold text\n<!-- live-run:end -->\n\nafter\n', encoding='utf-8')
        section = render_run.render(self.folder, 'decline-01')
        render_run.insert(article, section)
        first = article.read_text(encoding='utf-8')
        render_run.insert(article, section)
        self.assertEqual(first, article.read_text(encoding='utf-8'))
        self.assertTrue(first.startswith('# Title\n\nbefore\n\n<!-- live-run:start -->\n'))
        self.assertTrue(first.endswith('<!-- live-run:end -->\n\nafter\n'))
        self.assertNotIn('old text', first)
        self.assertIn(section, first)

    def test_insert_requires_exactly_one_marker_pair(self):
        article = self.root / 'article.md'
        article.write_text('no markers here\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            render_run.insert(article, 'section')


if __name__ == '__main__':
    unittest.main()
