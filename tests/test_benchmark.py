import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from datetime import datetime, timezone
from lab import benchmark as b


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.rates = {'input': 3, 'cache_write_5m': 3.75, 'cache_read': .3, 'output': 15}

    def test_read_write_and_output_are_separate_billing_buckets(self):
        usage = {'inputTokens': 2000, 'cacheReadInputTokens': 5000, 'cacheWriteInputTokens': 1000, 'outputTokens': 500}
        self.assertAlmostEqual(b.price(usage, self.rates), .01875)
        self.assertAlmostEqual(b.uncached_equivalent(usage, self.rates), .0315)

    def test_first_write_can_cost_more(self):
        result = b.economics(self.rates, requests=1)
        self.assertGreater(result['one_write_then_hits_usd'], result['uncached_usd'])

    def test_ten_request_example(self):
        result = b.economics(self.rates, requests=10)
        self.assertAlmostEqual(result['uncached_usd'], .435)
        self.assertAlmostEqual(result['one_write_then_hits_usd'], .1995)
        self.assertAlmostEqual(result['saving_percent'], 54.13793103448276)

    def test_unexpected_ttl_is_not_silently_underpriced(self):
        with self.assertRaises(ValueError):
            b.token_usage({'usage': {'inputTokens': 2, 'outputTokens': 5, 'cacheDetails': [{'ttl': '1h', 'cacheWriteInputTokens': 1024}]}})

    def test_missing_counters_are_not_zero(self):
        with self.assertRaises(ValueError):
            b.token_usage({'usage': {'outputTokens': 5}})

    def test_cache_state_is_observed_not_assumed(self):
        self.assertEqual(b.cache_state({'cacheReadInputTokens': 0, 'cacheWriteInputTokens': 0}), 'none')
        self.assertEqual(b.cache_state({'cacheReadInputTokens': 1000, 'cacheWriteInputTokens': 20}), 'read_and_write')

    def test_example_rates_cannot_start_paid_run(self):
        with self.assertRaises(ValueError):
            b.read_rates(b.ROOT / 'example-prices.json', live=True, model=b.MODEL, region=b.REGION)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.case = b.load(b.ROOT / 'lab' / 'cases.json')[0]
        self.answer = {'decision': 'investigate', 'current_success_rate': .4,
                       'evidence_ids': ['E1', 'E2'], 'explanation': 'The rate fell below half the baseline. Inspect error codes.'}

    def test_reference_answer_passes(self):
        self.assertTrue(b.evaluate(json.dumps(self.answer), self.case)['passed'])

    def test_invented_citation_fails(self):
        self.answer['evidence_ids'].append('E99')
        self.assertFalse(b.evaluate(json.dumps(self.answer), self.case)['passed'])

    def test_missing_required_citation_fails(self):
        self.answer['evidence_ids'] = ['E1']
        self.assertFalse(b.evaluate(json.dumps(self.answer), self.case)['passed'])

    def test_wrong_number_fails(self):
        self.answer['current_success_rate'] = .9
        self.assertFalse(b.evaluate(json.dumps(self.answer), self.case)['passed'])

    def test_boolean_is_not_a_rate(self):
        self.answer['current_success_rate'] = False
        self.assertFalse(b.evaluate(json.dumps(self.answer), self.case)['schema_ok'])

    def test_invalid_json_and_truncation_fail(self):
        self.assertFalse(b.evaluate('not JSON', self.case)['passed'])
        self.assertFalse(b.evaluate(json.dumps(self.answer), self.case, 'max_tokens')['passed'])

    def test_all_synthetic_expected_answers_are_consistent(self):
        for case in b.load(b.ROOT / 'lab' / 'cases.json'):
            expected = case['expected']
            answer = {'decision': expected['decision'], 'current_success_rate': expected['current_success_rate'],
                      'evidence_ids': expected['required_evidence_ids'], 'explanation': 'Synthetic fixture only.'}
            self.assertTrue(b.evaluate(json.dumps(answer), case)['passed'], case['id'])


class RequestTests(unittest.TestCase):
    def test_only_cache_control_differs_between_arms(self):
        case = b.load(b.ROOT / 'lab' / 'cases.json')[0]
        prefix = b.stable_text('test', '1')
        unmarked = b.request_for(case, prefix, b.MODEL, 'unmarked', 600)
        explicit = b.request_for(case, prefix, b.MODEL, 'explicit', 600)
        self.assertEqual(explicit['system'].pop(), {'cachePoint': {'type': 'default'}})
        self.assertEqual(unmarked, explicit)
        sent = json.loads(unmarked['messages'][0]['content'][0]['text'])
        self.assertEqual(set(sent), {'question', 'evidence'})

    def test_request_matches_installed_sdk_shape(self):
        import boto3
        from botocore.validate import validate_parameters
        client = boto3.client('bedrock-runtime', region_name='us-east-1',
                             aws_access_key_id='offline-test', aws_secret_access_key='offline-test')
        case = b.load(b.ROOT / 'lab' / 'cases.json')[0]
        request = b.request_for(case, b.stable_text('test', '1'), b.MODEL, 'explicit', 600)
        validate_parameters(request, client.meta.service_model.operation_model('Converse').input_shape)
        validate_parameters({'modelId': b.COUNT_MODEL, 'input': {'converse': {'system': request['system'][:1], 'messages': request['messages']}}},
                            client.meta.service_model.operation_model('CountTokens').input_shape)

    def test_prefix_revision_changes_first_block(self):
        self.assertNotEqual(b.stable_text('test', '1'), b.stable_text('test', '2'))

    def test_rebuilding_report_preserves_manual_reviews(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'requests.jsonl').write_text('', encoding='utf-8')
            (root / 'review.csv').write_text('my completed review', encoding='utf-8')
            b.report(root)
            self.assertEqual((root / 'review.csv').read_text(), 'my completed review')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = b.load(b.ROOT / 'example-prices.json')
        config.update(verified=True, verified_on=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        self.prices = self.root / 'prices.json'
        self.prices.write_text(json.dumps(config), encoding='utf-8')
        self.args = SimpleNamespace(model=b.MODEL, region=b.REGION, profile=None, prices=self.prices,
            experiment='unit-test', revision='1', mode='paired', limit=1, repeats=1,
            max_tokens=600, max_usd=1, gap_seconds=0, out=str(self.root / 'run'))
        self.answer = {'decision': 'investigate', 'current_success_rate': .4,
            'evidence_ids': ['E1', 'E2'], 'explanation': 'The approval rate fell. Check recent error codes.'}
        self.response = {'output': {'message': {'content': [{'text': json.dumps(self.answer)}]}},
            'usage': {'inputTokens': 500, 'outputTokens': 100, 'cacheReadInputTokens': 2000,
                      'cacheWriteInputTokens': 0}, 'stopReason': 'end_turn'}

    def tearDown(self):
        self.temp.cleanup()

    @patch('boto3.Session')
    def test_live_flow_with_mocked_service_preserves_observed_hits(self, session):
        client = session.return_value.client.return_value
        client.count_tokens.return_value = {'inputTokens': 2500}
        client.converse.return_value = self.response
        b.run(self.args)
        result = b.load(self.root / 'run' / 'summary.json')
        self.assertEqual(result['completed_requests'], 2)
        self.assertEqual(result['arms']['unmarked']['cache_read_requests'], 1)
        self.assertEqual(result['arms']['explicit']['passed'], 1)
        for call in client.count_tokens.call_args_list:
            self.assertEqual(call.kwargs['modelId'], b.COUNT_MODEL)
        config = session.return_value.client.call_args.kwargs['config']
        self.assertEqual(config.retries['total_max_attempts'], 1)

    @patch('boto3.Session')
    def test_estimate_stops_inference_before_spending(self, session):
        client = session.return_value.client.return_value
        client.count_tokens.return_value = {'inputTokens': 2500}
        self.args.max_usd = .0001
        with self.assertRaisesRegex(ValueError, 'above --max-usd'):
            b.run(self.args)
        client.converse.assert_not_called()

    @patch('boto3.Session')
    def test_error_preserves_prior_usage_and_stops(self, session):
        client = session.return_value.client.return_value
        client.count_tokens.return_value = {'inputTokens': 2500}
        client.converse.side_effect = [self.response, RuntimeError('synthetic failure')]
        with self.assertRaises(RuntimeError):
            b.run(self.args)
        result = b.load(self.root / 'run' / 'summary.json')
        self.assertEqual(result['completed_requests'], 1)
        self.assertEqual(result['failed_requests'], 1)
        self.assertEqual(client.converse.call_count, 2)


if __name__ == '__main__':
    unittest.main()
