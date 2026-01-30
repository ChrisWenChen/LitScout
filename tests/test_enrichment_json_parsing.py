import json
import unittest

class TestJSONParsing(unittest.TestCase):
    def test_valid_json(self):
        with open('tests/fixtures/valid_response.json') as f:
            data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_malformed_json(self):
        with self.assertRaises(json.JSONDecodeError):
            json.loads("{")

    def test_edge_case(self):
        with open('tests/fixtures/edge_case_papers.jsonl') as f:
            for line in f:
                data = json.loads(line)
                self.assertIsInstance(data, dict)

    def test_empty_json(self):
        self.assertEqual(json.loads('{}'), {})

    def test_non_json_input(self):
        with self.assertRaises(json.JSONDecodeError):
            json.loads('not a json')

if __name__ == '__main__':
    unittest.main()
