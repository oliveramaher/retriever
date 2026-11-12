import unittest
from nli import trim_passage, find_supported


class TrimmingTests(unittest.TestCase):
    def test_removes_irrelevant_edges_and_keeps_original_offsets(self):
        text = 'First sentence. The storm arrived. Henry stayed home. Last sentence.'
        def score(span, claim):
            return 0.9 if 'The storm arrived.' in span and 'Henry stayed home.' in span else 0.1
        result = trim_passage({'text': text, 'text_position': (40, 40 + len(text))}, 'claim', score)
        self.assertEqual(result['text'], 'The storm arrived. Henry stayed home.')
        self.assertEqual(result['text_position'], (56, 93))

    def test_unsupported_passage_stays_intact(self):
        text = 'Nothing relevant. More unrelated text.'
        result = trim_passage({'text': text, 'text_position': (5, 43)}, 'claim', lambda *_: 0.1)
        self.assertFalse(result['supported'])
        self.assertEqual(result['text'], text)
        self.assertEqual(result['text_position'], (5, 43))


    def test_checks_three_in_order_and_stops(self):
        passages = [{'text': t} for t in ['First.', 'Second.', 'Third.', 'Fourth.']]
        calls = []
        def score(text, claim):
            calls.append(text)
            return 0.9 if text == 'Second.' else 0.1
        result = find_supported(passages, 'claim', score)
        self.assertEqual(result['rank'], 2)
        self.assertEqual(calls, ['First.', 'Second.'])

    def test_does_not_check_fourth_passage(self):
        passages = [{'text': t} for t in ['First.', 'Second.', 'Third.', 'Fourth.']]
        calls = []
        def score(text, claim):
            calls.append(text)
            return 0.9 if text == 'Fourth.' else 0.1
        self.assertIsNone(find_supported(passages, 'claim', score))
        self.assertEqual(calls, ['First.', 'Second.', 'Third.'])


if __name__ == '__main__':
    unittest.main()
