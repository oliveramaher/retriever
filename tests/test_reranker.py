import unittest
from unittest.mock import Mock

import numpy as np
from rankings import combine_rankings
from reranker import rerank


class RerankerTests(unittest.TestCase):
    def test_rerank_keeps_original_passage_metadata(self):
        passages = [{'text': 'Same text', 'text_position': (0, 9)},
                    {'text': 'Same text', 'text_position': (20, 29)},
                    {'text': 'Other text', 'text_position': (40, 50)}]
        model = Mock()
        model.rerank.return_value = [{'index': 1}, {'index': 0}]
        result = rerank('question', passages, model, limit=2)
        self.assertEqual(result, [passages[1], passages[0]])
        model.rerank.assert_called_once_with('question', ['Same text', 'Same text', 'Other text'], top_n=2)

    def test_empty_candidates_do_not_call_model(self):
        model = Mock()
        self.assertEqual(rerank('question', [], model), [])
        model.rerank.assert_not_called()

    def test_candidate_pool_limit(self):
        scores = np.arange(30, dtype=float)
        self.assertEqual(len(combine_rankings(scores, scores)), 10)
        self.assertEqual(len(combine_rankings(scores, scores, limit=20)), 20)


if __name__ == '__main__':
    unittest.main()
