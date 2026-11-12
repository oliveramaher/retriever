import unittest
import itertools

import spacy
import segmentation as booksearch


class WordTokenizer:
    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return text.split()


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_nlp = booksearch.nlp
        booksearch.nlp = spacy.blank('en')
        booksearch.nlp.add_pipe('sentencizer')

    @classmethod
    def tearDownClass(cls):
        booksearch.nlp = cls.original_nlp

    def test_chunks_use_tokenizer_and_preserve_source(self):
        text = 'Alice arrived. Bob smiled. Carol waved.\n\nDaisy laughed. Edward sat. Fiona danced.'
        tokenizer = WordTokenizer()
        chunks = booksearch.chunk_text(text, tokenizer, size=8)
        self.assertGreater(tokenizer.calls, 0)
        for chunk in chunks:
            self.assertEqual(chunk['text'], text[slice(*chunk['text_position'])])
        for name in ['Alice', 'Bob', 'Carol', 'Daisy', 'Edward', 'Fiona']:
            self.assertTrue(any(name in chunk['text'] for chunk in chunks))

    def test_sentence_windows_cross_paragraph_boundaries(self):
        text = 'Tom spoke. He joked.\n\nMyrtle replied. Tom finished. Another sentence.'
        chunks = booksearch.sentence_window_chunks(text, WordTokenizer(), size=6, overlap=0.5)
        self.assertTrue(any('He joked.' in c['text'] and 'Myrtle replied.' in c['text'] for c in chunks))
        for chunk in chunks:
            self.assertEqual(chunk['text'], text[slice(*chunk['text_position'])])

    def test_final_sentence_and_empty_text(self):
        text = 'One sentence. Another sentence. A final sentence.'
        chunks = booksearch.chunk_text(text, WordTokenizer(), size=4)
        self.assertIn('A final sentence.', chunks[-1]['text'])
        self.assertEqual(booksearch.chunk_text('', WordTokenizer()), [])

    def test_long_sentence_is_not_split(self):
        text = ' '.join(['Word'] * 150) + '.'
        chunks = booksearch.chunk_text(text, WordTokenizer())
        self.assertEqual([chunk['text'] for chunk in chunks], [text])

    def test_sentence_windows_cover_all_sentences(self):
        sentences = [f'Sentence{i} ' + 'word ' * 18 + 'end.' for i in range(10)]
        text = ' '.join(sentences)
        chunks = booksearch.sentence_chunking(text, WordTokenizer(), offset=7)
        self.assertEqual(len(chunks), 3)
        for sentence in sentences:
            self.assertTrue(any(sentence in chunk['text'] for chunk in chunks))
        for chunk in chunks:
            start, end = chunk['text_position']
            self.assertEqual(chunk['text'], text[start - 7:end - 7])
        self.assertTrue(all(a['text_position'][1] > b['text_position'][0]
                            for a, b in zip(chunks, chunks[1:])))

    def test_sixty_token_sentences_use_three_overlapping_chunks(self):
        sentences = [f'Sentence{i} ' + 'word ' * 58 + 'end.' for i in range(4)]
        text = ' '.join(sentences)
        chunks = booksearch.sentence_chunking(text, WordTokenizer())
        self.assertEqual([c['text'] for c in chunks],
                         [' '.join(sentences[i:i + 2]) for i in range(3)])

    def test_sentence_dp_matches_exhaustive_search(self):
        for lengths in itertools.product([2, 3, 6], repeat=4):
            sentences = [f'Sentence{i} ' + 'word ' * (length - 2) + 'end.'
                         for i, length in enumerate(lengths)]
            text = ' '.join(sentences)
            chunks = booksearch.sentence_chunking(text, WordTokenizer(), size=5)
            spans = [(text.index(sentence), text.index(sentence) + len(sentence))
                     for sentence in sentences]
            windows = [(next(i for i, span in enumerate(spans) if span[0] == c['text_position'][0]),
                        next(i + 1 for i, span in enumerate(spans) if span[1] == c['text_position'][1]))
                       for c in chunks]

            def score(windows):
                result = sum((sum(lengths[a:b]) - 5) ** 2 for a, b in windows)
                return result + sum((sum(lengths[c:b]) - 2.5) ** 2
                                    for (_, b), (c, _) in zip(windows, windows[1:]))

            def paths(start, end):
                if end == len(lengths):
                    yield [(start, end)]
                else:
                    for next_start in range(start + 1, end + 1):
                        for next_end in range(end + 1, len(lengths) + 1):
                            for tail in paths(next_start, next_end):
                                yield [(start, end)] + tail

            possibilities = (path for end in range(1, len(lengths) + 1) for path in paths(0, end))
            self.assertEqual(score(windows), min(map(score, possibilities)))

    def test_chunk_count_is_capped_at_sentence_count(self):
        text = ' '.join(['word'] * 300) + '.'
        chunks = booksearch.sentence_chunking(text, WordTokenizer())
        self.assertEqual([c['text'] for c in chunks], [text])

    def test_chunk_count_near_target(self):
        for words, expected in [(110, 1), (140, 2)]:
            sentences = [f'Sentence{i} ' + 'word ' * 8 + 'end.' for i in range(words // 10)]
            chunks = booksearch.sentence_chunking(' '.join(sentences), WordTokenizer())
            self.assertEqual(len(chunks), expected)

    def test_paragraph_dp_looks_ahead(self):
        paragraphs = [' '.join([f'p{i}'] * count) for i, count in enumerate([60, 60, 40])]
        text = '\n\n'.join(paragraphs)
        groups = booksearch.paragraph_chunking(text, WordTokenizer(), allow_below_size=True)
        self.assertEqual([text[a:b] for a, b in groups],
                         [paragraphs[0], paragraphs[1] + '\n\n' + paragraphs[2]])

    def test_paragraph_dp_matches_all_small_partitions(self):
        for lengths in itertools.product([20, 60, 100], repeat=4):
            text = '\n\n'.join(' '.join(['word'] * length) for length in lengths)
            groups = booksearch.paragraph_chunking(text, WordTokenizer(), allow_below_size=True)
            def penalty(n):
                return (2 if n < 100 else 1) * (n - 100) ** 2
            actual = sum(penalty(len(text[a:b].split())) for a, b in groups)
            costs = []
            for cuts in itertools.product([False, True], repeat=len(lengths) - 1):
                totals, current = [], 0
                for i, length in enumerate(lengths):
                    current += length
                    if i == len(cuts) or cuts[i]:
                        totals.append(current)
                        current = 0
                costs.append(sum(penalty(total) for total in totals))
            self.assertEqual(actual, min(costs))

    def test_allow_below_size_switch_and_short_document(self):
        paragraphs = [' '.join(['word'] * n) for n in (110, 80)]
        text = '\n\n'.join(paragraphs)
        groups = booksearch.paragraph_chunking(text, WordTokenizer(), allow_below_size=False)
        self.assertEqual([text[a:b] for a, b in groups], [text])
        groups = booksearch.paragraph_chunking(text, WordTokenizer())
        self.assertEqual([text[a:b] for a, b in groups], paragraphs)
        short = 'only a few words'
        self.assertEqual(booksearch.paragraph_chunking(short, WordTokenizer(), allow_below_size=False), [(0, len(short))])
        self.assertEqual(booksearch.paragraph_chunking('', WordTokenizer()), [])


if __name__ == '__main__':
    unittest.main()
