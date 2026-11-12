import re
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_sm")

def tokenizer():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")

def extract_text(txt_path):
    return Path(txt_path).read_text(encoding="utf-8") # Return entire text as str

def sentence_chunking(text, tokenizer, size=100, overlap=0.5, offset=0, doc=None):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("size must be positive and overlap must be between 0 and 1")
    
    # Check if doc exists, otherwise load it
    doc = nlp(text) if doc is None else doc
    
    # Obtain sentences if they exist
    sentences = [s for s in doc.sents if s.text.strip()]
    if not sentences:
        return []

    # Create prefix sums
    prefix = [0]
    for sentence in sentences:
        prefix.append(prefix[-1] + len(tokenizer.encode(sentence.text)))

    # Use count() to quickly count tokens
    def count(start, end):
        return prefix[end] - prefix[start]

    # Initialize the costs and window lists
    best_by_end = [float("inf")] * (len(sentences) + 1)
    window_by_end = [None] * (len(sentences) + 1)
    costs = {}
    previous = {}

    # Going through sentences
    for start in range(len(sentences)):
        # Initalize best transition and previous best
        best_transition = float("inf")
        best_previous = None
        
        # Now testing possible end options
        for end in range(start + 1, len(sentences) + 1):
            
            # First window only costs how close it gets to size
            if start == 0:
                cost = (count(start, end) - size) ** 2
                previous[start, end] = None
                
            else:
                # As the end moves right, one more previous end is allowed.
                old_end = end - 1
                
                # DP to save window cost by checking best window at old_end
                if window_by_end[old_end] is not None:
                    shared = count(start, old_end)
                    transition = best_by_end[old_end] + (shared - size * overlap) ** 2
                    # If smallest, then you save it
                    if transition < best_transition:
                        best_transition = transition
                        best_previous = window_by_end[old_end]
                        
                if best_previous is None:
                    continue
                    
                cost = best_transition + (count(start, end) - size) ** 2
                previous[start, end] = best_previous
            costs[start, end] = cost

        # Compare all of the ends and update the best one if cheaper
        for end in range(start + 1, len(sentences) + 1):
            if (start, end) in costs and costs[start, end] < best_by_end[end]:
                best_by_end[end] = costs[start, end]
                window_by_end[end] = (start, end)

    # Goes through the windows_by_end and previous (linked list) to create one list of windows
    state = window_by_end[len(sentences)]
    windows = []
    while state is not None:
        windows.append(state)
        state = previous[state]
    windows.reverse()

    # Actually creating the chunks
    chunks = []
    for start, end in windows:
        first, last = sentences[start].start_char, sentences[end - 1].end_char
        chunks.append({"text": text[first:last],
                       "text_position": (offset + first, offset + last)})
    return chunks

def paragraph_chunking(text, tokenizer, size=100, allow_below_size=True):
    if size <= 0:
        raise ValueError("size must be positive")

    # Converts text into (start_position, end_position, token_count) for each original paragraph
    paragraphs = []
    for paragraph in re.finditer(r"[^\t]+?(?=\n\s*\n|\t|\Z)", text):
        raw = paragraph.group()
        
        if not raw.strip():
            continue
        
        start = paragraph.start() + (len(raw) - len(raw.lstrip()))
        end = paragraph.end() - (len(raw) - len(raw.rstrip()))
        paragraphs.append((start, end, len(tokenizer.encode(text[start:end]))))

    # best[end] has the lowest cost for grouping everything before end
    best = [0] + [float("inf")] * len(paragraphs)
    previous = [0] * (len(paragraphs) + 1)
    
    for end in range(1, len(paragraphs) + 1):
        total = 0
        
        # Moves backwards to test different starts
        for start in range(end - 1, -1, -1):
            total += paragraphs[start][2]
            
            # Skip small groups unless allowed or the whole text is short.
            if not allow_below_size and total < size and not (start == 0 and end == len(paragraphs)):
                continue
            
            # Determine penality and punish if below much stronger
            difference = total - size
            if difference < 0:
                penalty = 2 * difference ** 2
            else:
                penalty = difference ** 2
                
            # Calculate cost and remember if it's cheaper
            cost = best[start] + penalty
            if cost < best[end]:
                best[end] = cost
                previous[end] = start

    # Goes through the previous starts (linked list) to create one list of groups
    groups = []
    end = len(paragraphs)
    while end > 0:
        start = previous[end]
        groups.append((paragraphs[start][0], paragraphs[end - 1][1]))
        end = start
    groups.reverse()
    return groups

def paragraph_based_chunks(text, tokenizer, size=100, overlap=0.5):
    # Call paragraph_chunking
    groups = paragraph_chunking(text, tokenizer, size)
    
    # Do sentence_chunking
    chunks = []
    texts = [text[start:end] for start, end in groups]
    for (group_start, _), group, doc in zip(groups, texts, nlp.pipe(texts)):
        chunks.extend(sentence_chunking(group, tokenizer, size, overlap, group_start, doc))

    return chunks


def sentence_window_chunks(text, tokenizer, size=100, overlap=0.5):
    """Make overlapping sentence windows across paragraph boundaries."""
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("size must be positive and overlap must be between 0 and 1")
    sentences = [s for s in nlp(text).sents if s.text.strip()]
    # Prefix totals let us count any range without encoding it again.
    totals = [0]
    for sentence in sentences:
        totals.append(totals[-1] + len(tokenizer.encode(sentence.text)))

    chunks = []
    start = 0
    while start < len(sentences):
        # Add sentences until crossing the target, then choose the closer end.
        end = start + 1
        while end < len(sentences) and totals[end] - totals[start] < size:
            end += 1
        if end > start + 1 and abs(totals[end - 1] - totals[start] - size) <= abs(totals[end] - totals[start] - size):
            end -= 1
        first, last = sentences[start].start_char, sentences[end - 1].end_char
        chunks.append({'text': text[first:last], 'text_position': (first, last)})
        if end == len(sentences):
            break
        # Move roughly (1 - overlap) of the chunk, at least one sentence.
        advance = (totals[end] - totals[start]) * (1 - overlap)
        next_start = min(range(start + 1, end + 1),
                         key=lambda i: abs(totals[i] - totals[start] - advance))
        start = next_start
    return chunks


def chunk_text(text, tokenizer, size=100, overlap=0.5):
    return sentence_window_chunks(text, tokenizer, size, overlap)
