import math
import os
import re
import string
import sys
import time
from collections import defaultdict
from typing import Tuple, List, Dict, Any

import hyperscan
import multiprocess
import wordfreq

DB_FOLDER_NAME = "./dat/serialized_word_freq_in_chunks"

CHUNK_SIZE = 10000
MAX_FLOAT_VAL = float("+inf")
MIN_FLOAT_VAL = float("-inf")


def build_and_serialize_dfa_db(
    word_metadata: List[Tuple[str, int]],
    prefix_char: str,
    chunk_idx: int,
) -> None:
    start_time = time.time()

    db_folder_name_with_prefix_char = os.path.join(
        DB_FOLDER_NAME, str(ord(prefix_char))
    )
    if not os.path.exists(db_folder_name_with_prefix_char):
        os.mkdir(db_folder_name_with_prefix_char)

    expressions = [
        # Make sure each regex is a full match
        f"^{''.join([re.escape(c) + '+' for c in word])}$".encode()
        for word, _ in word_metadata
    ]
    flags = [
        hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_UTF8 | hyperscan.HS_FLAG_UCP
        for _ in range(len(expressions))
    ]
    db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
    db.compile(
        expressions=expressions,
        elements=len(expressions),
        flags=flags,
    )

    serialized_db = hyperscan.dumpb(db)
    with open(
        os.path.join(db_folder_name_with_prefix_char, f"{chunk_idx:02d}.db"),
        "wb",
    ) as f:
        f.write(serialized_db)
    with open(
        os.path.join(db_folder_name_with_prefix_char, f"{chunk_idx:02d}.txt"),
        "w",
    ) as f:
        for word, count in word_metadata:
            f.write(f"{word} {count}\n")

    end_time = time.time()
    print(
        f"Building db for {prefix_char} -- {chunk_idx:02d} with {len(word_metadata)} entries takes "
        f"{end_time - start_time:.2f} seconds"
    )


class MessageSegmentor:
    def __init__(self, overwrite_db=False) -> None:
        self.lang = "en"
        self.word_metadata_by_char: Dict[str, List[Tuple[str, float]]] = defaultdict(
            list
        )
        self.dbs_by_char: Dict[str, List[hyperscan.Database]] = defaultdict(list)
        self.raw_str: str = ""
        self.input_str: str = ""
        self.n: int = 0
        self.vis: List[bool] = []
        self.rcd: List[Tuple[float, int]] = []  # score, count, pos
        self.inf: int = sys.maxsize
        self.debug_mode: bool = False
        self.debug_logs: List[str] = []
        self.load_words_and_db(overwrite_db)

    def load_words_and_db(self, overwrite_db: bool):
        # Reference: https://github.com/rspeer/wordfreq
        frequency_dict = wordfreq.get_frequency_dict(self.lang)
        for word, frequency in frequency_dict.items():
            self.word_metadata_by_char[word[0]].append((word, math.log(frequency)))

        # Heuristics: sort each word_metadata_by_char value lexically
        for prefix_char, metadata in self.word_metadata_by_char.items():
            self.word_metadata_by_char[prefix_char] = sorted(metadata)

        if not os.path.exists(DB_FOLDER_NAME) or overwrite_db:
            if not os.path.exists(DB_FOLDER_NAME):
                os.mkdir(DB_FOLDER_NAME)

            # Reference: https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores
            mp = multiprocess.Pool(multiprocess.cpu_count() - 1)
            mp.starmap(
                build_and_serialize_dfa_db,
                [
                    (
                        word_metadata[idx : idx + CHUNK_SIZE],
                        prefix_char,
                        idx // CHUNK_SIZE,
                    )
                    for prefix_char, word_metadata in self.word_metadata_by_char.items()
                    for idx in range(0, len(word_metadata), CHUNK_SIZE)
                ],
            )
            print("dbs were created and cached")

        for prefix_char in self.word_metadata_by_char.keys():
            db_folder_name_with_prefix_char = os.path.join(
                DB_FOLDER_NAME, str(ord(prefix_char))
            )
            for file_name in sorted(os.listdir(db_folder_name_with_prefix_char)):
                if not file_name.endswith(".db"):
                    continue
                db_file_name = os.path.join(db_folder_name_with_prefix_char, file_name)
                with open(db_file_name, "rb") as f:
                    serialized_db = f.read()
                self.dbs_by_char[prefix_char].append(hyperscan.loadb(serialized_db))
        print("dbs were retrieved from cache")

    @staticmethod
    def on_match(
        regex_idx: int,
        st_pos: int,
        ed_pos: int,
        flags: int,
        context: Dict[str, Any] = None,
    ) -> None:
        assert "matched_regex_metadata" in context
        context["matched_regex_metadata"].append((regex_idx, ed_pos - st_pos))

    def dp_wordfreq(self, pos: int) -> float:
        if pos == self.n:
            return 0
        if self.vis[pos]:
            return self.rcd[pos][0]
        # Special case: always step forward at punctuations
        if self.input_str[pos] in string.punctuation:
            self.rcd[pos] = (self.dp_wordfreq(pos + 1), pos + 1)
            self.vis[pos] = True
            return self.rcd[pos][0]

        for n_pos in range(pos + 1, self.n + 1):
            n_score = self.dp_wordfreq(n_pos)
            word = self.input_str[pos:n_pos]
            prefix_char = word[0]
            for db_idx, db in enumerate(self.dbs_by_char[prefix_char]):
                matched_regex_metadata: List[Tuple[int, int]] = []
                db.scan(
                    word.encode(),
                    match_event_handler=self.on_match,
                    context={"matched_regex_metadata": matched_regex_metadata},
                )

                # There could still be multiple matches. For example, "aaa" matches "a+", "a+a+", "a+a+a+".
                for matched_regex_idx, matched_regex_len in matched_regex_metadata:
                    score = self.word_metadata_by_char[prefix_char][
                        db_idx * CHUNK_SIZE + matched_regex_idx
                    ][1]

                    self.rcd[pos] = max(
                        self.rcd[pos],
                        (
                            score + n_score,
                            n_pos,
                        ),
                    )
                    if self.debug_mode:
                        self.debug_logs.append(
                            f"{self.raw_str[pos:]}: '{word}' ({score:.5f}) + '{self.raw_str[n_pos:]}' ({n_score:.5f})"
                        )

        self.vis[pos] = True
        return self.rcd[pos][0]

    def backtrack(self) -> Tuple[List[str], List[float]]:
        segments: List[str] = []
        scores: List[float] = []
        pos = 0
        while pos < self.n:
            if self.rcd[pos][1] == -1:
                n_pos = self.n
            else:
                n_pos = self.rcd[pos][1]
            segments.append(self.raw_str[pos:n_pos])
            scores.append(
                self.rcd[pos][0] - self.rcd[n_pos][0]
                if n_pos < self.n
                else self.rcd[pos][0]
            )
            pos = n_pos
        return segments, scores

    def segment_sentence(
        self, sentence: str, debug_mode=False
    ) -> Tuple[List[str], List[float], float, List[str]]:
        self.debug_mode = debug_mode
        self.debug_logs = []

        # tokens = wordfreq.lossy_tokenize(
        #     text=sentence.lower(), lang=self.lang, include_punctuation=True
        # )
        tokens = sentence.lower().split()
        if not tokens:
            return [""], [MIN_FLOAT_VAL], MIN_FLOAT_VAL, self.debug_logs

        one_over_result = 0.0
        all_segments = []
        all_scores = []
        for token in tokens:
            self.raw_str = token
            self.input_str = wordfreq.smash_numbers(token)
            self.n = len(self.input_str)
            self.vis = [False for _ in range(self.n)]
            self.rcd = [(MIN_FLOAT_VAL, -1) for _ in range(self.n)]
            self.dp_wordfreq(0)
            segments, scores = self.backtrack()

            if self.input_str != self.raw_str:
                # If there is a digit sequence in the token, the digits are
                # internally replaced by 0s to aggregate their probabilities
                # together. We then assign a specific frequency to the digit
                # sequence using the `digit_freq` distribution.
                scores = [
                    score + math.log(wordfreq.digit_freq(token)) for score in scores
                ]
            all_segments += segments
            all_scores += scores

            # Frequencies for multiple tokens are combined using the formula
            #     1 / f = 1 / f1 + 1 / f2 + ...
            # Thus the resulting frequency is less than any individual frequency, and
            # the smallest frequency dominates the sum.
            for score in scores:
                one_over_result += (
                    1.0 / math.exp(score) if score > MIN_FLOAT_VAL else MAX_FLOAT_VAL
                )

        # Combine the frequencies of tokens we looked up.
        overall_frequency = 1.0 / one_over_result

        if wordfreq.get_language_info(self.lang)["tokenizer"] == "jieba":
            # If we used the Jieba tokenizer, we could tokenize anything to match
            # our wordlist, even nonsense. To counteract this, we multiply by a
            # probability for each word break that was inferred.
            overall_frequency *= wordfreq.INFERRED_SPACE_FACTOR ** -(
                len(all_segments) - 1
            )

        return (
            all_segments,
            all_scores,
            math.log(overall_frequency) if overall_frequency > 0 else MIN_FLOAT_VAL,
            self.debug_logs,
        )
