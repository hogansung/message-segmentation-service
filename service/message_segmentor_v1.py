import pathlib

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

DB_FOLDER_NAME = "./dat/serialized_word_freq_in_chunks_v1"

CHUNK_SIZE = 10000
MAX_FLOAT_VAL = float("+inf")
MIN_FLOAT_VAL = float("-inf")

MAX_REPETITION = 10
# SEPARATION_PATTERN = (
#     rf"([^a-z]{1,MAX_REPETITION}|a{3,MAX_REPETITION}|b{3,MAX_REPETITION}|c{3,MAX_REPETITION}|d{3,MAX_REPETITION}|"
#     rf"e{3,MAX_REPETITION}|f{3,MAX_REPETITION}|g{3,MAX_REPETITION}|h{3,MAX_REPETITION}|i{3,MAX_REPETITION}|"
#     rf"j{3,MAX_REPETITION}|k{3,MAX_REPETITION}|l{3,MAX_REPETITION}|m{3,MAX_REPETITION}|n{3,MAX_REPETITION}|"
#     rf"o{3,MAX_REPETITION}|p{3,MAX_REPETITION}|q{3,MAX_REPETITION}|r{3,MAX_REPETITION}|s{3,MAX_REPETITION}|"
#     rf"t{3,MAX_REPETITION}|u{3,MAX_REPETITION}|v{3,MAX_REPETITION}|w{3,MAX_REPETITION}|x{3,MAX_REPETITION}|"
#     rf"y{3,MAX_REPETITION}|z{3,MAX_REPETITION})?"
# )
SEPARATION_PATTERN = ""
# REPETITION_PATTERN = f"{1,MAX_REPETITION}"
REPETITION_PATTERN = f"+"


def build_and_serialize_dfa_db(
    word_metadata: List[Tuple[str, int]],
    prefix_codepoint: str,
    chunk_idx: int,
) -> None:
    start_time = time.time()

    db_folder_name_with_prefix_codepoint = os.path.join(
        DB_FOLDER_NAME, str(ord(prefix_codepoint))
    )
    pathlib.Path(db_folder_name_with_prefix_codepoint).mkdir(
        parents=True, exist_ok=True
    )

    expressions = [
        # Make sure each regex has the start anchor, separation pattern, and `+` quantifier
        f"^{SEPARATION_PATTERN.join([re.escape(c) + REPETITION_PATTERN for c in word])}".encode()
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
        os.path.join(db_folder_name_with_prefix_codepoint, f"{chunk_idx:02d}.db"),
        "wb",
    ) as f:
        f.write(serialized_db)
    with open(
        os.path.join(db_folder_name_with_prefix_codepoint, f"{chunk_idx:02d}.txt"),
        "w",
    ) as f:
        for word, count in word_metadata:
            f.write(f"{word} {count}\n")

    end_time = time.time()
    print(
        f"Building db for {prefix_codepoint} -- {chunk_idx:02d} with {len(word_metadata)} entries takes "
        f"{end_time - start_time:.2f} seconds"
    )


class MessageSegmentorV1:
    def __init__(self, b_overwrite_db=False) -> None:
        self.byte_idx_to_codepoint_idx = None
        self.codepoint_idx_to_byte_idx = None
        self.lang = "en"
        self.word_metadata_by_codepoint: Dict[
            str, List[Tuple[str, float]]
        ] = defaultdict(list)
        self.dbs_by_codepoint: Dict[str, List[hyperscan.Database]] = defaultdict(list)
        self.raw_str: str = ""
        self.smashed_str: str = ""
        self.num_codepoints: int = 0
        self.smashed_bytes: bytes = b""
        self.num_bytes: int = 0
        self.vis: List[bool] = []
        self.rcd: List[Tuple[float, int]] = []  # score, count, codepoint_idx
        self.num_op: int = 0
        self.inf: int = sys.maxsize
        self.debug_mode: bool = False
        self.debug_logs: List[str] = []
        self.load_words_and_db(b_overwrite_db)

    def load_words_and_db(self, b_overwrite_db: bool):
        # Reference: https://github.com/rspeer/wordfreq
        frequency_dict = wordfreq.get_frequency_dict(self.lang)
        for word, frequency in frequency_dict.items():
            self.word_metadata_by_codepoint[word[0]].append((word, math.log(frequency)))

        # Heuristics: sort each word_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in self.word_metadata_by_codepoint.items():
            self.word_metadata_by_codepoint[prefix_codepoint] = sorted(metadata)

        if not os.path.exists(DB_FOLDER_NAME) or b_overwrite_db:
            pathlib.Path(DB_FOLDER_NAME).mkdir(parents=True, exist_ok=True)

            # Reference: https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores
            mp = multiprocess.Pool(multiprocess.cpu_count() - 1)
            mp.starmap(
                build_and_serialize_dfa_db,
                [
                    (
                        word_metadata[idx : idx + CHUNK_SIZE],
                        prefix_codepoint,
                        idx // CHUNK_SIZE,
                    )
                    for prefix_codepoint, word_metadata in self.word_metadata_by_codepoint.items()
                    for idx in range(0, len(word_metadata), CHUNK_SIZE)
                ],
            )
            print("dbs were created and cached")

        for prefix_codepoint in self.word_metadata_by_codepoint.keys():
            db_folder_name_with_prefix_codepoint = os.path.join(
                DB_FOLDER_NAME, str(ord(prefix_codepoint))
            )
            pathlib.Path(db_folder_name_with_prefix_codepoint).mkdir(
                parents=True, exist_ok=True
            )
            for file_name in sorted(os.listdir(db_folder_name_with_prefix_codepoint)):
                if not file_name.endswith(".db"):
                    continue
                db_file_name = os.path.join(
                    db_folder_name_with_prefix_codepoint, file_name
                )
                with open(db_file_name, "rb") as f:
                    serialized_db = f.read()
                self.dbs_by_codepoint[prefix_codepoint].append(
                    hyperscan.loadb(serialized_db)
                )
        print("dbs were retrieved from cache")

    @staticmethod
    def on_match(
        regex_idx: int,
        st_byte_idx: int,
        ed_byte_idx: int,
        flags: int,
        context: Dict[str, Any] = None,
    ) -> None:
        assert "matched_regex_metadata" in context
        context["matched_regex_metadata"].append((regex_idx, ed_byte_idx - st_byte_idx))

    def dp_wordfreq(self, codepoint_idx: int) -> float:
        if codepoint_idx == self.num_codepoints:
            return 0
        if self.vis[codepoint_idx]:
            return self.rcd[codepoint_idx][0]
        # Special case: always step forward at punctuations
        if self.smashed_str[codepoint_idx] in string.punctuation:
            self.rcd[codepoint_idx] = (
                self.dp_wordfreq(codepoint_idx + 1),
                codepoint_idx + 1,
            )
            self.vis[codepoint_idx] = True
            return self.rcd[codepoint_idx][0]

        prefix_codepoint = self.smashed_str[codepoint_idx]
        byte_idx = self.codepoint_idx_to_byte_idx[codepoint_idx]
        for db_idx, db in enumerate(self.dbs_by_codepoint[prefix_codepoint]):
            matched_regex_metadata: List[Tuple[int, int]] = []
            db.scan(
                self.smashed_bytes[byte_idx:],
                match_event_handler=self.on_match,
                context={"matched_regex_metadata": matched_regex_metadata},
            )
            self.num_op += self.num_bytes - byte_idx

            for matched_regex_idx, matched_regex_len in matched_regex_metadata:
                self.num_op += 1
                score = self.word_metadata_by_codepoint[prefix_codepoint][
                    db_idx * CHUNK_SIZE + matched_regex_idx
                ][1]
                n_codepoint_idx = self.byte_idx_to_codepoint_idx[
                    byte_idx + matched_regex_len
                ]
                n_score = self.dp_wordfreq(n_codepoint_idx)
                self.rcd[codepoint_idx] = max(
                    self.rcd[codepoint_idx],
                    (
                        score + n_score,
                        n_codepoint_idx,
                    ),
                )
                if self.debug_mode:
                    self.debug_logs.append(
                        f"{self.raw_str[codepoint_idx:]}: '{self.raw_str[codepoint_idx:n_codepoint_idx]}' ({score:.5f})"
                        f" + '{self.raw_str[n_codepoint_idx:]}' ({n_score:.5f})"
                    )

        self.vis[codepoint_idx] = True
        return self.rcd[codepoint_idx][0]

    def backtrack(self) -> Tuple[List[str], List[float]]:
        segments: List[str] = []
        scores: List[float] = []
        codepoint_idx = 0
        while codepoint_idx < self.num_codepoints:
            if self.rcd[codepoint_idx][1] == -1:
                n_codepoint_idx = self.num_codepoints
            else:
                n_codepoint_idx = self.rcd[codepoint_idx][1]
            segments.append(self.raw_str[codepoint_idx:n_codepoint_idx])
            scores.append(
                self.rcd[codepoint_idx][0] - self.rcd[n_codepoint_idx][0]
                if n_codepoint_idx < self.num_codepoints
                else self.rcd[codepoint_idx][0]
            )
            codepoint_idx = n_codepoint_idx
        return segments, scores

    def segment_sentence(
        self, sentence: str, debug_mode=False
    ) -> Tuple[List[str], List[float], float, List[str]]:
        self.debug_mode = debug_mode
        self.debug_logs = []

        # tokens = wordfreq.lossy_tokenize(
        #     text=sentence.lower(), lang=self.lang, include_punctuation=True
        # )
        # tokens = sentence.lower().split()
        tokens = list(filter(None, re.split(r"(\d[\d.,]+)| ", sentence.lower())))

        if not tokens:
            return [""], [MIN_FLOAT_VAL], MIN_FLOAT_VAL, self.debug_logs

        one_over_result = 0.0
        all_segments = []
        all_scores = []
        self.num_op = 0
        for token in tokens:
            self.raw_str = token
            self.num_codepoints = len(self.raw_str)
            self.smashed_str = wordfreq.smash_numbers(token)
            self.convert_str_to_bytes_and_build_lookup_table()
            print(self.byte_idx_to_codepoint_idx)
            print(self.codepoint_idx_to_byte_idx)
            self.vis = [False for _ in range(self.num_codepoints)]
            self.rcd = [(MIN_FLOAT_VAL, -1) for _ in range(self.num_codepoints)]
            self.dp_wordfreq(0)
            segments, scores = self.backtrack()

            if self.smashed_str != self.raw_str:
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
        print(f"num op: {self.num_op}")

        return (
            all_segments,
            all_scores,
            math.log(overall_frequency) if overall_frequency > 0 else MIN_FLOAT_VAL,
            self.debug_logs,
        )

    def convert_str_to_bytes_and_build_lookup_table(self) -> None:
        self.byte_idx_to_codepoint_idx = dict()
        self.codepoint_idx_to_byte_idx = dict()
        byte_strs = []
        self.num_bytes = 0
        self.num_codepoints = 0
        for codepoint in self.smashed_str:
            byte_str = codepoint.encode()
            byte_strs.append(byte_str)
            self.byte_idx_to_codepoint_idx[self.num_bytes] = self.num_codepoints
            self.codepoint_idx_to_byte_idx[self.num_codepoints] = self.num_bytes
            self.num_bytes += len(byte_str)
            self.num_codepoints += 1
        self.byte_idx_to_codepoint_idx[self.num_bytes] = self.num_codepoints
        self.codepoint_idx_to_byte_idx[self.num_codepoints] = self.num_bytes
        self.smashed_bytes = b"".join(byte_strs)
