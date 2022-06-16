import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Tuple, List, Dict, Any

import hyperscan
import multiprocess

WORDS_FILE_PATH = "../../dat/en_full.txt"
DB_FOLDER_NAME = "../../res/serialized_en_dbs_full_in_chunks"

CHUNK_SIZE = 10000
FREQUENCY_SUM = 735777659
MIN_FLOAT_VAL = float("-inf")
MIN_ALLOWED_SCORE = -12.0


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
        f"^{''.join([re.escape(c) + '+' for c in word])}".encode()
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
        self.word_metadata_by_char: Dict[str, List[Tuple[str, float]]] = defaultdict(
            list
        )
        self.dbs_by_char: Dict[str, List[hyperscan.Database]] = defaultdict(list)
        self.input_str: str = ""
        self.n: int = 0
        self.rcd: List[Tuple[float, int, int]] = []  # score, count, pos
        self.load_words_and_db(overwrite_db)
        self.inf = sys.maxsize
        self.debug_flag = False

    def load_words_and_db(self, overwrite_db: bool):
        # Reference: https://github.com/dwyl/english-words
        with open(WORDS_FILE_PATH) as f:
            lines = f.readlines()
            for line in lines:
                word, count = line.lower().strip().split(" ")
                self.word_metadata_by_char[word[0]].append(
                    #                     (word, math.log(float(count)) * len(word) ** 2)
                    (word, math.log(float(count)) - math.log(FREQUENCY_SUM))
                )

        # Heuristics: sort each word_metadata_by_char value lexically
        for prefix_char, metadata in self.word_metadata_by_char.items():
            self.word_metadata_by_char[prefix_char] = sorted(metadata)

        #         for prefix_char, metadata in self.word_metadata_by_char.items():
        #             self.word_metadata_by_char[prefix_char] = metadata[:100]

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

    def dp(self, pos: int) -> Tuple[float, int]:
        if pos == self.n:
            return 0, 0
        if self.rcd[pos][0] != MIN_FLOAT_VAL:
            return self.rcd[pos][0], self.rcd[pos][1]

        prefix_char = self.input_str[pos]
        b_no_match = True

        for db_idx, db in enumerate(self.dbs_by_char[prefix_char]):
            matched_regex_metadata: List[Tuple[int, int]] = []
            db.scan(
                self.input_str[pos:].encode(),
                match_event_handler=self.on_match,
                context={"matched_regex_metadata": matched_regex_metadata},
            )

            for matched_regex_idx, matched_regex_len in matched_regex_metadata:
                b_no_match = False
                n_pos = pos + matched_regex_len
                score, num_segments = self.dp(n_pos)
                if self.debug_flag:
                    print(
                        prefix_char,
                        self.input_str[pos:],
                        score,
                        db_idx,
                        matched_regex_idx,
                        self.word_metadata_by_char[prefix_char][
                            db_idx * CHUNK_SIZE + matched_regex_idx
                        ][0],
                        self.word_metadata_by_char[prefix_char][
                            db_idx * CHUNK_SIZE + matched_regex_idx
                        ][1],
                    )
                self.rcd[pos] = max(
                    self.rcd[pos],
                    (
                        (
                            score
                            + self.word_metadata_by_char[prefix_char][
                                db_idx * CHUNK_SIZE + matched_regex_idx
                            ][1]
                        ),
                        num_segments + 1,
                        n_pos,
                    ),
                )

        if b_no_match:
            score, num_segments = self.dp(pos + 1)
            self.rcd[pos] = max(
                self.rcd[pos],
                (
                    score + MIN_ALLOWED_SCORE,
                    num_segments + 1,
                    pos + 1,
                ),
            )

        return self.rcd[pos][0], self.rcd[pos][1]

    def backtrack(self) -> Tuple[List[str], List[float]]:
        segments: List[str] = []
        scores: List[float] = []
        pos = 0
        while pos < self.n:
            n_pos = self.rcd[pos][2]
            scores.append(
                self.rcd[pos][0] - self.rcd[n_pos][0]
                if n_pos < self.n
                else self.rcd[pos][0]
            )
            segments.append(self.input_str[pos:n_pos])
            pos = n_pos
        return segments, scores

    @staticmethod
    def optimize_segments(
        segments: List[str], scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        optimized_segments: List[str] = []
        optimized_scores: List[float] = []
        accumulated_segments = []
        accumulated_score = 0.0
        for segment, score in zip(segments, scores):
            # Special case: we don't want to include spaces as segments
            if segment == " ":
                continue
            if score > MIN_ALLOWED_SCORE:
                if accumulated_segments:
                    optimized_segments.append("".join(accumulated_segments))
                    optimized_scores.append(accumulated_score)
                    accumulated_segments, accumulated_score = [], 0.0
                optimized_segments.append(segment)
                optimized_scores.append(score)
            else:
                accumulated_segments.append(segment)
                accumulated_score += score
        if accumulated_segments:
            optimized_segments.append("".join(accumulated_segments))
            optimized_scores.append(accumulated_score)

        return optimized_segments, optimized_scores

    def segment_word(
        self, input_str: str, b_optimized: bool = True, debug_flag=False
    ) -> Tuple[List[str], List[float]]:
        self.debug_flag = debug_flag
        self.input_str = input_str.lower()
        self.n = len(input_str)
        self.rcd = [(MIN_FLOAT_VAL, -1, -1) for _ in range(self.n)]
        self.dp(0)
        segments, scores = self.backtrack()
        optimized_segments, optimized_scores = self.optimize_segments(segments, scores)
        if b_optimized:
            return optimized_segments, optimized_scores
        else:
            return segments, scores
