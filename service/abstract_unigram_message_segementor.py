import pathlib

import math
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import hyperscan
import multiprocess
import wordfreq
from tqdm.autonotebook import tqdm


class AbstractUnigramMessageSegmentor(ABC):
    def __init__(self, overwrite_timestamp: Union[float, None] = None) -> None:
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
        self.byte_idx_to_codepoint_idx = None
        self.codepoint_idx_to_byte_idx = None
        self.vis: List[bool] = []
        self.rcd: List[Tuple[float, int]] = []  # score, count, codepoint_idx
        self.num_op: int = 0
        self.inf: int = sys.maxsize
        self.debug_mode: bool = False
        self.debug_logs: List[str] = []
        self._load_words()
        self._load_dbs(overwrite_timestamp)

    @property
    @abstractmethod
    def db_folder_path(self) -> str:
        pass

    @property
    def chunk_size(self) -> int:
        return 10000

    @property
    def max_float_val(self) -> float:
        return float("+inf")

    @property
    def min_float_val(self) -> float:
        return float("-inf")

    @property
    def min_log_prob(self) -> float:
        return -30.0

    @property
    def max_repetition(self) -> int:
        return 10

    @property
    def separation_pattern(self) -> str:
        return ""

    @property
    def repetition_pattern(self) -> str:
        return "+"

    @property
    def cpu_count(self) -> int:
        return multiprocess.cpu_count() - 1

    @abstractmethod
    def dp_wordfreq(self, codepoint_idx: int) -> float:
        pass

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
            return [""], [self.min_float_val], self.min_float_val, self.debug_logs

        one_over_result = 0.0
        all_segments = []
        all_scores = []
        self.num_op = 0
        for token in tokens:
            self.raw_str = token
            self.num_codepoints = len(self.raw_str)
            self.smashed_str = wordfreq.smash_numbers(token)
            self._convert_str_to_bytes_and_build_lookup_table()
            self.vis = [False for _ in range(self.num_codepoints)]
            self.rcd = [(self.min_float_val, -1) for _ in range(self.num_codepoints)]
            self.dp_wordfreq(0)
            segments, scores = self._backtrack()

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
                    1.0 / math.exp(score)
                    if score > self.min_float_val
                    else self.max_float_val
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
            math.log(overall_frequency)
            if overall_frequency > 0
            else self.min_float_val,
            self.debug_logs,
        )

    def _build_and_serialize_dfa_db(
        self,
        word_metadata: List[Tuple[str, int]],
        prefix_codepoint: str,
        chunk_idx: int,
        db_file_path: str,
        txt_file_path: str,
    ) -> None:
        start_time = time.time()
        expressions = [
            # Make sure each regex has the start anchor, separation pattern, and `+` quantifier
            f"^{self.separation_pattern.join([re.escape(c) + self.repetition_pattern for c in word])}".encode()
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
            db_file_path,
            "wb",
        ) as f:
            f.write(serialized_db)
        with open(
            txt_file_path,
            "w",
        ) as f:
            for word, count in word_metadata:
                f.write(f"{word} {count}\n")

        end_time = time.time()
        print(
            f"Building db for {prefix_codepoint} -- {chunk_idx:02d} with {len(word_metadata)} entries takes "
            f"{end_time - start_time:.2f} seconds",
            flush=True,
        )

    def _load_words(self):
        # Reference: https://github.com/rspeer/wordfreq
        frequency_dict = wordfreq.get_frequency_dict(self.lang)
        for word, frequency in tqdm(frequency_dict.items()):
            self.word_metadata_by_codepoint[word[0]].append((word, math.log(frequency)))

        # Heuristics: sort each word_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in tqdm(self.word_metadata_by_codepoint.items()):
            self.word_metadata_by_codepoint[prefix_codepoint] = sorted(metadata)

    def _load_dbs(self, overwrite_timestamp: Union[float, None]):
        print(f"Number of CPUs: {self.cpu_count}")

        if not os.path.exists(self.db_folder_path) or overwrite_timestamp is not None:
            pathlib.Path(self.db_folder_path).mkdir(parents=True, exist_ok=True)

            # Prepare data to be multi-processed.
            data_to_be_multi_processed = []
            skipped_count = 0
            for (
                prefix_codepoint,
                word_metadata,
            ) in self.word_metadata_by_codepoint.items():
                db_folder_path_with_prefix_codepoint = os.path.join(
                    self.db_folder_path, str(ord(prefix_codepoint))
                )
                pathlib.Path(db_folder_path_with_prefix_codepoint).mkdir(
                    parents=True, exist_ok=True
                )
                for idx in range(0, len(word_metadata), self.chunk_size):
                    chunk_idx = idx // self.chunk_size
                    db_file_path = os.path.join(
                        db_folder_path_with_prefix_codepoint, f"{chunk_idx:02d}.db"
                    )
                    txt_file_path = os.path.join(
                        db_folder_path_with_prefix_codepoint, f"{chunk_idx:02d}.txt"
                    )
                    if (
                        os.path.exists(db_file_path)
                        and os.path.getctime(db_file_path) >= overwrite_timestamp
                    ):
                        # print(
                        #     f"db for {prefix_codepoint} -- {chunk_idx:02d} with {len(word_metadata)} entries exists. "
                        #     f"Skipped.",
                        #     flush=True,
                        # )
                        skipped_count += 1
                    else:
                        data_to_be_multi_processed.append(
                            (
                                word_metadata[idx : idx + self.chunk_size],
                                prefix_codepoint,
                                chunk_idx,
                                db_file_path,
                                txt_file_path,
                            )
                        )
            print(
                f"Cached rate: {skipped_count/(skipped_count + len(data_to_be_multi_processed))*100}%"
                f" (={len(data_to_be_multi_processed)} / {skipped_count + len(data_to_be_multi_processed)})"
            )

            # Reference: https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores
            start_time = time.time()
            mp = multiprocess.Pool(self.cpu_count)
            mp.starmap(
                self._build_and_serialize_dfa_db,
                data_to_be_multi_processed,
            )
            end_time = time.time()
            print(f"dbs were created and cached in {end_time - start_time} seconds")

        for prefix_codepoint in tqdm(self.word_metadata_by_codepoint.keys()):
            db_folder_path_with_prefix_codepoint = os.path.join(
                self.db_folder_path, str(ord(prefix_codepoint))
            )
            pathlib.Path(db_folder_path_with_prefix_codepoint).mkdir(
                parents=True, exist_ok=True
            )
            for file_name in sorted(os.listdir(db_folder_path_with_prefix_codepoint)):
                if not file_name.endswith(".db"):
                    continue
                db_file_name = os.path.join(
                    db_folder_path_with_prefix_codepoint, file_name
                )
                with open(db_file_name, "rb") as f:
                    serialized_db = f.read()
                self.dbs_by_codepoint[prefix_codepoint].append(
                    hyperscan.loadb(serialized_db)
                )
        print("dbs were retrieved from cache")

    @staticmethod
    def _on_match(
        regex_idx: int,
        st_byte_idx: int,
        ed_byte_idx: int,
        flags: int,
        context: Dict[str, Any] = None,
    ) -> None:
        assert "matched_regex_metadata" in context
        context["matched_regex_metadata"].append((regex_idx, ed_byte_idx - st_byte_idx))

    def _backtrack(self) -> Tuple[List[str], List[float]]:
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

    def _convert_str_to_bytes_and_build_lookup_table(self) -> None:
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
