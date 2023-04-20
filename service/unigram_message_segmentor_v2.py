import gzip
import os
import re
from typing import Tuple, List

import math
import wordfreq
from tqdm.autonotebook import tqdm

from service.unigram_message_segmentor_v0 import UnigramMessageSegmentorV0


class UnigramMessageSegmentorV2(UnigramMessageSegmentorV0):
    """
    It runs at a complexity of `O(N * (N + K))`, where `N` is the length of string and `K` is the number of candidates.
    After adding `+` quantifier to each character of regex, the complexity of `K` could be as large as `N^2`. However,
    this could be prevented by preprocess the regexes.

    TODO: Preprocess the regexes to reduce the complexity to O(N^2) instead of the naive O(N^3).
    """

    unigram_folder_path = "./dat/google-dataset-v3/cleaned/v1/unigram"
    db_folder_path = "./dbs/serialized_dfa_dbs_v2"
    repetition_pattern = r"{1,2}"
    cpu_count = 16

    @staticmethod
    def reduce_codepoints(
        token: str,
    ) -> str:
        return re.sub(r"(.)\1+", r"\1\1", token)

    @staticmethod
    def reduce_digits(token: str) -> str:
        return re.sub(
            r"(\d{3,}|(\d)((?!\2))\d)", lambda m: "0" * len(m.group(1)), token
        )

    def segment_sentence(
        self, sentence: str, debug_mode=False
    ) -> Tuple[List[str], List[float], float, List[str]]:
        self.debug_mode = debug_mode
        self.debug_logs = []

        tokens = sentence.lower().split()
        if not tokens:
            return [""], [self.min_float_val], self.min_float_val, self.debug_logs

        one_over_result = 0.0
        all_segments = []
        all_scores = []
        self.num_op = 0
        for token in tokens:
            # Reduce two or more, same, consecutive codepoints into two codepoints.
            # TODO: We can actually do this normalization behind the scene, but it requires more sophisticated logic
            #  to maintain the match boundaries and recover the token at the end.
            token = UnigramMessageSegmentorV2.reduce_codepoints(token)
            self.raw_str = token
            self.num_codepoints = len(self.raw_str)
            # Reduce more than two, consecutive digits, or exactly two, different, consecutive digits into zeros.
            self.smashed_str = UnigramMessageSegmentorV2.reduce_digits(token)
            self._convert_str_to_bytes_and_build_lookup_table()
            self.vis = [False for _ in range(self.num_codepoints)]
            self.rcd = [(self.min_float_val, -1) for _ in range(self.num_codepoints)]
            self.dp_wordfreq(0)
            segments, scores = self._backtrack()

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

    def _load_words(self):
        total_occurrence = 0
        for file_name in tqdm(
            sorted(os.listdir(UnigramMessageSegmentorV2.unigram_folder_path))
        ):
            if not file_name.endswith(".gz"):
                continue
            file_path = os.path.join(
                UnigramMessageSegmentorV2.unigram_folder_path,
                file_name,
            )
            with gzip.open(file_path, "rt") as f:
                for line in f:
                    word, occurrence = line.strip().split()
                    total_occurrence += float(occurrence)
                    self.word_metadata_by_codepoint[word[0]].append(
                        (word, float(occurrence))
                    )

        # Convert occurrence to log-frequency
        # Heuristics: sort each word_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in tqdm(self.word_metadata_by_codepoint.items()):
            # The original file is already sorted by codepoints
            self.word_metadata_by_codepoint[prefix_codepoint] = [
                (word, math.log(occurrence / total_occurrence))
                for word, occurrence in metadata
            ]
        print("Finish reading source file")
