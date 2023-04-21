import gzip
import os
from collections import defaultdict
from typing import Tuple, List

import math
import wordfreq
from tqdm.autonotebook import tqdm

from service.abstract_bigram_message_segmentor import AbstractBigramMessageSegmentor
from service.abstract_message_segmentor import AbstractMessageSegmentor


class BigramMessageSegmentorV0(AbstractBigramMessageSegmentor):
    """
    It runs at a complexity of `O(N * (N + K))`, where `N` is the length of string and `K` is the number of candidates.
    After adding `+` quantifier to each character of regex, the complexity of `K` could be as large as `N^2`. However,
    this could be prevented by preprocess the regexes.

    TODO: Preprocess the regexes to reduce the complexity to O(N^2) instead of the naive O(N^3).
    """

    unigram_folder_path = "./dat/google-dataset-v3/cleaned/v1/unigram"
    bigram_folder_path = "./dat/google-dataset-v3/cleaned/v1/bigram"
    # db_folder_path = "./dbs/serialized_bigram_dfa_dbs_v0"
    db_folder_path = "./dbs/serialized_unigram_dfa_dbs_v2"
    repetition_pattern = r"{1,2}"
    cpu_count = 16

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
            token = AbstractMessageSegmentor.reduce_codepoints(token)
            self.raw_str = token
            self.num_codepoints = len(self.raw_str)
            # Reduce more than two, consecutive digits, or exactly two, different, consecutive digits into zeros.
            self.smashed_str = AbstractMessageSegmentor.reduce_digits(token)
            self._convert_str_to_bytes_and_build_lookup_table()
            self.vis = [defaultdict(lambda: False) for _ in range(self.num_codepoints)]
            self.rcd = [
                defaultdict(lambda: (self.min_float_val, -1, ""))
                for _ in range(self.num_codepoints)
            ]
            self._dp_bigram_segmentation(0, "")
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
            # probability for each unigram break that was inferred.
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

    def _dp_bigram_segmentation(self, codepoint_idx: int, fgram: str) -> float:
        if codepoint_idx == self.num_codepoints:
            return 0
        if self.vis[codepoint_idx][fgram]:
            return self.rcd[codepoint_idx][fgram][0]

        # Default a minimum prob to move forward one codepoint
        n_codepoint_idx = codepoint_idx + 1
        sgram = self.smashed_str[codepoint_idx:n_codepoint_idx]
        score = self.min_log_prob
        n_score = self._dp_bigram_segmentation(n_codepoint_idx, sgram)
        self.rcd[codepoint_idx][fgram] = max(
            self.rcd[codepoint_idx][fgram],
            (
                score + n_score,
                n_codepoint_idx,
                sgram,
            ),
        )
        if self.debug_mode:
            self.debug_logs.append(
                f"(min-prob) {self.raw_str[codepoint_idx:]}: '{sgram}' ({score:.5f})"
                f" + '{self.raw_str[n_codepoint_idx:]}' ({n_score:.5f})"
            )

        prefix_codepoint = self.smashed_str[codepoint_idx]
        byte_idx = self.codepoint_idx_to_byte_idx[codepoint_idx]
        for db_idx, db in enumerate(self.dbs_by_codepoint[prefix_codepoint]):
            matched_regex_metadata: List[Tuple[int, int]] = []
            db.scan(
                self.smashed_bytes[byte_idx:],
                match_event_handler=self._on_match,
                context={"matched_regex_metadata": matched_regex_metadata},
            )
            self.num_op += self.num_bytes - byte_idx

            for matched_regex_idx, matched_regex_len in matched_regex_metadata:
                n_codepoint_idx = self.byte_idx_to_codepoint_idx[
                    byte_idx + matched_regex_len
                ]
                sgram = self.smashed_str[codepoint_idx:n_codepoint_idx]
                score = (
                    self.unigram_metadata_by_codepoint[prefix_codepoint][
                        db_idx * self.chunk_size + matched_regex_idx
                    ][1]
                    if fgram is None
                    else self.log_prob_by_fgram_by_sgram[fgram][sgram]
                )
                n_score = self._dp_bigram_segmentation(n_codepoint_idx, sgram)
                self.num_op += len(fgram) + len(sgram)
                self.rcd[codepoint_idx][fgram] = max(
                    self.rcd[codepoint_idx][fgram],
                    (
                        score + n_score,
                        n_codepoint_idx,
                        sgram,
                    ),
                )
                if self.debug_mode:
                    self.debug_logs.append(
                        f"{self.raw_str[codepoint_idx:]}: '{sgram}' ({score:.5f})"
                        f" + '{self.raw_str[n_codepoint_idx:]}' ({n_score:.5f})"
                    )

        self.vis[codepoint_idx][fgram] = True
        return self.rcd[codepoint_idx][fgram][0]

    def _backtrack_bigram(self) -> Tuple[List[str], List[float]]:
        segments: List[str] = []
        scores: List[float] = []
        codepoint_idx = 0
        fgram = ""
        while codepoint_idx < self.num_codepoints:
            if self.rcd[codepoint_idx][fgram][1] == -1:
                assert False
            score, n_codepoint_idx, sgram = self.rcd[codepoint_idx][fgram]
            n_score = (
                self.rcd[n_codepoint_idx][sgram][0]
                if n_codepoint_idx < self.num_codepoints
                else 0
            )
            segments.append(self.raw_str[codepoint_idx:n_codepoint_idx])
            scores.append(score - n_score)
            codepoint_idx = n_codepoint_idx
            fgram = sgram
        return segments, scores

    def _load_unigrams(self):
        total_occurrence = 0
        for file_name in tqdm(
            sorted(os.listdir(BigramMessageSegmentorV0.unigram_folder_path)),
            desc="Load Unigram Metadata",
        ):
            if not file_name.endswith(".gz"):
                continue
            file_path = os.path.join(
                BigramMessageSegmentorV0.unigram_folder_path,
                file_name,
            )
            with gzip.open(file_path, "rt") as f:
                for line in f:
                    unigram, occurrence = line.strip().split()
                    total_occurrence += float(occurrence)
                    self.unigram_metadata_by_codepoint[unigram[0]].append(
                        (unigram, float(occurrence))
                    )

        # Convert occurrence to log-frequency
        for prefix_codepoint, metadata in tqdm(
            self.unigram_metadata_by_codepoint.items(),
            desc="Convert Unigram Occurrence to Log-Frequency",
        ):
            self.unigram_metadata_by_codepoint[prefix_codepoint] = [
                (unigram, math.log(occurrence / total_occurrence))
                for unigram, occurrence in metadata
            ]
        print("Finish reading unigram files")

    def _load_bigrams(self):
        for file_name in tqdm(
            sorted(os.listdir(BigramMessageSegmentorV0.bigram_folder_path)),
            desc="Load Bigram Occurrence from `Google DatasetV3`",
        ):
            if not file_name.endswith(".gz"):
                continue
            file_path = os.path.join(
                BigramMessageSegmentorV0.bigram_folder_path,
                file_name,
            )
            with gzip.open(file_path, "rt") as f:
                for line in f:
                    fgram, sgram, occurrence = line.strip().split()
                    self.log_prob_by_fgram_by_sgram[fgram][sgram] = float(occurrence)

        # Convert conditional occurrence to conditional log-frequency
        for fgram, log_prob_by_sgram in tqdm(
            self.log_prob_by_fgram_by_sgram.items(),
            desc="Calculate Conditional Log-Frequency",
        ):
            fgram_occurrence = sum(log_prob_by_sgram.values())
            for sgram, occurrence in log_prob_by_sgram.items():
                log_prob_by_sgram[sgram] = math.log(occurrence / fgram_occurrence)
        print("Finish reading bigram files")
