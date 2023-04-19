import os
import pathlib

import hyperscan
import math
import multiprocess

from service.message_segmentor_v1 import MessageSegmentorV1


class MessageSegmentorV2(MessageSegmentorV1):
    """
    It runs at a complexity of `O(N * (N + K))`, where `N` is the length of string and `K` is the number of candidates.
    After adding `+` quantifier to each character of regex, the complexity of `K` could be as large as `N^2`. However,
    this could be prevented by preprocess the regexes.

    TODO: Preprocess the regexes to reduce the complexity to O(N^2) instead of the naive O(N^3).
    """

    en_occurrence_file_path = "./dat/en_full.txt"
    db_folder_path = "./dat/serialized_word_freq_in_chunks_v2"

    def _load_words(self):
        total_occurrence = 0
        with open(MessageSegmentorV2.en_occurrence_file_path) as f:
            for line in f.readlines():
                word, occurrence = line.strip().split()
                total_occurrence += float(occurrence)
                self.word_metadata_by_codepoint[word[0]].append(
                    (word, float(occurrence))
                )

        # Convert occurrence to log-frequency
        # Heuristics: sort each word_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in self.word_metadata_by_codepoint.items():
            self.word_metadata_by_codepoint[prefix_codepoint] = sorted(
                [
                    (word, math.log(occurrence / total_occurrence))
                    for word, occurrence in metadata
                ]
            )
        print("Finish reading source file")
