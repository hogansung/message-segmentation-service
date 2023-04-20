import gzip
import os

import math
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
    repetition_pattern = "+"
    # cpu_count = 4

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
