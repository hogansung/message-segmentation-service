import math
from tqdm.autonotebook import tqdm

from service.unigram_message_segmentor_v0 import UnigramMessageSegmentorV0


class UnigramMessageSegmentorV1(UnigramMessageSegmentorV0):
    """
    It runs at a complexity of `O(N * (N + K))`, where `N` is the length of string and `K` is the number of candidates.
    After adding `+` quantifier to each character of regex, the complexity of `K` could be as large as `N^2`. However,
    this could be prevented by preprocess the regexes.

    TODO: Preprocess the regexes to reduce the complexity to O(N^2) instead of the naive O(N^3).
    """

    en_occurrence_file_path = "./dat/FrequencyWords/en_full.txt"
    db_folder_path = "./dbs/serialized_unigram_dfa_dbs_v1"

    def _load_unigrams(self):
        total_occurrence = 0
        with open(UnigramMessageSegmentorV1.en_occurrence_file_path) as f:
            for line in tqdm(
                f.readlines(),
                desc="Load Unigram Metadata",
            ):
                unigram, occurrence = line.strip().split()
                total_occurrence += float(occurrence)
                self.unigram_metadata_by_codepoint[unigram[0]].append(
                    (unigram, float(occurrence))
                )

        # Convert occurrence to log-frequency
        # Heuristics: sort each unigram_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in tqdm(
            self.unigram_metadata_by_codepoint.items(),
            desc="Convert Unigram Occurrence to Log-Frequency",
        ):
            self.unigram_metadata_by_codepoint[prefix_codepoint] = sorted(
                [
                    (unigram, math.log(occurrence / total_occurrence))
                    for unigram, occurrence in metadata
                ]
            )
        print("Finish reading unigram files")
