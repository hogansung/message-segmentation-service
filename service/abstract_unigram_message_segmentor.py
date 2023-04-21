from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import math
import wordfreq
from tqdm.autonotebook import tqdm

from service.abstract_message_segmentor import AbstractMessageSegmentor


class AbstractUnigramMessageSegmentor(AbstractMessageSegmentor, ABC):
    def __init__(self, overwrite_timestamp: Optional[float]) -> None:
        self.vis: List[bool] = []
        self.rcd: List[
            Tuple[float, int]
        ] = []  # codepoint_idx -> (score, n_codepoint_idx)
        super().__init__(overwrite_timestamp)

    def _load_data(self) -> None:
        self._load_unigrams()

    def _load_unigrams(self) -> None:
        # Reference: https://github.com/rspeer/wordfreq
        frequency_dict = wordfreq.get_frequency_dict(self.lang)
        for unigram, frequency in tqdm(
            frequency_dict.items(),
            desc="Load Unigram Frequency from `wordfreq`",
        ):
            self.unigram_metadata_by_codepoint[unigram[0]].append(
                (unigram, math.log(frequency))
            )

        # Heuristics: sort each unigram_metadata_by_codepoint value lexically
        for prefix_codepoint, metadata in tqdm(
            self.unigram_metadata_by_codepoint.items(),
            desc="Sort `unigram_metadata_by_codepoint` lexically",
        ):
            self.unigram_metadata_by_codepoint[prefix_codepoint] = sorted(metadata)

    def _dp_segmentation(self, codepoint_idx: int) -> float:
        return self._dp_unigram_segmentation(codepoint_idx)

    @abstractmethod
    def _dp_unigram_segmentation(self, codepoint_idx: int) -> float:
        raise NotImplemented

    def _backtrack(self) -> Tuple[List[str], List[float]]:
        return self._backtrack_unigram()

    def _backtrack_unigram(self) -> Tuple[List[str], List[float]]:
        segments: List[str] = []
        scores: List[float] = []
        codepoint_idx = 0
        while codepoint_idx < self.num_codepoints:
            if self.rcd[codepoint_idx][1] == -1:
                assert False
            score, n_codepoint_idx = self.rcd[codepoint_idx]
            n_score = (
                self.rcd[n_codepoint_idx][0]
                if n_codepoint_idx < self.num_codepoints
                else 0
            )
            segments.append(self.raw_str[codepoint_idx:n_codepoint_idx])
            scores.append(score - n_score)
            codepoint_idx = n_codepoint_idx
        return segments, scores
