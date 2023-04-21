from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

from service.abstract_message_segmentor import AbstractMessageSegmentor


class AbstractBigramMessageSegmentor(AbstractMessageSegmentor, ABC):
    def __init__(self, overwrite_timestamp: Optional[float]) -> None:
        self.vis: List[Dict[str, bool]] = []
        self.rcd: List[
            Dict[str, Tuple[float, int, str]]
        ] = []  # codepoint_idx -> fgram -> (score, n_codepoint_idx, sgram)
        self.log_prob_by_fgram_by_sgram: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        super().__init__(overwrite_timestamp)

    def _load_data(self):
        self._load_unigrams()
        self._load_bigrams()

    @abstractmethod
    def _load_bigrams(self):
        raise NotImplemented

    def _dp_segmentation(self, codepoint_idx: int) -> float:
        return self._dp_bigram_segmentation(codepoint_idx, "")

    @abstractmethod
    def _dp_bigram_segmentation(self, codepoint_idx: int, fgram: str) -> float:
        raise NotImplemented

    def _backtrack(self) -> Tuple[List[str], List[float]]:
        return self._backtrack_bigram()

    @abstractmethod
    def _backtrack_bigram(self) -> Tuple[List[str], List[float]]:
        raise NotImplemented
