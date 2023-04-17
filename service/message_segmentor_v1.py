import string
from typing import List, Tuple

from service.abstract_message_segementor import AbstractMessageSegmentor


class MessageSegmentorV1(AbstractMessageSegmentor):
    db_folder_name = "./dat/serialized_word_freq_in_chunks_v1"

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

        # Default a minimum prob to move forward
        self.rcd[codepoint_idx] = max(
            self.rcd[codepoint_idx],
            (
                self.min_log_prob + self.dp_wordfreq(codepoint_idx + 1),
                codepoint_idx + 1,
            ),
        )

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
                score = self.word_metadata_by_codepoint[prefix_codepoint][
                    db_idx * self.chunk_size + matched_regex_idx
                ][1]
                self.num_op += 1
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
