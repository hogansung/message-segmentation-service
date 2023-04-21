import sys
import time
from typing import List, Union, Dict
from flask import request, flash, Flask

from unigram_message_segmentor_exact_match import MessageSegmentorExactMatch
from unigram_message_segmentor_v0 import UnigramMessageSegmentorV0
from unigram_message_segmentor_v1 import UnigramMessageSegmentorV1
from unigram_message_segmentor_v2 import UnigramMessageSegmentorV2
from bigram_message_segmentor_v0 import BigramMessageSegmentorV0

api = Flask(__name__)

# overwrite_timestamp = time.mktime(
#     time.strptime("2023-04-20 20:00:00", "%Y-%m-%d %H:%M:%S")
# )
overwrite_timestamp = None

# message_segmentor = UnigramMessageSegmentorExactMatch(b_overwrite_db=True)
# message_segmentor = UnigramMessageSegmentorV0(b_overwrite_db=False)
# message_segmentor = UnigramMessageSegmentorV1(b_overwrite_db=True)
# message_segmentor = UnigramMessageSegmentorV2(overwrite_timestamp)
message_segmentor = BigramMessageSegmentorV0(overwrite_timestamp)

sys.setrecursionlimit(10000)


@api.route("/query", methods=["POST"])
def query() -> Dict[str, List[Dict[str, Union[str, float]]]]:
    json = request.json
    print(json)
    if "message" not in json:
        flash("`message` is required.")
    if "debug_mode" not in json:
        flash("`debug_mode` is required.")

    # # To make escaped message normal message. For example r"\u1234" should be regarded as "\u1234".
    # message = (
    #     request.json["message"]
    #     .encode("utf-8", "surrogateescape")
    #     .decode("unicode-escape")
    # )
    message = request.json["message"]
    debug_mode = request.json["debug_mode"]

    segments, scores, overall_score, debug_logs = message_segmentor.segment_sentence(
        sentence=message, debug_mode=debug_mode
    )

    return {
        "response_entities": [
            {
                "segment": segment,
                # Serialization on inf won't work, so return `None` instead
                "score": score if score != float("-inf") else None,
            }
            for segment, score in zip(segments, scores)
        ],
        "overall_score": overall_score,
        "debug_logs": debug_logs,
    }
