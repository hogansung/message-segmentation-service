import sys
from typing import List, Union, Dict
from flask import request, flash, Flask

from message_segmentor_v0 import MessageSegmentorV0

# from message_segmentor_v1 import MessageSegmentorV1

api = Flask(__name__)
message_segmentor = MessageSegmentorV0(b_overwrite_db=False)
# message_segmentor = MessageSegmentorV1(b_overwrite_db=False)

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
