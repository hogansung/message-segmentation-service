from typing import List, Union, Dict

from flask import request, flash, Flask

from message_segmentor import MessageSegmentor

api = Flask(__name__)
message_segmentor = MessageSegmentor()


@api.route("/query", methods=["POST"])
def query() -> Dict[str, List[Dict[str, Union[str, float]]]]:
    json = request.json
    if "message" not in json:
        flash("`message` is required.")
    if "b_optimized" not in json:
        flash("`b_optimized` is required.")

    message = request.json["message"]
    b_optimized = request.json["b_optimized"]

    segments, scores = message_segmentor.segment_word(
        input_str=message, b_optimized=b_optimized, debug_flag=False
    )
    print(segments)

    return {
        "response_entities": [
            {
                "segment": segment,
                "score": score,
            }
            for segment, score in zip(segments, scores)
        ]
    }
