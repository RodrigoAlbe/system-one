"""Local validation is mandatory even when a provider enforces a schema."""

import json
import math

from .errors import InvalidResponseError
from .primitives import Choice, Noul, Score, QuestionResult


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty dictionary")
    for key, question in questions.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Question IDs must be non-empty strings")
        if not isinstance(question, (Choice, Noul, Score)):
            raise ValueError("Questions must be Choice, Noul, or Score instances")
        # Recheck mutable dataclasses before each request.
        question.__post_init__()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponseError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise InvalidResponseError("Non-finite JSON number")


def _probability(value, field):
    # bool is an int subclass; numeric strings are deliberately not coerced.
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise InvalidResponseError(f"{field} must be a finite number between 0 and 1")
    if not math.isfinite(value):
        raise InvalidResponseError(f"{field} must be finite")
    return float(value)


def parse_answers(content, questions):
    validate_questions(questions)
    if not isinstance(content, str) or not content.strip():
        raise InvalidResponseError("Response content must be non-empty JSON text")
    try:
        parsed = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except InvalidResponseError:
        raise
    except (ValueError, RecursionError) as exc:
        raise InvalidResponseError("Response content is not valid JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != set(questions):
        raise InvalidResponseError(
            "Response must contain exactly the requested question IDs"
        )

    answers = {}
    for key, question in questions.items():
        field = {"choice": "selected", "noul": "probability", "score": "level"}[
            question.type
        ]
        item = parsed[key]
        if not isinstance(item, dict) or set(item) != {field, "confidence"}:
            raise InvalidResponseError(
                f"Answer for {key!r} must contain {field!r} and 'confidence' only"
            )
        confidence = _probability(item["confidence"], "confidence")
        value = item[field]
        if isinstance(question, Noul):
            value = _probability(value, "probability")
        else:
            options = (
                question.options if isinstance(question, Choice) else question.levels
            )
            if not isinstance(value, str) or value not in options:
                raise InvalidResponseError(
                    f"Answer for {key!r} is outside the allowed options"
                )
        answers[key] = QuestionResult(key, question.type, value, confidence)
    return answers
