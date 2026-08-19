"""Small deterministic VQA-style metrics for DVR compatibility checks."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation


class _EvalAIAnswerProcessor:
    """TextVQA answer normalization used by lmms-eval."""

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = {"a", "an", "the"}
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def word_tokenize(self, word: str) -> str:
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, text: str) -> str:
        output = text
        for punctuation in self.PUNCTUATIONS:
            if (punctuation + " " in text or " " + punctuation in text) or re.search(self.COMMA_STRIP, text):
                output = output.replace(punctuation, "")
            else:
                output = output.replace(punctuation, " ")
        return self.PERIOD_STRIP.sub("", output, re.UNICODE)

    def process_digit_article(self, text: str) -> str:
        output = []
        for word in text.lower().split():
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                output.append(word)
        for idx, word in enumerate(output):
            if word in self.CONTRACTIONS:
                output[idx] = self.CONTRACTIONS[word]
        return " ".join(output)

    def __call__(self, item: str | None) -> str:
        text = "" if item is None else str(item)
        text = self.word_tokenize(text)
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = self.process_punctuation(text)
        return self.process_digit_article(text)


_TEXTVQA_PROCESSOR = _EvalAIAnswerProcessor()


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _as_text(text: str | None) -> str:
    return "" if text is None else str(text)


def _chartqa_to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def exact_match_ignore_case_punctuation(prediction: str, answer: str) -> float:
    table = str.maketrans("", "", string.punctuation)
    prediction_norm = _as_text(prediction).strip().lower().translate(table)
    answer_norm = _as_text(answer).lower().translate(table)
    return float(prediction_norm == answer_norm)


def relaxed_accuracy(prediction: str, answer: str) -> float:
    prediction_text = _as_text(prediction).strip()
    answer_text = _as_text(answer)
    pred_num = _chartqa_to_float(prediction_text)
    answer_num = _chartqa_to_float(answer_text)
    if pred_num is not None and answer_num:
        relative_change = abs(pred_num - answer_num) / abs(answer_num)
        return float(relative_change <= 0.05)
    return float(prediction_text.lower() == answer_text.lower())


def _answer_refs(answers: Sequence[str] | None, fallback_answer: str) -> list[str]:
    if isinstance(answers, str):
        return [answers]
    if answers:
        return [str(answer) for answer in answers]
    return [_as_text(fallback_answer)]


def textvqa_consensus(prediction: str, answers: Sequence[str] | None, fallback_answer: str) -> float:
    refs = [_TEXTVQA_PROCESSOR(answer) for answer in _answer_refs(answers, fallback_answer)]
    if not refs:
        return 0.0
    pred = _TEXTVQA_PROCESSOR(_as_text(prediction).strip())
    scores = []
    for idx in range(len(refs)):
        other_refs = refs[:idx] + refs[idx + 1 :]
        matches = sum(answer == pred for answer in other_refs)
        scores.append(min(1.0, float(matches) / 3.0))
    return sum(scores) / len(scores)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + int(ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def anls(prediction: str, answers: list[str] | None, fallback_answer: str) -> float:
    refs = _answer_refs(answers, fallback_answer)
    prediction_text = _as_text(prediction).strip()
    pred = " ".join(prediction_text.lower().split())
    distances = []
    for answer in refs:
        ref = " ".join(_as_text(answer).strip().lower().split())
        denom = max(len(_as_text(answer).upper()), len(prediction_text.upper()))
        distances.append(0.0 if denom == 0 else float(_levenshtein(ref, pred)) / float(denom))
    score = 1.0 - min(distances)
    return score if score >= 0.5 else 0.0


def pope_yes_no_accuracy(prediction: str, answer: str) -> float:
    pred = _as_text(prediction).strip().lower()
    if pred.startswith("yes"):
        pred = "yes"
    elif pred.startswith("no"):
        pred = "no"
    gt = _as_text(answer).strip().lower()
    return float(pred == gt and gt in {"yes", "no"})


def seed_choice_accuracy(prediction: str, answer: str) -> float:
    pred = _as_text(prediction).strip()
    pred = pred[0].upper() if pred else ""
    gt = _as_text(answer).strip().upper()
    return float(pred == gt and gt in {"A", "B", "C", "D"})


def multiple_choice_accuracy(prediction: str, answer: str) -> float:
    """Score the first standalone option letter for MMMU-family outputs."""
    pred_match = re.search(r"\b([A-J])\b", _as_text(prediction).upper())
    pred = pred_match.group(1) if pred_match else ""
    gt = _as_text(answer).strip().upper()
    return float(pred == gt and bool(re.fullmatch(r"[A-J]", gt)))


_DIRECT_ANSWER_PREFIX = re.compile(
    r"^(?:final\s+answer|short\s+answer|answer)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
_NUMERIC_SEQUENCE = re.compile(
    r"^[\s,;:/\[\](){}+-]*(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?)"
    r"(?:[\s,;:/\[\](){}+-]+[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?)*"
    r"[\s,;:/\[\](){}+-]*$"
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_direct_answer(text: str | None) -> str:
    """Extract a conservative final-answer span from short model output."""
    value = _as_text(text).strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", value)
    if boxed:
        value = boxed[-1]
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) > 1:
        prefixed = [line for line in lines if _DIRECT_ANSWER_PREFIX.match(line)]
        value = prefixed[-1] if prefixed else lines[-1]
    return _DIRECT_ANSWER_PREFIX.sub("", value).strip()


def _canonical_reasoning_answer(text: str | None) -> str:
    value = _extract_direct_answer(text).lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\times": "*",
        "\\cdot": "*",
        "×": "*",
        "−": "-",
        "$": "",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", value)
    value = value.strip().rstrip(".。")
    return re.sub(r"\s+", "", value)


def _numeric_values(text: str | None) -> list[Decimal] | None:
    value = _extract_direct_answer(text).lower().strip().rstrip(".。")
    value = value.replace("−", "-").replace("\\,", " ")
    value = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", value)
    if not value or not _NUMERIC_SEQUENCE.fullmatch(value):
        return None
    values: list[Decimal] = []
    for match in _NUMBER.finditer(value):
        try:
            number = Decimal(match.group(0))
        except InvalidOperation:
            return None
        suffix = value[match.end() : match.end() + 1]
        values.append(number / Decimal(100) if suffix == "%" else number)
    return values or None


def reasoning_strict_accuracy(prediction: str, answer: str) -> float:
    """Strict deterministic matching for short visual-reasoning answers.

    Numeric lists receive only a tiny serialization tolerance. Algebraic or
    textual answers require exact equality after conservative formatting
    normalization; semantic substring matching is deliberately not used.
    """
    pred_numbers = _numeric_values(prediction)
    answer_numbers = _numeric_values(answer)
    if pred_numbers is not None and answer_numbers is not None:
        if len(pred_numbers) != len(answer_numbers):
            return 0.0
        tolerance = Decimal("0.0001")
        return float(all(abs(pred - ref) <= tolerance for pred, ref in zip(pred_numbers, answer_numbers)))
    return float(_canonical_reasoning_answer(prediction) == _canonical_reasoning_answer(answer))


def dynamath_float_accuracy(prediction: str, answer: str) -> float:
    """DynaMath's published absolute tolerance for floating-point answers."""
    pred_match = _NUMBER.search(_extract_direct_answer(prediction).replace(",", ""))
    answer_match = _NUMBER.search(_extract_direct_answer(answer).replace(",", ""))
    if pred_match is None or answer_match is None:
        return 0.0
    try:
        pred = Decimal(pred_match.group(0))
        ref = Decimal(answer_match.group(0))
    except InvalidOperation:
        return 0.0
    return float(abs(pred - ref) <= Decimal("0.001"))


def wemath2pro_mathruler_accuracy(prediction: str, answer: str) -> float:
    """Score We-Math2.0-Pro using the benchmark's MathRuler contract.

    The official R1-V reward extracts an ``<answer>`` span when present and
    otherwise passes the stripped response directly to ``grade_answer``.
    Missing MathRuler support is a hard error: falling back to string exact
    match would change the route-label reward.
    """
    try:
        from mathruler.grader import grade_answer
    except ImportError as exc:  # pragma: no cover - environment contract failure
        raise RuntimeError(
            "wemath2pro_mathruler_accuracy requires mathruler==0.1.0"
        ) from exc

    content_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    given_answer = content_match.group(1).strip() if content_match else prediction.strip()
    try:
        return float(bool(grade_answer(given_answer, answer.strip())))
    except Exception:
        return 0.0


def score_prediction(
    metric_name: str,
    prediction: str,
    answer: str,
    all_answer_norms: list[str] | None = None,
) -> float:
    metric = metric_name.lower()
    if metric == "wemath2pro_mathruler_accuracy":
        return wemath2pro_mathruler_accuracy(prediction, answer)
    if metric == "dynamath_float_accuracy":
        return dynamath_float_accuracy(prediction, answer)
    if metric in {"dynamath_multiple_choice_accuracy", "reasoning_multiple_choice_accuracy"}:
        return multiple_choice_accuracy(prediction, answer)
    if metric in {"dynamath_text_accuracy", "reasoning_strict_accuracy"}:
        return reasoning_strict_accuracy(prediction, answer)
    if "pope" in metric or "yes_no" in metric:
        return pope_yes_no_accuracy(prediction, answer)
    if "seed" in metric or "choice" in metric:
        if "mmstar" in metric or "mmmu" in metric:
            return multiple_choice_accuracy(prediction, answer)
        return seed_choice_accuracy(prediction, answer)
    if "mmmu" in metric:
        return multiple_choice_accuracy(prediction, answer)
    if "textvqa" in metric or "consensus" in metric:
        return textvqa_consensus(prediction, all_answer_norms, answer)
    if "relaxed" in metric:
        return relaxed_accuracy(prediction, answer)
    if "anls" in metric:
        return anls(prediction, all_answer_norms, answer)
    if re.search(r"exact|match", metric):
        return exact_match_ignore_case_punctuation(prediction, answer)
    return exact_match_ignore_case_punctuation(prediction, answer)
