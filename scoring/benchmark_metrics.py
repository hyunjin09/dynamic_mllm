from __future__ import annotations

import re
import string
from typing import Iterable


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCTUATION = str.maketrans("", "", string.punctuation)

# TextVQA uses the EvalAI/VQA answer processor. Keep these tables local so
# candidate construction, reference scoring, and generated-answer evaluation
# cannot drift between environments.
_TEXTVQA_CONTRACTIONS = {
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
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
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
_TEXTVQA_NUMBER_MAP = {
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
_TEXTVQA_ARTICLES = {"a", "an", "the"}
_TEXTVQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_TEXTVQA_COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
_TEXTVQA_PUNCTUATIONS = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
]


def normalize_exact(text: str) -> str:
    text = str(text).lower().translate(_PUNCTUATION)
    return " ".join(text.split())


def normalize_textvqa(text: str) -> str:
    # Mirrors facebookresearch/mmf EvalAIAnswerProcessor, which TextVQA uses
    # for both answer consensus and prediction normalization.
    text = str(text).lower()
    text = text.replace(",", "").replace("?", "").replace("'s", " 's").strip()
    text = text.replace("\n", " ").replace("\t", " ").strip()
    original = text
    for punctuation in _TEXTVQA_PUNCTUATIONS:
        if (
            punctuation + " " in original
            or " " + punctuation in original
            or _TEXTVQA_COMMA_STRIP.search(original) is not None
        ):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = _TEXTVQA_PERIOD_STRIP.sub("", text)
    words = []
    for word in text.lower().split():
        word = _TEXTVQA_NUMBER_MAP.get(word, word)
        if word not in _TEXTVQA_ARTICLES:
            words.append(_TEXTVQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


def _as_number(text: str) -> float | None:
    raw = str(text).strip().lower().replace(",", "")
    raw = raw.removesuffix("%").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def relaxed_accuracy(prediction: str, answer: str) -> float:
    pred_number = _as_number(prediction)
    answer_number = _as_number(answer)
    if pred_number is not None and answer_number is not None:
        if answer_number == 0:
            return float(abs(pred_number) <= 0.05)
        return float(abs(pred_number - answer_number) / abs(answer_number) <= 0.05)
    return float(normalize_exact(prediction) == normalize_exact(answer))


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: Iterable[str]) -> float:
    prediction = normalize_exact(prediction)
    best = 0.0
    for raw_answer in answers:
        answer = normalize_exact(raw_answer)
        denominator = max(len(prediction), len(answer), 1)
        similarity = 1.0 - levenshtein_distance(prediction, answer) / denominator
        best = max(best, similarity if similarity >= 0.5 else 0.0)
    return best


def textvqa_consensus(prediction: str, answers: Iterable[str]) -> float:
    prediction = normalize_textvqa(prediction)
    matches = sum(normalize_textvqa(answer) == prediction for answer in answers)
    return min(1.0, matches / 3.0)


def score_record(record: dict, prediction: str) -> float:
    metric = record["metric_name"]
    answer = str(record["answer"])
    if metric == "exact_match_ignore_case_punctuation":
        return float(normalize_exact(prediction) == normalize_exact(answer))
    if metric == "relaxed_accuracy":
        return relaxed_accuracy(prediction, answer)
    if metric == "anls":
        answers = record.get("all_answer_norms") or [answer]
        return anls(prediction, answers)
    if metric == "textvqa_evalai_consensus":
        answers = record.get("all_answer_norms") or [answer]
        return textvqa_consensus(prediction, answers)
    raise ValueError(f"Unsupported benchmark metric: {metric}")
