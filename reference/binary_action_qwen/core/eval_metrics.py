"""Small deterministic VQA-style metrics for DVR compatibility checks."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence


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


def score_prediction(
    metric_name: str,
    prediction: str,
    answer: str,
    all_answer_norms: list[str] | None = None,
) -> float:
    metric = metric_name.lower()
    if "textvqa" in metric or "consensus" in metric:
        return textvqa_consensus(prediction, all_answer_norms, answer)
    if "relaxed" in metric:
        return relaxed_accuracy(prediction, answer)
    if "anls" in metric:
        return anls(prediction, all_answer_norms, answer)
    if re.search(r"exact|match", metric):
        return exact_match_ignore_case_punctuation(prediction, answer)
    return exact_match_ignore_case_punctuation(prediction, answer)
