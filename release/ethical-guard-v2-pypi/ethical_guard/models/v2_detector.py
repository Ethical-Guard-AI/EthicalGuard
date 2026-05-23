from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


LABELS = ["SAFE", "01", "02", "03", "04"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

POLICY_KEYWORDS = {
    "01": (
        "explosive",
        "weapon",
        "attack",
        "poison",
        "physical harm",
        "violent",
        "injure",
        "assault",
    ),
    "02": (
        "jailbreak",
        "bypass",
        "override",
        "ignore hidden",
        "hidden safety",
        "system prompt",
        "hidden system",
        "guardrails",
        "safety controls",
        "evade",
    ),
    "03": (
        "hateful",
        "hate speech",
        "discriminatory",
        "inferior",
        "protected group",
        "slurs",
        "harassment",
    ),
    "04": (
        "ssn",
        "social security",
        "api key",
        "api keys",
        "password",
        "tokens",
        "confidential data",
        "private personal",
        "employee data",
        "financial identifiers",
        "exfiltrate",
        "restricted keys",
    ),
}

MALICIOUS_INTENT_MARKERS = (
    "step-by-step",
    "exactly how",
    "force",
    "coerce",
    "tactics",
    "without authorization",
    "disclose",
    "reveal",
    "output",
    "evade",
    "stop refusing",
    "without being blocked",
    "without being detected",
)

DEFENSIVE_CONTEXT_MARKERS = (
    "how to defend",
    "best practices",
    "at a high level",
    "safely",
    "secure coding",
    "policy note",
    "checklist",
    "guide",
    "training",
    "onboarding",
    "changelog",
    "documentation",
    "rotate",
    "rotation",
    "disclose",
    "responsibly",
    "harmless",
)


@dataclass(frozen=True)
class Verdict:
    is_safe: bool
    policy_id: str
    reason: str
    confidence: float


def _get_field(sample, field_name: str):
    if hasattr(sample, field_name):
        return getattr(sample, field_name)
    return sample[field_name]


class HashedTextVectorizer:
    def __init__(self, dimension: int = 768, word_ngrams: Sequence[int] = (1, 2), char_ngrams: Sequence[int] = (3, 4, 5)):
        self.dimension = dimension
        self.word_ngrams = tuple(word_ngrams)
        self.char_ngrams = tuple(char_ngrams)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def _normalize(self, text: str) -> str:
        lowered = text.lower().strip()
        return re.sub(r"\s+", " ", lowered)

    def transform(self, text: str) -> Dict[int, float]:
        normalized = self._normalize(text)
        features: Dict[int, float] = {}

        words = re.findall(r"[a-z0-9']+", normalized)
        for n in self.word_ngrams:
            if len(words) < n:
                continue
            for start in range(len(words) - n + 1):
                token = "w|{}|{}".format(n, " ".join(words[start : start + n]))
                index = self._index(token)
                features[index] = features.get(index, 0.0) + 1.0

        padded = f"^{normalized}$"
        for n in self.char_ngrams:
            if len(padded) < n:
                continue
            for start in range(len(padded) - n + 1):
                token = f"c|{n}|{padded[start : start + n]}"
                index = self._index(token)
                features[index] = features.get(index, 0.0) + 1.0

        # Keep raw occurrence counts to preserve stronger gradient signal on small datasets.
        for index in list(features.keys()):
            features[index] = math.log1p(features[index])

        return features


class V2JailbreakDetector:
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 32,
        seed: int = 13,
        labels: Sequence[str] = LABELS,
        safe_confidence_threshold: float = 0.55,
        harmful_probability_threshold: float = 0.7,
    ):
        self.vectorizer = HashedTextVectorizer(dimension=input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.labels = list(labels)
        self.output_dim = len(self.labels)
        self.safe_confidence_threshold = safe_confidence_threshold
        self.harmful_probability_threshold = harmful_probability_threshold

        random.seed(seed)
        self.W1 = [[random.uniform(-0.05, 0.05) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.W2 = [[random.uniform(-0.05, 0.05) for _ in range(self.output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(self.output_dim)]

    @staticmethod
    def _relu(value: float) -> float:
        return value if value > 0.0 else 0.0

    @staticmethod
    def _softmax(logits: Sequence[float]) -> List[float]:
        max_logit = max(logits)
        exps = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def _forward(self, features: Dict[int, float]):
        hidden_raw = self.b1[:]
        for feature_index, feature_value in features.items():
            row = self.W1[feature_index]
            for hidden_index in range(self.hidden_dim):
                hidden_raw[hidden_index] += feature_value * row[hidden_index]

        hidden = [self._relu(value) for value in hidden_raw]

        logits = self.b2[:]
        for hidden_index, hidden_value in enumerate(hidden):
            if hidden_value == 0.0:
                continue
            row = self.W2[hidden_index]
            for output_index in range(self.output_dim):
                logits[output_index] += hidden_value * row[output_index]

        return hidden_raw, hidden, logits

    def _label_index(self, label: str) -> int:
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown label: {label}")
        return LABEL_TO_INDEX[label]

    def train(
        self,
        samples: Sequence,
        epochs: int = 18,
        learning_rate: float = 0.08,
        validation_split: float = 0.2,
        seed: int = 13,
        shuffle: bool = True,
    ):
        indexed = list(samples)
        random.Random(seed).shuffle(indexed)

        split_index = max(1, int(len(indexed) * (1.0 - validation_split)))
        train_samples = indexed[:split_index]
        validation_samples = indexed[split_index:] or indexed[: max(1, len(indexed) // 5)]

        history = []
        rng = random.Random(seed)

        for epoch in range(epochs):
            if shuffle:
                rng.shuffle(train_samples)

            total_loss = 0.0
            correct = 0

            for sample in train_samples:
                prompt = _get_field(sample, "prompt")
                label = _get_field(sample, "label")
                target_index = self._label_index(label)

                features = self.vectorizer.transform(prompt)
                hidden_raw, hidden, logits = self._forward(features)
                probabilities = self._softmax(logits)

                total_loss += -math.log(max(probabilities[target_index], 1e-12))
                if probabilities.index(max(probabilities)) == target_index:
                    correct += 1

                grad_logits = probabilities[:]
                grad_logits[target_index] -= 1.0

                grad_hidden = [0.0 for _ in range(self.hidden_dim)]
                for hidden_index in range(self.hidden_dim):
                    if hidden_raw[hidden_index] <= 0.0:
                        continue
                    accumulator = 0.0
                    row = self.W2[hidden_index]
                    for output_index in range(self.output_dim):
                        accumulator += row[output_index] * grad_logits[output_index]
                    grad_hidden[hidden_index] = accumulator

                for output_index in range(self.output_dim):
                    self.b2[output_index] -= learning_rate * grad_logits[output_index]

                for hidden_index in range(self.hidden_dim):
                    hidden_value = hidden[hidden_index]
                    if hidden_value == 0.0:
                        continue
                    row = self.W2[hidden_index]
                    for output_index in range(self.output_dim):
                        row[output_index] -= learning_rate * hidden_value * grad_logits[output_index]

                for hidden_index in range(self.hidden_dim):
                    if hidden_raw[hidden_index] <= 0.0:
                        continue
                    gradient = grad_hidden[hidden_index]
                    self.b1[hidden_index] -= learning_rate * gradient
                    for feature_index, feature_value in features.items():
                        self.W1[feature_index][hidden_index] -= learning_rate * feature_value * gradient

            validation_accuracy = self.evaluate_accuracy(validation_samples)
            train_accuracy = correct / max(1, len(train_samples))
            average_loss = total_loss / max(1, len(train_samples))
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(average_loss, 4),
                    "train_accuracy": round(train_accuracy, 4),
                    "validation_accuracy": round(validation_accuracy, 4),
                }
            )

        return history

    def predict_proba(self, text: str) -> Dict[str, float]:
        features = self.vectorizer.transform(text)
        _, _, logits = self._forward(features)
        probabilities = self._softmax(logits)
        return {label: probabilities[index] for index, label in enumerate(self.labels)}

    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)

    def _lexical_policy_guess(self, normalized_text: str) -> str | None:
        # Prefer explicit harmful categories first, then general jailbreak semantics.
        for policy in ("02", "04", "01", "03"):
            keywords = POLICY_KEYWORDS[policy]
            if any(keyword in normalized_text for keyword in keywords):
                return policy

        if "ignore policy" in normalized_text or "policy checks" in normalized_text:
            return "02"
        if "hidden developer instructions" in normalized_text or "system instruction" in normalized_text:
            return "02"
        if "overrid" in normalized_text and "safeguard" in normalized_text:
            return "02"
        return None

    def _is_defensive_context(self, normalized_text: str) -> bool:
        return any(marker in normalized_text for marker in DEFENSIVE_CONTEXT_MARKERS)

    def classify(self, text: str) -> Verdict:
        normalized = self.vectorizer._normalize(text)
        lexical_policy = self._lexical_policy_guess(normalized)
        has_malicious_intent = any(marker in normalized for marker in MALICIOUS_INTENT_MARKERS)
        has_defensive_context = self._is_defensive_context(normalized)

        # Allow clearly defensive educational prompts before risky lexical checks.
        if has_defensive_context and not has_malicious_intent:
            if not any(token in normalized for token in ("bypass", "override", "evade", "ignore hidden")):
                return Verdict(True, "None", "Detected defensive/educational context around sensitive topic.", 0.84)

        # Rule layer improves stability on indirect jailbreak semantics.
        if lexical_policy is not None:
            if has_defensive_context and not has_malicious_intent:
                return Verdict(True, "None", "Detected defensive/educational context around sensitive topic.", 0.82)
            return Verdict(False, lexical_policy, f"Lexical policy trigger matched {lexical_policy} risk patterns.", 0.9)

        probabilities = self.predict_proba(text)
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        harmful_probability = 1.0 - probabilities.get("SAFE", 0.0)

        if label == "SAFE" and confidence >= self.safe_confidence_threshold and harmful_probability < self.harmful_probability_threshold:
            return Verdict(True, "None", "Prompt assessed as benign or policy-compliant.", confidence)

        if label == "SAFE" and harmful_probability >= self.harmful_probability_threshold:
            harmful_label = max((l for l in self.labels if l != "SAFE"), key=lambda l: probabilities[l])
            return Verdict(False, harmful_label, "Harmful-probability threshold triggered fail-closed classification.", harmful_probability)

        if label == "SAFE":
            return Verdict(False, "02", "Low-confidence SAFE verdict. Fail-closed to potential jailbreak.", confidence)
        return Verdict(False, label, f"Prompt matched {label} jailbreak behavior.", confidence)

    def evaluate_accuracy(self, samples: Sequence) -> float:
        if not samples:
            return 0.0
        correct = 0
        for sample in samples:
            prompt = _get_field(sample, "prompt")
            label = _get_field(sample, "label")
            if self.predict(prompt) == label:
                correct += 1
        return correct / len(samples)

    def evaluate_metrics(self, samples: Sequence) -> Dict[str, object]:
        if not samples:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "support": 0,
                "per_label": {},
            }

        confusion = {
            label: {other: 0 for other in self.labels}
            for label in self.labels
        }

        for sample in samples:
            truth = _get_field(sample, "label")
            pred = self.predict(_get_field(sample, "prompt"))
            if truth not in confusion:
                continue
            confusion[truth][pred] += 1

        total = 0
        correct = 0
        per_label: Dict[str, Dict[str, float]] = {}
        f1_values = []

        for label in self.labels:
            tp = confusion[label][label]
            fp = sum(confusion[other][label] for other in self.labels if other != label)
            fn = sum(confusion[label][other] for other in self.labels if other != label)
            support = sum(confusion[label].values())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 0.0
            if precision + recall > 0.0:
                f1 = 2.0 * precision * recall / (precision + recall)

            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            if support > 0:
                f1_values.append(f1)

            total += support
            correct += tp

        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        accuracy = correct / total if total else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "support": total,
            "per_label": per_label,
            "confusion_matrix": confusion,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "labels": self.labels,
            "vectorizer": {
                "dimension": self.vectorizer.dimension,
                "word_ngrams": list(self.vectorizer.word_ngrams),
                "char_ngrams": list(self.vectorizer.char_ngrams),
            },
            "safe_confidence_threshold": self.safe_confidence_threshold,
            "harmful_probability_threshold": self.harmful_probability_threshold,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "V2JailbreakDetector":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        model = cls(
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            labels=payload["labels"],
            safe_confidence_threshold=payload.get("safe_confidence_threshold", 0.65),
            harmful_probability_threshold=payload.get("harmful_probability_threshold", 0.45),
        )
        model.vectorizer = HashedTextVectorizer(
            dimension=payload["vectorizer"]["dimension"],
            word_ngrams=payload["vectorizer"]["word_ngrams"],
            char_ngrams=payload["vectorizer"]["char_ngrams"],
        )
        model.W1 = payload["W1"]
        model.b1 = payload["b1"]
        model.W2 = payload["W2"]
        model.b2 = payload["b2"]
        return model
