"""Optional Qwen slow path with deterministic grounding validation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from ats.qa import StructuredAnswer
from ats.timeline import ActivityInterval

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ALLOWED_MODALITIES = {"accelerometer", "gyroscope", "both", "N/A"}
ALLOWED_CHANNELS = {
    "Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z", "All", "N/A"
}


class GroundingError(ValueError):
    """Raised when generated content is malformed or unsupported by evidence."""


def _extract_json(text: str) -> Mapping[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise GroundingError("model output contains no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GroundingError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GroundingError("model output must be a JSON object")
    return payload


def validate_generated_answer(
    payload: Mapping[str, object], intervals: Sequence[ActivityInterval]
) -> StructuredAnswer:
    required = {"answer", "activity_event", "timestamps", "modality", "channels", "explanation"}
    missing = required - set(payload)
    if missing:
        raise GroundingError(f"missing generated fields: {sorted(missing)}")
    answer = str(payload["answer"]).strip()
    activity_event = str(payload["activity_event"]).strip() or "N/A"
    explanation = str(payload["explanation"]).strip()
    modality = str(payload["modality"])
    channels = payload["channels"]
    raw_timestamps = payload["timestamps"]
    if not answer or not explanation:
        raise GroundingError("answer and explanation must be nonempty")
    if modality not in ALLOWED_MODALITIES:
        raise GroundingError(f"unsupported modality: {modality}")
    if not isinstance(channels, list) or any(str(channel) not in ALLOWED_CHANNELS for channel in channels):
        raise GroundingError("unsupported or malformed channel list")
    if not isinstance(raw_timestamps, list):
        raise GroundingError("timestamps must be a list of [start, end] pairs")

    timestamps: List[tuple[float, float]] = []
    for pair in raw_timestamps:
        if not isinstance(pair, list) or len(pair) != 2:
            raise GroundingError("each timestamp must be [start, end]")
        start, end = float(pair[0]), float(pair[1])
        if start > end:
            raise GroundingError("timestamp start exceeds end")
        containing = [item for item in intervals if start >= item.start - 1e-6 and end <= item.end + 1e-6]
        if not containing:
            raise GroundingError(f"timestamp [{start}, {end}] is outside supplied evidence")
        if modality != "N/A" and not any(item.modality in {modality, "both"} or modality == "both" for item in containing):
            raise GroundingError("claimed modality is not supported by the cited interval")
        timestamps.append((start, end))

    if answer.lower() not in {"inconclusive", "n/a"} and not timestamps:
        raise GroundingError("a supported claim must cite at least one evidence interval")
    return StructuredAnswer(
        answer=answer,
        activity_event=activity_event,
        timestamps=timestamps,
        modality=modality,
        channels=[str(channel) for channel in channels if str(channel) != "N/A"],
        explanation=explanation,
    )


def build_prompt(question: str, intervals: Sequence[ActivityInterval]) -> str:
    evidence = [item.to_dict() for item in intervals]
    schema = {
        "answer": "direct answer or Inconclusive",
        "activity_event": "event name or N/A",
        "timestamps": [[0.0, 1.0]],
        "modality": "accelerometer|gyroscope|both|N/A",
        "channels": ["All"],
        "explanation": "reason using only supplied evidence",
    }
    return (
        "You answer wearable-sensor questions. Use only the EVIDENCE JSON. "
        "Do not invent activities, timestamps, modalities, channels, or medical conclusions. "
        "If the evidence does not support the question, answer Inconclusive. Return one JSON object "
        "with exactly the shown keys.\n"
        f"SCHEMA={json.dumps(schema, separators=(',', ':'))}\n"
        f"QUESTION={json.dumps(question)}\n"
        f"EVIDENCE={json.dumps(evidence, separators=(',', ':'))}"
    )


class GroundedQwen:
    """Lazy local Qwen runner; importing this module never downloads weights."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, local_files_only: bool = False):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Task 4 model dependencies are unavailable. Install torch, transformers, and accelerate, "
                "or run with --no-slm."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        kwargs: Dict[str, object] = {"local_files_only": self.local_files_only}
        if torch.cuda.is_available():
            kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
        else:
            kwargs.update({"torch_dtype": torch.float32})
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

    def answer(self, question: str, intervals: Sequence[ActivityInterval]) -> StructuredAnswer:
        self._load()
        prompt = build_prompt(question, intervals)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = self._model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        continuation = generated[0, inputs["input_ids"].shape[1]:]
        output = self._tokenizer.decode(continuation, skip_special_tokens=True)
        return validate_generated_answer(_extract_json(output), intervals)
