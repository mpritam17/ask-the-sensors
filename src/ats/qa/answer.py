"""Deterministic question routing and evidence-grounded answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from ats.timeline import ActivityInterval, intervals_for

DISPLAY_NAMES = {
    "lying_down": "Lying down",
    "sitting": "Sitting",
    "standing_in_place": "Standing in place",
    "standing_and_moving": "Standing and moving",
    "walking": "Walking",
    "running": "Running",
    "bicycling": "Bicycling",
}
ALIASES = {
    "lying_down": ("lying down", "lying", "lay down", "resting"),
    "sitting": ("sitting", "seated", "sit"),
    "standing_in_place": ("standing in place", "standing still", "stood still"),
    "standing_and_moving": ("standing and moving", "moving while standing"),
    "walking": ("walking", "walked", "walk"),
    "running": ("running", "ran", "run"),
    "bicycling": ("bicycling", "cycling", "cycled", "bicycle", "bike"),
}


@dataclass
class StructuredAnswer:
    answer: str
    activity_event: str = "N/A"
    timestamps: List[tuple[float, float]] = field(default_factory=list)
    modality: str = "N/A"
    channels: List[str] = field(default_factory=list)
    explanation: str = "N/A"

    def format(self) -> str:
        timestamp_text = (
            "; ".join(f"{start:.2f} to {end:.2f} seconds from start" for start, end in self.timestamps)
            if self.timestamps else "N/A"
        )
        channel_text = ", ".join(self.channels) if self.channels else "N/A"
        return "\n".join([
            f"Answer: {self.answer}",
            f"Activity/Event: {self.activity_event}",
            "Evidence:",
            f"  Timestamp(s): {timestamp_text}",
            f"  Sensor Modality: {self.modality}",
            f"  Sensor Channel(s): {channel_text}",
            f"Explanation: {self.explanation}",
        ])


def _mentioned_activities(question: str) -> List[str]:
    lowered = question.lower()
    hits = []
    for activity, aliases in ALIASES.items():
        first = min((lowered.find(alias) for alias in aliases if alias in lowered), default=-1)
        if first >= 0:
            hits.append((first, activity))
    return [activity for _, activity in sorted(hits)]


def route_question(question: str) -> tuple[str, List[str]]:
    lowered = " ".join(question.lower().split())
    activities = _mentioned_activities(lowered)
    if len(activities) >= 2 and any(token in lowered for token in ("longer", "more time", "compare", "than")):
        return "comparison", activities[:2]
    if activities and any(token in lowered for token in ("how long", "duration", "how much time", "total time")):
        return "duration", activities[:1]
    if activities and any(token in lowered for token in ("how many times", "how often", "count")):
        return "count", activities[:1]
    if activities and any(token in lowered for token in ("when", "first", "begin", "started", "onset")):
        return "onset", activities[:1]
    if activities and re.match(r"^(did|does|was|were|is|are|has|have)\b", lowered):
        return "verification", activities[:1]
    if any(token in lowered for token in ("what activity", "which activity", "main activity", "doing")):
        return "identification", activities[:1]
    return "open_world", activities


def _evidence_answer(answer: str, activity: str, matches: Sequence[ActivityInterval], explanation: str) -> StructuredAnswer:
    if not matches:
        return StructuredAnswer(answer, DISPLAY_NAMES.get(activity, activity), explanation=explanation)
    modalities = {match.modality for match in matches}
    modality = next(iter(modalities)) if len(modalities) == 1 else "both"
    channels = sorted({channel for match in matches for channel in match.channels})
    mean_confidence = sum(match.confidence for match in matches) / len(matches)
    feature_values = {}
    for match in matches:
        for name, value in match.features.items():
            feature_values.setdefault(name, []).append(float(value))
    feature_text = "; ".join(
        f"{name}={sum(values) / len(values):.3f}"
        for name, values in sorted(feature_values.items())
    )
    measured = f" Mean interval confidence={mean_confidence:.3f}."
    if feature_text:
        measured += f" Mean measured features: {feature_text}."
    return StructuredAnswer(
        answer=answer,
        activity_event=DISPLAY_NAMES.get(activity, activity),
        timestamps=[(match.start, match.end) for match in matches],
        modality=modality,
        channels=channels,
        explanation=explanation + measured,
    )


def answer_question(question: str, timeline: Sequence[ActivityInterval]) -> StructuredAnswer:
    intent, activities = route_question(question)
    if intent == "identification":
        if not timeline:
            return StructuredAnswer("Inconclusive", explanation="No supported activity interval is available.")
        totals = {}
        for interval in timeline:
            totals[interval.activity] = totals.get(interval.activity, 0.0) + interval.duration
        activity = max(totals, key=totals.get)
        matches = intervals_for(timeline, activity)
        return _evidence_answer(
            DISPLAY_NAMES[activity], activity, matches,
            f"This activity has the largest supported duration ({totals[activity]:.2f} seconds).",
        )

    if intent in {"verification", "duration", "count", "onset"} and activities:
        activity = activities[0]
        matches = intervals_for(timeline, activity)
        if intent == "verification":
            answer = "Yes" if matches else "No"
            explanation = (
                f"Found {len(matches)} supported {DISPLAY_NAMES[activity].lower()} interval(s)."
                if matches else f"No supported {DISPLAY_NAMES[activity].lower()} interval was found."
            )
            return _evidence_answer(answer, activity, matches, explanation)
        if intent == "duration":
            total = sum(match.duration for match in matches)
            return _evidence_answer(
                f"{total:.2f} seconds", activity, matches,
                f"Summed the durations of {len(matches)} supported interval(s).",
            )
        if intent == "count":
            return _evidence_answer(
                str(len(matches)), activity, matches,
                "Counted separated, gap-aware supported intervals.",
            )
        if not matches:
            return _evidence_answer("N/A", activity, [], "No supported matching interval was found.")
        first = min(matches, key=lambda item: item.start)
        return _evidence_answer(
            f"{first.start:.2f} seconds from start", activity, [first],
            "Selected the start of the first supported matching interval.",
        )

    if intent == "comparison" and len(activities) == 2:
        left, right = activities
        left_matches, right_matches = intervals_for(timeline, left), intervals_for(timeline, right)
        left_total = sum(item.duration for item in left_matches)
        right_total = sum(item.duration for item in right_matches)
        if abs(left_total - right_total) < 1e-9:
            answer = f"Equal duration ({left_total:.2f} seconds each)"
            event = f"{DISPLAY_NAMES[left]} vs {DISPLAY_NAMES[right]}"
        else:
            winner = left if left_total > right_total else right
            answer = f"{DISPLAY_NAMES[winner]} by {abs(left_total - right_total):.2f} seconds"
            event = f"{DISPLAY_NAMES[left]} vs {DISPLAY_NAMES[right]}"
        matches = [*left_matches, *right_matches]
        result = _evidence_answer(
            answer, left, matches,
            f"Compared supported totals: {DISPLAY_NAMES[left]} {left_total:.2f} seconds; "
            f"{DISPLAY_NAMES[right]} {right_total:.2f} seconds.",
        )
        result.activity_event = event
        return result

    return answer_open_world(question, timeline)


def answer_open_world(question: str, timeline: Sequence[ActivityInterval]) -> StructuredAnswer:
    lowered = question.lower()
    if any(token in lowered for token in ("prolonged rest", "long rest", "rested for a long")):
        matches = [item for item in intervals_for(timeline, "lying_down") if item.duration >= 10.0]
        return _evidence_answer(
            "Yes" if matches else "Inconclusive", "lying_down", matches,
            "A prolonged-rest claim requires a supported lying-down interval of at least 10 seconds.",
        )
    if any(token in lowered for token in ("wheeled", "pedal", "cycling", "bicycle", "bike")):
        matches = intervals_for(timeline, "bicycling")
        return _evidence_answer(
            "Yes" if matches else "Inconclusive", "bicycling", matches,
            "Wheeled movement is supported only by detected bicycling intervals.",
        )
    if any(token in lowered for token in ("strenuous", "intense", "vigorous")):
        matches = intervals_for(timeline, "running")
        return _evidence_answer(
            "Yes" if matches else "Inconclusive", "running", matches,
            "Strenuous activity is supported only when a high-intensity running interval is present.",
        )
    return StructuredAnswer(
        "Inconclusive",
        explanation="The question is unsupported by the deterministic timeline rules; no ungrounded claim was generated.",
    )
