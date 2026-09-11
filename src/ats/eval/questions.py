"""Balanced deterministic question-set construction with paraphrases."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from ats.timeline import ActivityInterval


@dataclass(frozen=True)
class QuestionCase:
    question_type: str
    question: str
    expected: Dict[str, object]

    def to_dict(self):
        return asdict(self)


def generate_question_set(timeline: Sequence[ActivityInterval]) -> List[QuestionCase]:
    """Generate two phrasings for each supported deterministic question type."""
    if not timeline:
        return []
    activities = sorted({item.activity for item in timeline})
    primary = activities[0]
    primary_text = primary.replace("_", " ")
    matches = [item for item in timeline if item.activity == primary]
    total = sum(item.duration for item in matches)
    first = min(item.start for item in matches)
    second = activities[1] if len(activities) > 1 else primary
    second_text = second.replace("_", " ")
    second_total = sum(item.duration for item in timeline if item.activity == second)
    evidence = [[item.start, item.end] for item in matches]
    cases = [
        QuestionCase("identification", "What was the main activity?", {"timeline_required": True}),
        QuestionCase("identification", "Which activity occupied the recording most?", {"timeline_required": True}),
        QuestionCase("verification", f"Did the person perform {primary_text}?", {"answer": "Yes", "evidence": evidence}),
        QuestionCase("verification", f"Was any {primary_text} detected?", {"answer": "Yes", "evidence": evidence}),
        QuestionCase("duration", f"How long did {primary_text} last?", {"seconds": total, "evidence": evidence}),
        QuestionCase("duration", f"What was the total time spent {primary_text}?", {"seconds": total, "evidence": evidence}),
        QuestionCase("count", f"How many times did {primary_text} occur?", {"count": len(matches), "evidence": evidence}),
        QuestionCase("count", f"Count the separate {primary_text} periods.", {"count": len(matches), "evidence": evidence}),
        QuestionCase("onset", f"When did {primary_text} first begin?", {"seconds": first, "evidence": [evidence[0]]}),
        QuestionCase("onset", f"Give the first onset of {primary_text}.", {"seconds": first, "evidence": [evidence[0]]}),
        QuestionCase("comparison", f"Compare time in {primary_text} and {second_text}.", {"left_seconds": total, "right_seconds": second_total}),
        QuestionCase("comparison", f"Was {primary_text} longer than {second_text}?", {"left_seconds": total, "right_seconds": second_total}),
        QuestionCase("grounding", f"What signal evidence supports {primary_text}?", {"evidence": evidence}),
        QuestionCase("grounding", f"Cite the sensor interval for {primary_text}.", {"evidence": evidence}),
    ]
    return cases
