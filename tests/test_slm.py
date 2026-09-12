import pytest

from ats.slm import GroundingError, build_prompt, validate_generated_answer
from ats.timeline import ActivityInterval


def _intervals():
    return [ActivityInterval("running", 4.0, 9.0, 0.9, "both", ["All"], {"acc_rms_g": 1.2})]


def test_prompt_contains_only_structured_evidence():
    prompt = build_prompt("Was activity strenuous?", _intervals())
    assert "EVIDENCE=" in prompt
    assert '"start":4.0' in prompt
    assert "raw signal" not in prompt.lower()


def test_validator_accepts_contained_timestamp():
    answer = validate_generated_answer({
        "answer": "Yes",
        "activity_event": "Strenuous activity",
        "timestamps": [[5.0, 8.0]],
        "modality": "both",
        "channels": ["All"],
        "explanation": "The supplied running interval has elevated motion energy.",
    }, _intervals())
    assert answer.timestamps == [(5.0, 8.0)]


def test_validator_rejects_invented_timestamp():
    with pytest.raises(GroundingError):
        validate_generated_answer({
            "answer": "Yes",
            "activity_event": "Strenuous activity",
            "timestamps": [[2.0, 8.0]],
            "modality": "both",
            "channels": ["All"],
            "explanation": "Unsupported time.",
        }, _intervals())


def test_validator_rejects_unsupported_modality_and_channels():
    payload = {
        "answer": "Yes",
        "activity_event": "Running",
        "timestamps": [[5.0, 8.0]],
        "modality": "accelerometer",
        "channels": ["Acc X"],
        "explanation": "Claim.",
    }
    accelerometer_only = [
        ActivityInterval("sitting", 4.0, 9.0, 0.9, "accelerometer", ["Acc Y"])
    ]
    with pytest.raises(GroundingError):
        validate_generated_answer(payload, accelerometer_only)
    payload["modality"] = "both"
    payload["channels"] = ["Acc Y"]
    with pytest.raises(GroundingError):
        validate_generated_answer(payload, accelerometer_only)


def test_validator_canonicalizes_modality_enum_case():
    payload = {
        "answer": "Yes",
        "activity_event": "Strenuous activity",
        "timestamps": [[5.0, 8.0]],
        "modality": "Both",
        "channels": ["All"],
        "explanation": "The supplied running interval supports the answer.",
    }
    answer = validate_generated_answer(payload, _intervals())
    assert answer.modality == "both"
