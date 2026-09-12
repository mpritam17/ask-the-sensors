from ats.qa import answer_question, route_question
from ats.timeline import ActivityInterval
import pytest


def _timeline():
    return [
        ActivityInterval("walking", 2.0, 8.0, 0.8, "both", ["All"]),
        ActivityInterval("sitting", 10.0, 16.0, 0.9, "accelerometer", ["Acc X", "Acc Y", "Acc Z"]),
        ActivityInterval("walking", 20.0, 24.0, 0.7, "both", ["All"]),
        ActivityInterval("running", 30.0, 34.0, 0.9, "both", ["All"]),
    ]


def test_router_and_exact_format():
    assert route_question("How long did the person walk?")[0] == "duration"
    result = answer_question("How long did the person walk?", _timeline())
    assert result.answer == "10.00 seconds"
    lines = result.format().splitlines()
    assert lines[0].startswith("Answer:")
    assert lines[1].startswith("Activity/Event:")
    assert lines[2] == "Evidence:"
    assert lines[-1].startswith("Explanation:")
    assert "confidence=" in lines[-1]


def test_count_onset_comparison_and_grounding():
    timeline = _timeline()
    assert answer_question("How many times did they walk?", timeline).answer == "2"
    assert answer_question("When did running begin?", timeline).answer == "30.00 seconds from start"
    comparison = answer_question("Did they spend longer walking than sitting?", timeline)
    assert comparison.answer == "Walking by 4.00 seconds"
    assert comparison.timestamps


def test_unsupported_open_world_question_is_inconclusive():
    result = answer_question("Did the person fall?", _timeline())
    assert result.answer == "Inconclusive"
    assert result.timestamps == []


@pytest.mark.parametrize(("question", "intent"), [
    ("Which activity occupied the most time?", "identification"),
    ("Has the person done walking?", "verification"),
    ("What was the total time spent walking?", "duration"),
    ("Count the walking intervals.", "count"),
    ("What was the onset of walking?", "onset"),
    ("Compare the duration of walking and sitting.", "comparison"),
    ("Did the person do anything vigorous?", "open_world"),
])
def test_evaluation_paraphrases_route_to_expected_intent(question, intent):
    assert route_question(question)[0] == intent
