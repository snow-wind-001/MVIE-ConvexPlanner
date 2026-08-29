import json
from pathlib import Path


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "forest_simulator"
    / "src"
    / "data"
    / "forest-results.json"
)


def test_checked_in_forest_evidence_is_internally_consistent():
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    margin = payload["metadata"]["safety_margin_m"]

    assert len(scenarios) == 30
    assert sum(item["metrics"]["reactive_success"] for item in scenarios) == payload[
        "overall"
    ]["reactive_success_count"]
    assert payload["overall"]["reactive_success_count"] == 30
    assert sum(payload["overall"]["avoidance_direction_counts"].values()) == payload[
        "overall"
    ]["reactive_success_count"]
    assert "branch capsules" in payload["metadata"]["collision_geometry"]

    for scenario in scenarios:
        metrics = scenario["metrics"]
        assert metrics["full_safe"]
        assert metrics["deadline_met"]
        assert scenario["reference_collision_segments"]
        assert sum(tree["dynamic"] for tree in scenario["trees"]) == 1
        assert all(len(tree["branches"]) >= 3 for tree in scenario["trees"])
        assert all(len(tree["canopy_spheres"]) >= 3 for tree in scenario["trees"])
        dynamic_tree = next(tree for tree in scenario["trees"] if tree["dynamic"])
        assert any(
            branch["route_blocker"] for branch in dynamic_tree["branches"]
        ) or dynamic_tree["radius"] >= 0.30
        if metrics["reactive_success"]:
            assert scenario["realtime_path"] is not None
            assert metrics["reactive_safe"]
            assert metrics["reactive_min_clearance_m"] >= margin - 1e-3
        else:
            assert scenario["realtime_path"] is None
            assert not metrics["reactive_safe"]
