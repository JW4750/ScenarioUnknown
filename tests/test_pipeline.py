from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from highd_scenario_mining import run


def _make_sample_recordings(data_dir: Path) -> None:
    frames = list(range(1, 11))
    rows = []
    for frame in frames:
        rows.append(
            {
                "id": 1,
                "frame": frame,
                "x": frame * 1.5,
                "y": 0.0,
                "laneId": 2,
                "xVelocity": 30.0,
                "xAcceleration": 0.0,
                "precedingId": 2,
                "followingId": 0,
                "leftPrecedingId": 0,
                "rightPrecedingId": 0,
                "leftFollowingId": 0,
                "rightFollowingId": 0,
                "leftAlongsideId": 0,
                "rightAlongsideId": 0,
                "drivingDirection": 2,
                "class": "Car",
                "ttc": 2.5,
                "thw": 1.8,
                "dhw": 6.0,
            }
        )
    for frame in frames:
        rows.append(
            {
                "id": 2,
                "frame": frame,
                "x": frame * 1.5 + 20.0,
                "y": 0.0,
                "laneId": 2,
                "xVelocity": 31.0,
                "xAcceleration": 0.0,
                "precedingId": 0,
                "followingId": 1,
                "leftPrecedingId": 0,
                "rightPrecedingId": 0,
                "leftFollowingId": 0,
                "rightFollowingId": 0,
                "leftAlongsideId": 0,
                "rightAlongsideId": 0,
                "drivingDirection": 2,
                "class": "Car",
                "ttc": 0.0,
                "thw": 0.0,
                "dhw": 0.0,
            }
        )
    tracks = pd.DataFrame(rows)
    tracks.to_csv(data_dir / "sample_tracks.csv", index=False)

    tracks_meta = pd.DataFrame(
        {
            "id": [1, 2],
            "traveledDistance": [300.0, 320.0],
        }
    )
    tracks_meta.to_csv(data_dir / "sample_tracksMeta.csv", index=False)

    recording_meta = pd.DataFrame({"frameRate": [25.0]})
    recording_meta.to_csv(data_dir / "sample_recordingMeta.csv", index=False)


def test_run_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_sample_recordings(data_dir)

    outputs = run(prefix="sample", data_dir=data_dir, output_dir=tmp_path)

    expected = {
        "tags": "highd_tags.csv",
        "scenarios": "highd_scenarios.csv",
        "counts": "highd_scenario_counts.csv",
        "unknown_events": "highd_unknown_danger_events.csv",
        "unknown_stats": "highd_unknown_danger_stats.csv",
        "coverage": "highd_coverage_metrics.csv",
        "report": "highd_report.html",
    }
    for key, filename in expected.items():
        path = outputs[key]
        assert path.name == filename
        assert path.exists()

    counts = pd.read_csv(outputs["counts"])
    assert "scenario_code" in counts.columns
    assert (counts["scenario_code"] == "C1_leading_cruising").any()

    coverage = pd.read_csv(outputs["coverage"])
    assert "metric" in coverage.columns
