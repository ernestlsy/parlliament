import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_recsys.diagnostics import analyze_prediction_artifact
from agentic_recsys.journal import Journal
from agentic_recsys.research import ResearchEngine, ScreeningConfig


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_dataset(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    header = [
        "user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like",
        "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms",
        "duration_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
        "is_rand", "tab",
    ]
    training = []
    for date_index, date in enumerate((20220408, 20220409, 20220410, 20220411)):
        for row_index in range(4):
            user = row_index % 2
            video = (row_index + date_index) % 3
            label = (row_index + date_index) % 2
            training.append([
                user, video, date, 900 + row_index * 100, date * 100 + row_index,
                label, 0, 0, 0, 0, 0, label, 1000, 10000 + video * 1000,
                0, 0, 0, 0, str(row_index % 2),
            ])
    validation = [
        [0, 0, 20220422, 900, 1, 1, 0, 0, 0, 0, 0, 1, 1000, 10000, 0, 0, 0, 0, "0"],
        [0, 1, 20220422, 1000, 2, 0, 0, 0, 0, 0, 0, 0, 1000, 11000, 0, 0, 0, 0, "0"],
        [1, 1, 20220422, 1100, 3, 1, 0, 0, 0, 0, 0, 1, 1000, 11000, 0, 0, 0, 0, "1"],
        [1, 2, 20220422, 1200, 4, 0, 0, 0, 0, 0, 0, 0, 1000, 12000, 0, 0, 0, 0, "1"],
    ]
    write_csv(data / "log_standard_4_08_to_4_21_pure.csv", header, training)
    write_csv(data / "log_standard_4_22_to_5_08_pure.csv", header, validation)
    write_csv(
        data / "video_features_basic_pure.csv",
        [
            "video_id", "author_id", "video_type", "upload_dt", "upload_type",
            "video_duration", "music_id", "music_type", "tag",
        ],
        [
            [0, 10, "NORMAL", "2022-04-01", "Web", 10000, 100, 1, "a"],
            [1, 11, "NORMAL", "2022-04-02", "Web", 11000, 101, 1, "b"],
            [2, 10, "NORMAL", "2022-04-03", "App", 12000, 102, 2, "c"],
        ],
    )
    write_csv(
        data / "user_features_pure.csv",
        [
            "user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer",
            "is_video_author", "follow_user_num_range", "fans_user_num_range",
            "friend_user_num_range", "register_days_range",
        ],
        [
            [0, "full_active", 0, 0, 1, "[10,50)", "[10,50)", "[1,5)", "365+"],
            [1, "middle_active", 0, 0, 0, "[1,10)", "[1,10)", "[1,5)", "90+"],
        ],
    )
    return data


class ResearchEngineIntegrationTests(unittest.TestCase):
    def test_historical_development_features_exclude_current_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = ResearchEngine(root, root / "run", ScreeningConfig())
            development = pd.DataFrame({
                "date": [20220408, 20220408],
                "time_ms": [1, 2],
                "user_id": [1, 1],
                "video_id": [7, 7],
                "author_id": [9, 9],
                "long_view": [1, 0],
            })
            holdout = pd.DataFrame({
                "date": [20220409], "time_ms": [3], "user_id": [1],
                "video_id": [7], "author_id": [9], "long_view": [0],
            })
            enriched, _ = engine._add_train_only_history(development, holdout)
            self.assertEqual(enriched.loc[0, "prior_item_impressions"], 0)
            self.assertEqual(enriched.loc[0, "prior_item_long_rate"], 0.5)
            self.assertEqual(enriched.loc[1, "prior_item_impressions"], 1)

    def test_screening_is_train_only_cached_and_uncounted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(root)
            run = root / "run"
            journal = Journal(run / "journal.jsonl")
            engine = ResearchEngine(
                data,
                run,
                ScreeningConfig(
                    holdout_fraction=0.25,
                    timeout_seconds=60,
                    hash_dimensions=128,
                    proxy_epochs=2,
                ),
            )
            report = engine.ensure()
            self.assertEqual(report["development_dates"], [20220408, 20220409, 20220410])
            self.assertEqual(report["holdout_dates"], [20220411])
            self.assertTrue(report["candidates"])
            self.assertEqual(journal.records(), [])
            manifest_before = engine.manifest_path.read_text(encoding="utf-8")
            self.assertFalse(json.loads(manifest_before)["official_validation_accessed"])
            cached = engine.ensure()
            self.assertEqual(cached, report)
            self.assertEqual(engine.manifest_path.read_text(encoding="utf-8"), manifest_before)
            self.assertNotIn(
                "feature:log_standard_4_08_to_4_21_pure.csv:play_time_ms",
                engine.evidence_ids(),
            )

    def test_post_score_segment_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(root)
            artifact = root / "predictions_valid.npz"
            np.savez(
                artifact,
                row_ids=np.arange(4, dtype=np.int64),
                scores=np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64),
            )
            result = analyze_prediction_artifact(artifact, data)
            self.assertEqual(result["scope"], "post_score_official_validation_diagnostics")
            self.assertEqual(result["global"]["primary"], 1.0)
            self.assertIn("user_frequency", result["segments"])
            self.assertIn("duration_bucket", result["segments"])


if __name__ == "__main__":
    unittest.main()
