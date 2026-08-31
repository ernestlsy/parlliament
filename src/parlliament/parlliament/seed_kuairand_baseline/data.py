"""KuaiRand five-field baseline data loader adapted to ParLLiaMent's fixed contract."""

import csv
import os

import numpy as np


LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]


def load(data_dir, max_rows_per_split=None):
    video_to_author = {}
    with open(
        os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            video_to_author[row["video_id"]] = row["author_id"]

    rows = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append((
                    int(row["date"]),
                    row["user_id"],
                    row["video_id"],
                    video_to_author.get(row["video_id"], "UNK"),
                    row["tab"],
                    float(row["duration_ms"]),
                    1 if row[LABEL] != "0" else 0,
                ))
    result = {}
    for name, (low, high) in SPLITS.items():
        selected = [row for row in rows if low <= row[0] <= high]
        result[name] = (
            selected if max_rows_per_split is None else selected[:max_rows_per_split]
        )
    return result


def _bucket_edges(durations, bucket_count=10):
    return np.quantile(
        np.asarray(durations, dtype=np.float64),
        np.linspace(0.0, 1.0, bucket_count + 1)[1:-1],
    )


def encode(splits):
    train = splits["train"]
    if not train:
        raise ValueError("training split is empty")
    edges = _bucket_edges([row[5] for row in train])

    def raw(row):
        return [
            row[1], row[2], row[3], row[4],
            str(int(np.searchsorted(edges, row[5]))),
        ]

    vocabularies = [dict() for _ in FIELDS]
    for row in train:
        for index, value in enumerate(raw(row)):
            if value not in vocabularies[index]:
                vocabularies[index][value] = len(vocabularies[index])
    unknown = [len(vocabulary) for vocabulary in vocabularies]
    field_dimensions = [len(vocabulary) + 1 for vocabulary in vocabularies]
    offsets = np.cumsum([0] + field_dimensions[:-1]).astype(np.int32)

    encoded = {}
    for name, rows in splits.items():
        features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        labels = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype="U32")
        for row_index, row in enumerate(rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = (
                    vocabularies[field_index].get(value, unknown[field_index])
                    + offsets[field_index]
                )
            labels[row_index] = row[6]
            users[row_index] = row[1]
        encoded[name] = (features, labels, users)
    return encoded, int(sum(field_dimensions))
