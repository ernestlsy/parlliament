"""Neutral additive data contract with impression-time request context features."""

import csv
import os
from datetime import datetime

import numpy as np


LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]


def load(data_dir, max_rows_per_split=None):
    result = {name: [] for name in SPLITS}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                split_name = next(
                    (
                        name
                        for name, (low, high) in SPLITS.items()
                        if low <= date <= high
                    ),
                    None,
                )
                if split_name is None:
                    continue
                base = (
                    date,
                    row["user_id"],
                    row["video_id"],
                    row["hourmin"],
                    row["tab"],
                )
                if split_name == "test":
                    result[split_name].append(base)
                else:
                    result[split_name].append(base + (1 if row[LABEL] != "0" else 0,))

    if max_rows_per_split is not None:
        for name in result:
            result[name] = result[name][:max_rows_per_split]
    return result


def encode(splits):
    train = splits["train"]
    if not train:
        raise ValueError("training split is empty")

    def raw(row):
        hour = int(float(row[3])) // 100
        hour = max(0, min(23, hour))
        weekday = datetime.strptime(str(int(row[0])), "%Y%m%d").weekday()
        return [row[1], row[2], row[4], hour, weekday]

    vocabs = [dict() for _ in FIELDS]
    for row in train:
        for index, value in enumerate(raw(row)):
            if value not in vocabs[index]:
                vocabs[index][value] = len(vocabs[index])

    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
    encoded = {}

    for name, rows in splits.items():
        features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        users = np.empty(len(rows), dtype="U32")
        for row_index, row in enumerate(rows):
            values = raw(row)
            for field_index, value in enumerate(values):
                features[row_index, field_index] = (
                    vocabs[field_index].get(value, unknown[field_index])
                    + offsets[field_index]
                )
            users[row_index] = row[1]

        if name == "test":
            encoded[name] = (features, users)
        else:
            labels = np.asarray(
                [row[5] for row in rows], dtype=np.float32
            )
            encoded[name] = (features, labels, users)

    return encoded, int(sum(dimensions))
