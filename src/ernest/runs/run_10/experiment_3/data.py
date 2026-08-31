"""Data contract for additive ID and request-context categorical features."""

import csv
import datetime
import os

import numpy as np


LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]


def load(data_dir, max_rows_per_split=None):
    rows = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date_value = int(row["date"])
                hour = int(float(row["hourmin"]) // 100)
                if not 0 <= hour <= 23:
                    raise ValueError("hour derived from hourmin is out of range")
                weekday = datetime.datetime.strptime(
                    str(date_value), "%Y%m%d"
                ).weekday()
                rows.append((
                    date_value,
                    row["user_id"],
                    row["video_id"],
                    row["tab"],
                    hour,
                    weekday,
                    1 if row[LABEL] != "0" else 0,
                ))
    result = {}
    for name, (low, high) in SPLITS.items():
        selected = [row for row in rows if low <= row[0] <= high]
        result[name] = selected if max_rows_per_split is None else selected[:max_rows_per_split]
    return result


def encode(splits):
    train = splits["train"]
    if not train:
        raise ValueError("training split is empty")

    def raw(row):
        return [row[1], row[2], row[3], row[4], row[5]]

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
        labels = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype="U32")
        for row_index, row in enumerate(rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = (
                    vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                )
            labels[row_index] = row[6]
            users[row_index] = row[1]
        encoded[name] = (features, labels, users)
    return encoded, int(sum(dimensions))
