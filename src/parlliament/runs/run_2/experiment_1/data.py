"""Data loading and train-fitted categorical encoding for additive ID/context scoring."""

import csv
import math
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
    rows = []
    test_low, test_high = SPLITS["test"]
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_date = row["date"]
                date_value = int(raw_date)
                calendar_date = datetime.strptime(raw_date, "%Y%m%d").date()

                hourmin = float(row["hourmin"])
                if not math.isfinite(hourmin):
                    raise ValueError("hourmin must be finite")
                hour = math.floor(hourmin / 100)
                if not 0 <= hour <= 23:
                    raise ValueError("derived hour must be in 0..23")
                weekday = calendar_date.weekday()

                values = (
                    date_value,
                    row["user_id"],
                    row["video_id"],
                    row["tab"],
                    hour,
                    weekday,
                )
                if test_low <= date_value <= test_high:
                    rows.append(values)
                else:
                    rows.append(values + (1 if row[LABEL] != "0" else 0,))

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
        users = np.empty(len(rows), dtype="U32")
        for row_index, row in enumerate(rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = (
                    vocabs[field_index].get(value, unknown[field_index])
                    + offsets[field_index]
                )
            users[row_index] = row[1]

        if name == "test":
            encoded[name] = (features, users)
        else:
            labels = np.empty(len(rows), dtype=np.float32)
            for row_index, row in enumerate(rows):
                labels[row_index] = row[6]
            encoded[name] = (features, labels, users)

    return encoded, int(sum(dimensions))
