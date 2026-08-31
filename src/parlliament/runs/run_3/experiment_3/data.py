"""Seed data contract with training-only categorical request-context vocabularies."""

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
BASE_FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]
FIELDS = BASE_FIELDS + ["hashed_video_id_tab"]
INTERACTION_DIMENSION = 262144
_FNV_OFFSET_BASIS_64 = 14695981039346656037
_FNV_PRIME_64 = 1099511628211
_FNV_MASK_64 = (1 << 64) - 1


def _hour_from_hourmin(hourmin):
    hour = math.floor(float(hourmin) / 100)
    if not 0 <= hour <= 23:
        raise ValueError("derived hour must be in 0..23")
    return hour


def _weekday_from_date(date):
    return datetime.strptime(str(date), "%Y%m%d").weekday()


def _video_tab_bucket(video_id, tab):
    value = (str(video_id) + "\x1f" + str(tab)).encode("utf-8")
    hashed = _FNV_OFFSET_BASIS_64
    for byte in value:
        hashed ^= byte
        hashed = (hashed * _FNV_PRIME_64) & _FNV_MASK_64
    return hashed % INTERACTION_DIMENSION


def raw(row):
    return [
        row[1],
        row[2],
        row[3],
        row[4],
        row[5],
        _video_tab_bucket(row[2], row[3]),
    ]


def load(data_dir, max_rows_per_split=None):
    rows = []
    test_low, test_high = SPLITS["test"]
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                hour = _hour_from_hourmin(row["hourmin"])
                weekday = _weekday_from_date(date)
                label = None
                if not test_low <= date <= test_high:
                    label = 1 if row[LABEL] != "0" else 0
                rows.append((
                    date,
                    row["user_id"],
                    row["video_id"],
                    row["tab"],
                    hour,
                    weekday,
                    label,
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

    vocabs = [dict() for _ in BASE_FIELDS]
    for row in train:
        for index, value in enumerate(raw(row)[:len(BASE_FIELDS)]):
            if value not in vocabs[index]:
                vocabs[index][value] = len(vocabs[index])
    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs] + [INTERACTION_DIMENSION]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
    encoded = {}
    for name, rows in splits.items():
        features = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        labels = None if name == "test" else np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype="U32")
        for row_index, row in enumerate(rows):
            values = raw(row)
            for field_index, value in enumerate(values[:len(BASE_FIELDS)]):
                features[row_index, field_index] = (
                    vocabs[field_index].get(value, unknown[field_index]) + offsets[field_index]
                )
            features[row_index, len(BASE_FIELDS)] = (
                values[len(BASE_FIELDS)] + offsets[len(BASE_FIELDS)]
            )
            if labels is not None:
                labels[row_index] = row[6]
            users[row_index] = row[1]
        encoded[name] = (features, labels, users)
    return encoded, int(sum(dimensions))
