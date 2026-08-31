"""Neutral seed data contract using user, item, and request-context identifiers."""

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


def _value(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _weekday(date_value):
    try:
        return str(datetime.datetime.strptime(str(date_value), "%Y%m%d").weekday())
    except ValueError:
        return ""


def _hour(row):
    value = _value(row, ("hour", "request_hour", "impression_hour"))
    if value != "":
        return str(value)
    timestamp = _value(row, ("timestamp", "request_time", "impression_time", "time"))
    if timestamp:
        text = str(timestamp).strip()
        for format_string in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                return str(datetime.datetime.strptime(text, format_string).hour)
            except ValueError:
                pass
    return ""


def load(data_dir, max_rows_per_split=None):
    rows = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(os.path.join(data_dir, filename), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                rows.append((
                    date,
                    row["user_id"],
                    row["video_id"],
                    _value(row, ("tab",)),
                    _hour(row),
                    _value(row, ("weekday",), _weekday(date)),
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
