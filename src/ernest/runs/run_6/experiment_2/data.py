"""Neutral seed data contract with impression-time request-context features."""

import csv
import datetime
import os
import re

import numpy as np


LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"]


def _row_value(row, names):
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _date_value(value):
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _hour_value(row):
    explicit = _row_value(row, ("hour", "impression_hour", "request_hour", "hour_of_day"))
    if explicit is not None:
        return explicit

    timestamp = _row_value(
        row,
        ("timestamp", "datetime", "request_time", "impression_time", "event_time", "time"),
    )
    if timestamp is None:
        return "0"

    text = timestamp.strip()
    try:
        number = float(text)
        if number.is_integer() and 0 <= number <= 23:
            return str(int(number))
        if number.is_integer() and 0 <= number <= 2359 and int(number) % 100 < 60:
            return str(int(number) // 100)
        if number > 10000000000:
            number /= 1000.0
        if number > 100000000:
            return str(datetime.datetime.utcfromtimestamp(number).hour)
    except (TypeError, ValueError, OverflowError, OSError):
        pass

    normalized = text.replace("Z", "+00:00")
    try:
        return str(datetime.datetime.fromisoformat(normalized).hour)
    except ValueError:
        match = re.search(r"(?:T|\s)([01]?\d|2[0-3])(?::\d{2})?", text)
        return match.group(1).lstrip("0") or "0" if match else "0"


def _weekday_value(row):
    explicit = _row_value(row, ("weekday", "day_of_week", "week_day"))
    if explicit is not None:
        return explicit
    parsed = _date_value(_row_value(row, ("date",)))
    return str(parsed.weekday()) if parsed is not None else "0"


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
                    row["tab"],
                    _hour_value(row),
                    _weekday_value(row),
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
