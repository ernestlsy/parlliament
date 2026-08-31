"""Profile data and screen feature groups using only the official training period.

The Research Engine builds leakage-aware schemas and causal feature recipes, runs a temporal proxy
holdout, and caches auditable evidence without consuming an experiment ID or reading validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCREENING_VERSION = "1"
TRAIN_END = 20220421

POST_IMPRESSION_COLUMNS = {
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time",
    "is_profile_enter",
}

FEATURE_GROUPS: Dict[str, List[str]] = {
    "item_metadata": [
        "author_id", "video_type", "upload_type", "music_id", "music_type", "tag",
    ],
    "content_duration": ["duration_ms", "video_duration"],
    "request_context": ["tab", "hour", "weekday"],
    "upload_recency": ["upload_age_days"],
    "user_profile": [
        "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
        "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
        "register_days_range",
    ],
    "historical_item_response": ["prior_item_impressions", "prior_item_long_rate"],
    "historical_author_response": ["prior_author_impressions", "prior_author_long_rate"],
    "historical_user_author_affinity": [
        "prior_user_author_impressions", "prior_user_author_long_rate",
    ],
}

# Canonical recipes bridge the names used by screening with the raw dataset columns that generated
# experiment code must actually read. Keep these deterministic and auditable rather than asking an
# LLM to infer aliases from a conceptual feature name.
DERIVED_FEATURE_RECIPES: Dict[str, Dict[str, Any]] = {
    "hour": {
        "source_file": "log_standard_*.csv",
        "source_columns": ["hourmin"],
        "recipe": "integer hour = floor(numeric hourmin / 100), constrained to 0..23",
    },
    "weekday": {
        "source_file": "log_standard_*.csv",
        "source_columns": ["date"],
        "recipe": "calendar weekday from YYYYMMDD date, Monday=0 through Sunday=6",
    },
    "upload_age_days": {
        "source_file": "video_features_basic_pure.csv joined to log_standard_*.csv",
        "source_columns": ["upload_dt", "date", "video_id"],
        "recipe": "nonnegative calendar days between impression date and upload_dt after video_id join",
    },
    "prior_item_impressions": {
        "source_file": "log_standard_4_08_to_4_21_pure.csv",
        "source_columns": ["video_id", "date", "time_ms"],
        "recipe": "count strictly earlier training impressions for the video_id",
    },
    "prior_item_long_rate": {
        "source_file": "log_standard_4_08_to_4_21_pure.csv",
        "source_columns": ["video_id", "date", "time_ms", "long_view"],
        "recipe": "smoothed long_view rate from strictly earlier training impressions only",
    },
    "prior_author_impressions": {
        "source_file": "training log joined with video_features_basic_pure.csv",
        "source_columns": ["video_id", "author_id", "date", "time_ms"],
        "recipe": "count strictly earlier training impressions for the joined author_id",
    },
    "prior_author_long_rate": {
        "source_file": "training log joined with video_features_basic_pure.csv",
        "source_columns": ["video_id", "author_id", "date", "time_ms", "long_view"],
        "recipe": "smoothed author long_view rate from strictly earlier training impressions only",
    },
    "prior_user_author_impressions": {
        "source_file": "training log joined with video_features_basic_pure.csv",
        "source_columns": ["user_id", "video_id", "author_id", "date", "time_ms"],
        "recipe": "count strictly earlier training impressions for the user_id-author_id pair",
    },
    "prior_user_author_long_rate": {
        "source_file": "training log joined with video_features_basic_pure.csv",
        "source_columns": [
            "user_id", "video_id", "author_id", "date", "time_ms", "long_view",
        ],
        "recipe": "smoothed pair long_view rate from strictly earlier training impressions only",
    },
}


@dataclass(frozen=True)
class ScreeningConfig:
    holdout_fraction: float = 0.25
    seed: int = 0
    timeout_seconds: int = 900
    force: bool = False
    hash_dimensions: int = 2 ** 18
    proxy_epochs: int = 3

    def validate(self) -> None:
        if not 0.1 <= self.holdout_fraction <= 0.5:
            raise ValueError("screening holdout_fraction must be between 0.1 and 0.5")
        if self.timeout_seconds <= 0 or self.hash_dimensions <= 0 or self.proxy_epochs <= 0:
            raise ValueError("screening limits must be positive")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _fast_dataset_fingerprint(data_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(data_dir.glob("*.csv"), key=lambda item: item.name):
        stat = path.stat()
        with path.open("rb") as handle:
            header = handle.readline()
        entry = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_sha256": hashlib.sha256(header).hexdigest(),
        }
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    return digest.hexdigest(), entries


def temporal_partition(dates: Sequence[int], holdout_fraction: float) -> Tuple[List[int], List[int]]:
    """Return earlier development dates and the latest holdout dates."""
    unique = sorted({int(value) for value in dates})
    if len(unique) < 2:
        raise ValueError("temporal screening requires at least two distinct training dates")
    holdout_count = max(1, int(math.ceil(len(unique) * holdout_fraction)))
    holdout_count = min(holdout_count, len(unique) - 1)
    return unique[:-holdout_count], unique[-holdout_count:]


def _safe_columns(source: str, columns: Iterable[str]) -> List[Dict[str, Any]]:
    result = []
    statistic_source = source == "video_features_statistic_pure.csv"
    for column in columns:
        if column in POST_IMPRESSION_COLUMNS:
            status = "forbidden"
            reason = "current-impression outcome unavailable when ranking"
        elif statistic_source:
            status = "quarantined"
            reason = "aggregate snapshot has no per-impression availability timestamp"
        elif column in {"is_rand"}:
            status = "excluded"
            reason = "exposure-policy marker, constant in the standard log"
        elif column in {"time_ms"}:
            status = "excluded"
            reason = "raw timestamp replaced by bounded calendar features"
        else:
            status = "eligible"
            reason = "available or derivable before the ranking decision"
        result.append({
            "evidence_id": f"feature:{source}:{column}",
            "source": source,
            "feature": column,
            "status": status,
            "reason": reason,
        })
    return result


class ResearchEngine:
    """Profiles and screens data without touching the experiment journal."""

    def __init__(self, data_dir: Path, run_dir: Path, config: ScreeningConfig):
        config.validate()
        self.data_dir = Path(data_dir)
        self.research_dir = Path(run_dir) / "research"
        self.config = config

    @property
    def catalog_path(self) -> Path:
        return self.research_dir / "feature_catalog.json"

    @property
    def report_path(self) -> Path:
        return self.research_dir / "screening_report.json"

    @property
    def manifest_path(self) -> Path:
        return self.research_dir / "manifest.json"

    def _dependency_versions(self) -> Dict[str, str]:
        import numpy
        import pandas
        import sklearn

        return {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        }

    def feature_schema(self) -> Dict[str, Any]:
        """Return compact, deterministic field provenance for implementation agents."""
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                "feature schema is unavailable until train-only screening has completed"
            )
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        profiles = catalog.get("profiles", {})
        sources: Dict[str, List[Dict[str, Any]]] = {}
        for item in catalog.get("policy", []):
            source = str(item.get("source", ""))
            if not source or source == "derived_feature_group":
                continue
            feature = str(item.get("feature", ""))
            entry = {
                "name": feature,
                "status": str(item.get("status", "")),
                "reason": str(item.get("reason", "")),
            }
            profile = profiles.get(feature)
            if isinstance(profile, dict):
                entry["training_profile"] = {
                    key: profile[key]
                    for key in (
                        "dtype", "missing_rate", "cardinality", "holdout_unseen_rate",
                    )
                    if key in profile
                }
            sources.setdefault(source, []).append(entry)
        return {
            "schema_version": 1,
            "scope": "raw headers plus train-only profiles; no official validation statistics",
            "training_date_boundary": {"date_lte": TRAIN_END},
            "raw_sources": {
                source: sorted(entries, key=lambda value: value["name"])
                for source, entries in sorted(sources.items())
            },
            "derived_features": DERIVED_FEATURE_RECIPES,
            "feature_groups": FEATURE_GROUPS,
            "implementation_rules": [
                "Use exact raw source column names; do not invent aliases.",
                "Use canonical derived-feature recipes when a requested feature is derived.",
                "Never use forbidden or quarantined fields as model inputs.",
                "Fit vocabularies, imputers, scalers, and aggregates on training data only.",
                "Preserve canonical source row order for validation predictions.",
            ],
        }

    def _cache_key(self, fingerprint: str, versions: Mapping[str, str]) -> str:
        value = {
            "screening_version": SCREENING_VERSION,
            "dataset_fingerprint": fingerprint,
            "dependencies": dict(versions),
            "config": {key: value for key, value in asdict(self.config).items() if key != "force"},
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_headers(self) -> List[Dict[str, Any]]:
        import pandas as pd

        catalog: List[Dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("*.csv"), key=lambda item: item.name):
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            catalog.extend(_safe_columns(path.name, columns))
        for group, fields in FEATURE_GROUPS.items():
            catalog.append({
                "evidence_id": f"group:{group}",
                "source": "derived_feature_group",
                "feature": group,
                "fields": fields,
                "status": "eligible",
                "reason": "predefined leakage-reviewed screening candidate",
            })
        return catalog

    def _load_training_frame(self):
        import pandas as pd

        path = self.data_dir / "log_standard_4_08_to_4_21_pure.csv"
        if not path.is_file():
            raise FileNotFoundError("KuaiRand training log is missing")
        # Deliberately do not open the 4/22-5/08 file in the research stage.
        log = pd.read_csv(path, low_memory=False)
        log = log.loc[log["date"].astype(int) <= TRAIN_END].copy()
        log["long_view"] = (
            pd.to_numeric(log["long_view"], errors="coerce").fillna(0).ne(0).astype("int8")
        )
        log["date"] = log["date"].astype(int)
        if "hourmin" in log:
            log["hour"] = (
                log["hourmin"].astype(float).fillna(0).astype(int) // 100
            ).astype(str)
        else:
            log["hour"] = "__MISSING__"
        parsed_date = pd.to_datetime(log["date"].astype(str), format="%Y%m%d", errors="coerce")
        log["weekday"] = parsed_date.dt.dayofweek.fillna(-1).astype(int).astype(str)

        basic_path = self.data_dir / "video_features_basic_pure.csv"
        if basic_path.is_file():
            basic = pd.read_csv(basic_path, low_memory=False)
            log = log.merge(basic, on="video_id", how="left", suffixes=("", "_basic"))
            if "upload_dt" in log:
                upload = pd.to_datetime(log["upload_dt"], errors="coerce")
                log["upload_age_days"] = (parsed_date - upload).dt.days.clip(lower=0)
        user_path = self.data_dir / "user_features_pure.csv"
        if user_path.is_file():
            users = pd.read_csv(user_path, low_memory=False)
            log = log.merge(users, on="user_id", how="left", suffixes=("", "_user"))
        return log

    @staticmethod
    def _learn_aggregate(development, keys: List[str], prefix: str):
        grouped = development.groupby(keys, dropna=False)["long_view"].agg(["size", "sum"]).reset_index()
        grouped = grouped.rename(columns={
            "size": f"prior_{prefix}_impressions",
            "sum": f"prior_{prefix}_positives",
        })
        return grouped

    @staticmethod
    def _apply_aggregate(frame, aggregate, keys: List[str], prefix: str, global_rate: float):
        count = f"prior_{prefix}_impressions"
        positives = f"prior_{prefix}_positives"
        frame = frame.merge(aggregate, on=keys, how="left")
        frame[count] = frame[count].fillna(0).astype(float)
        frame[f"prior_{prefix}_long_rate"] = (
            frame[positives].fillna(0).astype(float) + 20.0 * global_rate
        ) / (frame[count] + 20.0)
        return frame.drop(columns=[positives])

    def _add_train_only_history(self, development, holdout):
        global_rate = float(development["long_view"].mean())
        for keys, prefix in (
            (["video_id"], "item"),
            (["author_id"], "author"),
            (["user_id", "author_id"], "user_author"),
        ):
            if all(key in development.columns for key in keys):
                aggregate = self._learn_aggregate(development, keys, prefix)
                order_columns = ["date"] + (["time_ms"] if "time_ms" in development else [])
                ordered = development.sort_values(order_columns, kind="stable")
                count_name = f"prior_{prefix}_impressions"
                rate_name = f"prior_{prefix}_long_rate"
                counts = ordered.groupby(keys, dropna=False).cumcount().astype(float)
                positives = (
                    ordered.groupby(keys, dropna=False)["long_view"].cumsum()
                    - ordered["long_view"]
                ).astype(float)
                development.loc[ordered.index, count_name] = counts.to_numpy()
                development.loc[ordered.index, rate_name] = (
                    positives.to_numpy() + 20.0 * global_rate
                ) / (counts.to_numpy() + 20.0)
                holdout = self._apply_aggregate(holdout, aggregate, keys, prefix, global_rate)
        return development, holdout

    @staticmethod
    def _records(frame, fields: Sequence[str]):
        available = [field for field in fields if field in frame.columns]
        for values in frame[available].itertuples(index=False, name=None):
            record: Dict[str, float] = {}
            for field, value in zip(available, values):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    value = "__MISSING__"
                if isinstance(value, (int, float)) and field.startswith("prior_"):
                    record[f"num:{field}"] = float(value)
                else:
                    record[f"cat:{field}={value}"] = 1.0
            yield record

    def _fit_proxy(self, development, holdout, fields: Sequence[str]) -> Dict[str, Any]:
        import numpy as np
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.linear_model import SGDClassifier

        hasher = FeatureHasher(
            n_features=self.config.hash_dimensions, input_type="dict", alternate_sign=False,
        )
        train_x = hasher.transform(self._records(development, fields))
        valid_x = hasher.transform(self._records(holdout, fields))
        classifier = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-6,
            max_iter=self.config.proxy_epochs, tol=None,
            random_state=self.config.seed, shuffle=True,
        )
        classifier.fit(train_x, development["long_view"].to_numpy())
        scores = classifier.decision_function(valid_x)
        metrics = self._proxy_metrics(holdout, scores)
        daily = []
        for date, indexes in holdout.groupby("date").groups.items():
            positions = holdout.index.get_indexer(indexes)
            subset = holdout.loc[indexes]
            day_metrics = self._proxy_metrics(subset, np.asarray(scores)[positions])
            daily.append({"date": int(date), "primary": day_metrics["primary"]})
        daily_values = [item["primary"] for item in daily]
        user_seen = holdout["user_id"].isin(set(development["user_id"]))
        item_seen = holdout["video_id"].isin(set(development["video_id"]))
        cold_start = {}
        for name, mask in {
            "cold_user": ~user_seen,
            "cold_item": ~item_seen,
            "warm_user_and_item": user_seen & item_seen,
        }.items():
            if bool(mask.any()):
                subset = holdout.loc[mask]
                cold_start[name] = self._proxy_metrics(subset, np.asarray(scores)[mask.to_numpy()])
                cold_start[name]["rows"] = int(mask.sum())
        return {
            "metrics": metrics,
            "daily_primary": daily,
            "daily_primary_std": float(np.std(daily_values)) if daily_values else 0.0,
            "cold_start_metrics": cold_start,
        }

    @staticmethod
    def _proxy_metrics(frame, scores) -> Dict[str, float]:
        from .evaluation import evaluate

        values = evaluate(frame["user_id"].astype(str), frame["long_view"], scores)
        return {key: float(values[key]) for key in ("GAUC", "nDCG@5", "primary")}

    @staticmethod
    def _profile_column(frame, column: str, development_dates: set, holdout_dates: set) -> Dict[str, Any]:
        development = frame.loc[frame["date"].isin(development_dates), column]
        holdout = frame.loc[frame["date"].isin(holdout_dates), column]
        development_values = set(development.dropna().astype(str).unique())
        holdout_non_null = holdout.dropna().astype(str)
        unseen = (~holdout_non_null.isin(development_values)).sum()
        development_text = development.fillna("__MISSING__").astype(str)
        holdout_text = holdout.fillna("__MISSING__").astype(str)
        most_common = set(development_text.value_counts().head(200).index)
        most_common.update(holdout_text.value_counts().head(200).index)
        development_distribution = development_text.value_counts(normalize=True)
        holdout_distribution = holdout_text.value_counts(normalize=True)
        total_variation = 0.5 * sum(
            abs(
                float(development_distribution.get(value, 0.0))
                - float(holdout_distribution.get(value, 0.0))
            )
            for value in most_common
        )
        development_frame = frame.loc[frame["date"].isin(development_dates), [column, "long_view"]]
        common_values = development_text.value_counts().head(100).index
        label_rates = development_frame.loc[
            development_frame[column].fillna("__MISSING__").astype(str).isin(common_values)
        ].assign(__value=development_frame[column].fillna("__MISSING__").astype(str)).groupby(
            "__value", dropna=False
        )["long_view"].mean()
        return {
            "dtype": str(frame[column].dtype),
            "missing_rate": float(frame[column].isna().mean()),
            "cardinality": int(frame[column].nunique(dropna=True)),
            "holdout_unseen_rate": float(unseen / len(holdout_non_null)) if len(holdout_non_null) else 0.0,
            "temporal_distribution_total_variation_top_values": float(total_variation),
            "development_positive_rate_range_top_values": (
                [float(label_rates.min()), float(label_rates.max())]
                if len(label_rates) else [0.0, 0.0]
            ),
        }

    def ensure(self) -> Dict[str, Any]:
        """Create or reuse the screening artifacts and return the report."""
        started = time.monotonic()
        self.research_dir.mkdir(parents=True, exist_ok=True)
        fingerprint, files = _fast_dataset_fingerprint(self.data_dir)
        versions = self._dependency_versions()
        cache_key = self._cache_key(fingerprint, versions)
        if not self.config.force and self.manifest_path.is_file() and self.report_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") == cache_key:
                return json.loads(self.report_path.read_text(encoding="utf-8"))

        raw_catalog = self._read_headers()
        frame = self._load_training_frame()
        if int(frame["date"].max()) > TRAIN_END:
            raise RuntimeError("screening boundary violation: official validation rows were loaded")
        development_dates, holdout_dates = temporal_partition(
            frame["date"].tolist(), self.config.holdout_fraction
        )
        development_set, holdout_set = set(development_dates), set(holdout_dates)
        development = frame.loc[frame["date"].isin(development_set)].copy().reset_index(drop=True)
        holdout = frame.loc[frame["date"].isin(holdout_set)].copy().reset_index(drop=True)
        development, holdout = self._add_train_only_history(development, holdout)

        profile_by_name = {}
        for field in sorted({item for fields in FEATURE_GROUPS.values() for item in fields}):
            if field in frame.columns:
                profile_by_name[field] = self._profile_column(
                    frame, field, development_set, holdout_set
                )
        catalog = {
            "screening_version": SCREENING_VERSION,
            "policy": raw_catalog,
            "profiles": profile_by_name,
        }
        _write_json(self.catalog_path, catalog)

        baseline_fields = [field for field in ("user_id", "video_id") if field in development]
        baseline = self._fit_proxy(development, holdout, baseline_fields)
        candidates = []
        for group, group_fields in FEATURE_GROUPS.items():
            if time.monotonic() - started >= self.config.timeout_seconds:
                break
            available = [field for field in group_fields if field in development.columns]
            if not available:
                continue
            result = self._fit_proxy(development, holdout, baseline_fields + available)
            result.update({
                "evidence_id": f"screen:{group}",
                "feature_group": group,
                "fields": available,
                "primary_lift": result["metrics"]["primary"] - baseline["metrics"]["primary"],
                "coverage": float(1.0 - holdout[available].isna().all(axis=1).mean()),
                "cost": "high" if group.startswith("historical_") else "medium",
                "leakage_status": "eligible",
            })
            baseline_daily = {
                item["date"]: item["primary"] for item in baseline["daily_primary"]
            }
            result["daily_primary_lift"] = [{
                "date": item["date"],
                "lift": item["primary"] - baseline_daily.get(item["date"], item["primary"]),
            } for item in result["daily_primary"]]
            if result["primary_lift"] >= 0.002 and result["coverage"] >= 0.5:
                result["recommendation"] = "prioritize"
            elif result["primary_lift"] > 0.0:
                result["recommendation"] = "defer"
            else:
                result["recommendation"] = "reject"
            candidates.append(result)
        candidates.sort(key=lambda item: (item["primary_lift"], -item["daily_primary_std"]), reverse=True)
        report = {
            "scope": "training_only_internal_temporal_holdout",
            "status": "complete" if len(candidates) == len([
                group for group, fields in FEATURE_GROUPS.items()
                if any(field in development.columns for field in fields)
            ]) else "partial_timeout",
            "development_dates": development_dates,
            "holdout_dates": holdout_dates,
            "development_rows": len(development),
            "holdout_rows": len(holdout),
            "baseline": {"evidence_id": "screen:base_ids", "fields": baseline_fields, **baseline},
            "candidates": candidates,
            "excluded_features": [
                item for item in raw_catalog if item["status"] in {"forbidden", "quarantined"}
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(self.report_path, report)
        manifest = {
            "cache_key": cache_key,
            "screening_version": SCREENING_VERSION,
            "dataset_fingerprint": fingerprint,
            "files": files,
            "dependencies": versions,
            "config": asdict(self.config),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "official_validation_accessed": False,
            "journal_records_created": 0,
        }
        _write_json(self.manifest_path, manifest)
        return report

    def evidence_ids(
        self, archive: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> List[str]:
        values = set()
        if self.catalog_path.is_file():
            catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            values.update(
                f"feature:{item['feature']}" for item in catalog.get("policy", [])
                if item.get("status") == "eligible"
                and item.get("source") not in {
                    "log_random_4_22_to_5_08_pure.csv",
                    "log_standard_4_22_to_5_08_pure.csv",
                }
            )
            values.update(f"profile:{name}" for name in catalog.get("profiles", {}))
        if self.report_path.is_file():
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
            values.add(report.get("baseline", {}).get("evidence_id", "screen:base_ids"))
            values.update(item["evidence_id"] for item in report.get("candidates", []))
        scored = [item for item in (archive or []) if item.get("status") == "scored"]
        if scored:
            for item in scored:
                values.add(f"experiment:{item['experiment_id']}:metrics")
            latest = scored[-1]
            experiment_id = latest["experiment_id"]
            segments = latest.get("metrics", {}).get("segment_diagnostics", {}).get("segments", {})
            for dimension, entries in segments.items():
                for item in entries:
                    values.add(
                        f"experiment:{experiment_id}:segment:{dimension}:{item.get('value')}"
                    )
        return sorted(value for value in values if value)

    def build_brief(self, archive: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        excluded_groups: Dict[Tuple[str, str, str], List[str]] = {}
        for item in report.get("excluded_features", []):
            key = (
                str(item.get("status", "excluded")),
                str(item.get("source", "unknown")),
                str(item.get("reason", "")),
            )
            excluded_groups.setdefault(key, []).append(str(item.get("feature", "")))
        compact_exclusions = [{
            "status": status,
            "source": source,
            "reason": reason,
            "features": sorted(feature for feature in features if feature),
        } for (status, source, reason), features in sorted(excluded_groups.items())]
        ranked = [{
            "evidence_id": item["evidence_id"],
            "feature_group": item["feature_group"],
            "fields": item["fields"],
            "primary_lift": item["primary_lift"],
            "daily_primary_std": item["daily_primary_std"],
            "coverage": item["coverage"],
            "daily_primary_lift": item.get("daily_primary_lift", []),
            "cold_start_metrics": item.get("cold_start_metrics", {}),
            "recommendation": item.get("recommendation", "defer"),
        } for item in report.get("candidates", [])]
        scored = [item for item in archive if item.get("status") == "scored"]
        latest = scored[-1] if scored else None
        by_id = {int(item["experiment_id"]): item for item in scored}
        calibration = []
        for item in scored:
            parent = by_id.get(int(item.get("parent_experiment_id", -1)))
            parent_primary = None if parent is None else parent.get("metrics", {}).get("primary")
            observed = item.get("metrics", {}).get("primary")
            calibration.append({
                "experiment_id": item["experiment_id"],
                "predicted_primary_gain": item.get("hypothesis_prediction", {}).get(
                    "expected_primary_gain"
                ),
                "observed_primary": observed,
                "parent_experiment_id": item.get("parent_experiment_id"),
                "parent_primary": parent_primary,
                "observed_primary_gain": (
                    None if parent_primary is None or observed is None
                    else float(observed) - float(parent_primary)
                ),
            })
        segment_evidence = []
        if latest is not None:
            experiment_id = latest["experiment_id"]
            segments = latest.get("metrics", {}).get("segment_diagnostics", {}).get("segments", {})
            for dimension, entries in segments.items():
                for item in entries:
                    segment_evidence.append({
                        "evidence_id": (
                            f"experiment:{experiment_id}:segment:{dimension}:{item.get('value')}"
                        ),
                        "dimension": dimension,
                        **item,
                    })
            segment_evidence.sort(
                key=lambda item: (item.get("primary", float("inf")), -item.get("rows", 0))
            )
        return {
            "screening_scope": report["scope"],
            "screening_status": report["status"],
            "ranked_feature_evidence": ranked,
            "weakest_segment_evidence": segment_evidence[:30],
            "excluded_features": compact_exclusions,
            "latest_experiment": None if latest is None else {
                "experiment_id": latest["experiment_id"],
                "hypothesis_text": latest["hypothesis_text"],
                "metrics": latest["metrics"],
            },
            "calibration_history": calibration,
        }
