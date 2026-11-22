"""Persistent top-five massings for comparable live site configurations."""

import base64
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from .core_plans import CORE_LAYOUT_VERSION
from .experiment import write_json


def layout_scope(env, site_id, plan, organic):
    """Separate geometry, FAR targets and incompatible scoring rules."""
    scene_file = getattr(env, "_scene_cache_path", None)
    record = {
        "site": site_id or f"synthetic:{plan}",
        "scene": scene_file.name if scene_file is not None else env.seed,
        "shape": list(env.shape),
        "grid": (env.context or {}).get("gridAlignment", "world"),
        "targetFar": env.target_far,
        "farTolerance": env.far_tolerance,
        "depthWeight": env.depth_weight,
        "coreLayout": CORE_LAYOUT_VERSION if env.cores_enabled else None,
        "coreSpacing": env.min_core_spacing if env.cores_enabled else None,
        "coreTravelLimit": env.max_core_dist if env.cores_enabled else None,
        "score": "organic.v1" if organic else "legacy.v1",
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest(), record


def _pack(mask):
    return np.packbits(mask.detach().cpu().numpy()).tobytes()


def _unpack(encoded, count):
    raw = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    if len(raw) != (count + 7) // 8:
        raise ValueError("Saved mask has an incompatible size")
    return np.unpackbits(raw, count=count).tolist()


class LayoutArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.scope = None
        self.records = []
        self.warning = None

    def _read(self, scope):
        if scope != self.scope:
            path = self.root / f"{scope}.json"
            try:
                records = json.loads(path.read_text())["layouts"]
            except FileNotFoundError:
                records = []
            self.scope, self.records = scope, records
        return self.records

    def summary(self, scope):
        try:
            rows = self._read(scope)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.warning = str(exc)
            rows = []
        return {
            "scope": scope,
            "layouts": [r["summary"] for r in rows],
            "warning": self.warning,
        }

    def _history_db(self):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.root / "episodes.sqlite3")
        db.execute(
            "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, scope TEXT, stats TEXT)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS episode_scope ON episodes (scope, id)")
        return db

    def record_episodes(self, scope, rows):
        records = [
            (scope, json.dumps(row, separators=(",", ":"), allow_nan=False))
            for row in rows
        ]
        with closing(self._history_db()) as db, db:
            db.executemany("INSERT INTO episodes (scope, stats) VALUES (?, ?)", records)

    def chart_history(self, scope, after=0):
        with closing(self._history_db()) as db:
            records = db.execute(
                "SELECT id, stats FROM episodes WHERE scope = ? AND id > ? ORDER BY id DESC LIMIT 5000",
                (scope, after),
            ).fetchall()
        keys = (
            "episode",
            "siteId",
            "seed",
            "training",
            "success",
            "episodeSeconds",
            "far",
            "targetFar",
            "organicObjective",
            "directSunProxy",
            "diffuseFacadeProxy",
            "usableGoodFraction",
            "coreScore",
            "longNarrowFraction",
        )
        rows = []
        for row_id, payload in reversed(records):
            stats = json.loads(payload)
            rows.append(
                dict(id=row_id, **{key: stats[key] for key in keys if key in stats})
            )
        return rows

    def remember(self, scope, settings, row, env):
        if not row["success"] or not env.far_satisfied:
            return
        score = float(row.get("organicObjective", row["quality"]))
        if not math.isfinite(score):
            return
        rows = self._read(scope)
        if len(rows) == 5 and score <= rows[-1]["summary"]["score"]:
            return
        packed = _pack(env.occupied)
        layout_id = hashlib.sha256(packed).hexdigest()
        duplicate = next((r for r in rows if r["summary"]["id"] == layout_id), None)
        if duplicate is not None and duplicate["summary"]["score"] >= score:
            return
        summary = {
            "id": layout_id,
            "episode": row["episode"],
            "score": score,
            "far": row["far"],
            "usableArea": row.get("usableGoodArea", row["potentialUsableArea"]),
            "savedAt": datetime.now(timezone.utc).isoformat(),
        }
        record = {
            "summary": summary,
            "occupied": base64.b64encode(packed).decode(),
            "coreMask": base64.b64encode(_pack(env.core_mask & env.occupied)).decode(),
            "cores": [core.to_dict() for core in env.cores],
        }
        distinct = [r for r in rows if r is not duplicate]
        ranked = sorted([*distinct, record], key=lambda r: -r["summary"]["score"])[:5]
        write_json(
            self.root / f"{scope}.json", {"settings": settings, "layouts": ranked}
        )
        self.records = ranked
        self.warning = None

    def preview(self, scope, layout_id, shape):
        record = next(
            (r for r in self._read(scope) if r["summary"]["id"] == layout_id), None
        )
        if record is None:
            raise KeyError(layout_id)
        count = math.prod(shape)
        return {
            "layout": record["summary"],
            "shape": list(shape),
            "occupied": _unpack(record["occupied"], count),
            "coreMask": _unpack(record["coreMask"], count),
            "cores": record["cores"],
        }
