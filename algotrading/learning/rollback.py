"""
learning/rollback.py
--------------------
Deployment geri alma (rollback) mekanizmasi.

Kullanim:
  snapshot = RollbackManager.snapshot(current_config)
  # ... canary basarisiz ...
  RollbackManager.rollback(snapshot)
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)
UTC = timezone.utc


@dataclass
class ConfigSnapshot:
    """Bir an icin kaydedilen strateji konfigurasyonu."""
    snapshot_id   : str
    config        : dict
    created_at    : datetime
    label         : str = ""

    def to_json(self) -> str:
        return json.dumps({
            "snapshot_id": self.snapshot_id,
            "config"     : self.config,
            "created_at" : self.created_at.isoformat(),
            "label"      : self.label,
        }, indent=2)


class RollbackManager:
    """
    Son N konfigurasyonu bellekte ve disk'te saklar.
    rollback() en son snapshot'a geri doner.
    """

    def __init__(self, max_snapshots: int = 5, persist_dir: Optional[Path] = None) -> None:
        self._max       = max_snapshots
        self._snapshots : List[ConfigSnapshot] = []
        self._dir       = Path(persist_dir) if persist_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, config: dict, label: str = "") -> ConfigSnapshot:
        """Mevcut konfigurasyonu kaydet."""
        import hashlib, json
        ts  = datetime.now(UTC)
        sid = hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
        snap = ConfigSnapshot(
            snapshot_id = sid,
            config      = copy.deepcopy(config),
            created_at  = ts,
            label       = label or f"snapshot_{len(self._snapshots)+1}",
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max:
            self._snapshots.pop(0)
        if self._dir:
            path = self._dir / f"{sid}.json"
            path.write_text(snap.to_json())
        logger.info("Snapshot kaydedildi: %s (%s)", sid, snap.label)
        return snap

    def rollback(self, steps: int = 1) -> Optional[ConfigSnapshot]:
        """Son 'steps' adim onceki konfigurasyona don."""
        if len(self._snapshots) < steps + 1:
            logger.error("Yeterli snapshot yok (mevcut=%d, istenen=%d)",
                         len(self._snapshots), steps + 1)
            return None
        target = self._snapshots[-(steps + 1)]
        logger.warning("ROLLBACK: %s (%s)", target.snapshot_id, target.label)
        return target

    def latest(self) -> Optional[ConfigSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def history(self) -> List[ConfigSnapshot]:
        return list(self._snapshots)
