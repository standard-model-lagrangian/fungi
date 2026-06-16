"""
Pipeline performance and object-count logging.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class PerformanceLogger:
    def __init__(self, output_dir: Path):
        self.path = Path(output_dir) / "performance_log.txt"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pipeline_start = time.perf_counter()
        self.log(f"=== Pipeline started {datetime.now(timezone.utc).isoformat()} ===")

    def log(self, message: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")

    def log_counts(
        self,
        frame_index: Optional[int],
        total_objects: int,
        fungal_objects: int,
        bacterial_objects: int,
        skeletonized_objects: int,
        extra: Optional[str] = None,
    ) -> None:
        prefix = f"frame {frame_index}" if frame_index is not None else "summary"
        line = (
            f"[counts] {prefix}: total={total_objects} fungal={fungal_objects} "
            f"bacterial={bacterial_objects} skeletonized={skeletonized_objects}"
        )
        if extra:
            line += f" ({extra})"
        self.log(line)

    @contextmanager
    def timed(self, stage: str):
        start = time.perf_counter()
        self.log(f"[start] {stage}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(f"[done] {stage} elapsed_sec={elapsed:.3f}")

    def finish(self) -> None:
        total = time.perf_counter() - self._pipeline_start
        self.log(f"=== Pipeline finished total_elapsed_sec={total:.3f} ===")
