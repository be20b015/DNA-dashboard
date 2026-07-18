"""
KPI aggregation over a streamed collection of SequenceRecord objects.

Kept O(1) in memory per-record: we maintain running sums/counters rather
than storing every sequence, so this scales to millions of records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from .parser import SequenceRecord


@dataclass
class RunningKPIs:
    n_records: int = 0
    n_malformed: int = 0  # Track malformed/corrupted records
    total_length: int = 0
    total_gc: int = 0
    min_length: int = None
    max_length: int = None
    length_samples: List[int] = field(default_factory=list)  # reservoir for histogram
    gc_samples: List[float] = field(default_factory=list)
    ambiguous_base_count: int = 0  # N, or non-ACGT/U bases — proxy for "anomalies"
    
    # Phred Quality Tracking
    total_quality_sum: float = 0.0
    total_quality_bases: int = 0
    
    # System & Execution Metrics
    elapsed_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    
    _max_samples: int = 20000

    def update(self, record: SequenceRecord) -> None:
        # 1. Guard against malformed records
        # (Adjust this condition if SequenceRecord exposes a specific validation flag or method)
        if not record or getattr(record, "is_malformed", False) or not record.sequence:
            self.n_malformed += 1
            return

        self.n_records += 1
        self.total_length += record.length
        
        # Optimize string manipulation by calling upper once
        seq_upper = record.sequence.upper()
        gc = seq_upper.count("G") + seq_upper.count("C")
        self.total_gc += gc

        # Track min/max bounds
        if self.min_length is None or record.length < self.min_length:
            self.min_length = record.length
        if self.max_length is None or record.length > self.max_length:
            self.max_length = record.length

        # Count ambiguous bases
        ambiguous = sum(1 for b in seq_upper if b not in "ACGTU")
        self.ambiguous_base_count += ambiguous

        # Aggregate Phred Quality scores (Assumes record.phred_quality returns a list of numerical scores)
        if hasattr(record, "phred_quality") and record.phred_quality:
            self.total_quality_sum += sum(record.phred_quality)
            self.total_quality_bases += len(record.phred_quality)

        # Reservoir-style cap so histogram data doesn't grow unbounded on huge files
        if len(self.length_samples) < self._max_samples:
            self.length_samples.append(record.length)
            # Support both direct float metrics and fallbacks
            gc_content = getattr(record, "gc_content", (gc / record.length if record.length else 0.0))
            self.gc_samples.append(gc_content)

    @property
    def mean_length(self) -> float:
        return round(self.total_length / self.n_records, 1) if self.n_records else 0.0

    @property
    def overall_gc_pct(self) -> float:
        return round(100 * self.total_gc / self.total_length, 2) if self.total_length else 0.0

    @property
    def ambiguous_pct(self) -> float:
        return round(100 * self.ambiguous_base_count / self.total_length, 4) if self.total_length else 0.0

    @property
    def mean_phred_quality(self) -> float:
        if not self.total_quality_bases:
            return 0.0
        return round(self.total_quality_sum / self.total_quality_bases, 2)

    def summary_dict(self) -> dict:
        return {
            "records": self.n_records,
            "n_malformed": self.n_malformed,
            "total_bases": self.total_length,
            "mean_length": self.mean_length,
            "min_length": self.min_length or 0,
            "max_length": self.max_length or 0,
            "overall_gc_pct": self.overall_gc_pct,
            "ambiguous_base_pct": self.ambiguous_pct,
            "mean_phred_quality": self.mean_phred_quality,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }
