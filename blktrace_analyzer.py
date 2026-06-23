#!/usr/bin/env python3
"""
blktrace Analyzer — Queue Depth, LBA Hotspots, I/O Sizes, Throughput & Latency
================================================================================
Parses blkparse text output and generates comprehensive storage performance charts.

Usage:
    # From live blktrace capture:
    blktrace -d /dev/sda -o - | blkparse -i - -o trace.txt
    python3 blktrace_analyzer.py trace.txt

    # From saved blktrace binary:
    blkparse -i sda -o trace.txt
    python3 blktrace_analyzer.py trace.txt

    # With options:
    python3 blktrace_analyzer.py trace.txt --time-bucket 0.1 --lba-bins 256 --output-dir ./results

"""

import re
import sys
import os
import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ────────────────────────────────────────────────────────────────
SECTOR_SIZE = 512  # bytes per sector
KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

# blkparse action codes we care about
ACTION_QUEUE = "Q"
ACTION_GETRQ = "G"
ACTION_INSERT = "I"
ACTION_ISSUE = "D"
ACTION_COMPLETE = "C"
ACTION_MERGE = "M"
ACTION_FRONTMERGE = "F"
ACTION_PLUG = "P"
ACTION_UNPLUG = "U"

RWBS_READ = "R"
RWBS_WRITE = "W"
RWBS_DISCARD = "D"
RWBS_FLUSH = "F"
RWBS_FUA = "N"
RWBS_SYNC = "S"


# ─── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class TraceEvent:
    """Single parsed blkparse event."""

    major: int
    minor: int
    cpu: int
    seq: int
    timestamp: float  # seconds
    pid: int
    action: str
    rwbs: str
    sector: int
    nblocks: int
    process: str
    is_read: bool
    is_write: bool
    is_discard: bool
    size_bytes: int


@dataclass
class IORequest:
    """Tracks an I/O request through its lifecycle for latency computation."""

    sector: int
    nblocks: int
    is_read: bool
    queue_time: Optional[float] = None
    issue_time: Optional[float] = None
    complete_time: Optional[float] = None

    @property
    def q2c_latency_us(self) -> Optional[float]:
        """Total latency: queue → completion (microseconds)."""
        if self.queue_time is not None and self.complete_time is not None:
            return (self.complete_time - self.queue_time) * 1e6
        return None

    @property
    def d2c_latency_us(self) -> Optional[float]:
        """Device latency: issue → completion (microseconds)."""
        if self.issue_time is not None and self.complete_time is not None:
            return (self.complete_time - self.issue_time) * 1e6
        return None

    @property
    def q2d_latency_us(self) -> Optional[float]:
        """Software stack latency: queue → issue (microseconds)."""
        if self.queue_time is not None and self.issue_time is not None:
            return (self.issue_time - self.queue_time) * 1e6
        return None


# ─── Parser ───────────────────────────────────────────────────────────────────
# blkparse default output format:
#   8,0    3      152     0.003294012  1234  Q  WS 123456 + 8 [dd]
#   major,minor cpu seq    timestamp    pid action rwbs sector + nblocks [proc]
# Some lines may lack sector info (e.g., plug/unplug events).

BLKPARSE_RE = re.compile(
    r"^\s*(\d+),(\d+)\s+"  # major,minor
    r"(\d+)\s+"  # cpu
    r"(\d+)\s+"  # sequence
    r"([\d.]+)\s+"  # timestamp
    r"(\d+)\s+"  # pid
    r"([A-Z])\s+"  # action (single char)
    r"([A-Z]*)\s+"  # rwbs flags
    r"(\d+)\s*"  # sector
    r"\+\s*(\d+)\s*"  # + nblocks
    r"(?:\[(.+?)\])?"  # [process] (optional)
)

# Lines without sector info (plug/unplug/etc.)
BLKPARSE_NOSECTOR_RE = re.compile(
    r"^\s*(\d+),(\d+)\s+"
    r"(\d+)\s+"
    r"(\d+)\s+"
    r"([\d.]+)\s+"
    r"(\d+)\s+"
    r"([A-Z])\s+"
    r"([A-Z]*)"
)


def parse_blkparse_line(line: str) -> Optional[TraceEvent]:
    """Parse a single blkparse output line into a TraceEvent."""
    m = BLKPARSE_RE.match(line)
    if not m:
        return None

    rwbs = m.group(8)
    is_read = RWBS_READ in rwbs
    is_write = RWBS_WRITE in rwbs
    is_discard = RWBS_DISCARD in rwbs and m.group(7) != ACTION_ISSUE
    nblocks = int(m.group(10))

    return TraceEvent(
        major=int(m.group(1)),
        minor=int(m.group(2)),
        cpu=int(m.group(3)),
        seq=int(m.group(4)),
        timestamp=float(m.group(5)),
        pid=int(m.group(6)),
        action=m.group(7),
        rwbs=rwbs,
        sector=int(m.group(9)),
        nblocks=nblocks,
        process=m.group(11) or "",
        is_read=is_read,
        is_write=is_write,
        is_discard=is_discard,
        size_bytes=nblocks * SECTOR_SIZE,
    )


def load_trace(filepath: str) -> List[TraceEvent]:
    """Load and parse a blkparse output file."""
    events = []
    parse_errors = 0
    total_lines = 0

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("CPU") or line.startswith("Total") or "==" in line:
                continue
            total_lines += 1
            ev = parse_blkparse_line(line)
            if ev:
                events.append(ev)
            else:
                parse_errors += 1

    events.sort(key=lambda e: e.timestamp)

    print(f"  Parsed {len(events):,} events from {total_lines:,} lines "
          f"({parse_errors} unparseable)")
    if events:
        duration = events[-1].timestamp - events[0].timestamp
        print(f"  Trace duration: {duration:.3f} seconds")
        print(f"  Time range: {events[0].timestamp:.6f} – {events[-1].timestamp:.6f}")
    return events


# ─── Analysis Engine ──────────────────────────────────────────────────────────
class BlktraceAnalyzer:
    """Core analysis engine for blktrace data."""

    def __init__(self, events: List[TraceEvent], time_bucket: float = 0.1,
                 lba_bins: int = 256):
        self.events = events
        self.time_bucket = time_bucket
        self.lba_bins = lba_bins

        if not events:
            raise ValueError("No events to analyze")

        self.t0 = events[0].timestamp
        self.t_end = events[-1].timestamp
        self.duration = self.t_end - self.t0

        # Separate by direction
        self.reads = [e for e in events if e.is_read]
        self.writes = [e for e in events if e.is_write]
        self.discards = [e for e in events if e.is_discard]

        # Precompute
        self._compute_latencies()

    # ── Latency Tracking ──────────────────────────────────────────────────
    def _compute_latencies(self):
        """Match Q/D/C events to compute per-I/O latencies."""
        # Key: (sector, nblocks) → list of IORequest
        # We use a dict of lists to handle overlapping I/Os to same sector
        pending: Dict[Tuple[int, int], List[IORequest]] = defaultdict(list)
        self.completed_ios: List[IORequest] = []

        for ev in self.events:
            key = (ev.sector, ev.nblocks)

            if ev.action == ACTION_QUEUE:
                req = IORequest(
                    sector=ev.sector,
                    nblocks=ev.nblocks,
                    is_read=ev.is_read,
                    queue_time=ev.timestamp,
                )
                pending[key].append(req)

            elif ev.action == ACTION_ISSUE:
                if pending[key]:
                    req = pending[key][0]  # FIFO match
                    req.issue_time = ev.timestamp
                else:
                    # Issue without prior queue (can happen with merges)
                    req = IORequest(
                        sector=ev.sector,
                        nblocks=ev.nblocks,
                        is_read=ev.is_read,
                        issue_time=ev.timestamp,
                    )
                    pending[key].append(req)

            elif ev.action == ACTION_COMPLETE:
                if pending[key]:
                    req = pending[key].pop(0)
                    req.complete_time = ev.timestamp
                    self.completed_ios.append(req)

        print(f"  Matched {len(self.completed_ios):,} complete I/O lifecycles")

    # ── Queue Depth Over Time ─────────────────────────────────────────────
    def compute_queue_depth(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute instantaneous queue depth (D events increment, C events decrement)."""
        # Collect all D and C events with timestamps
        changes = []
        for ev in self.events:
            if ev.action == ACTION_ISSUE:
                changes.append((ev.timestamp - self.t0, +1, ev.is_read))
            elif ev.action == ACTION_COMPLETE:
                changes.append((ev.timestamp - self.t0, -1, ev.is_read))

        if not changes:
            empty = np.array([0.0])
            return empty, np.array([0]), np.array([0]), np.array([0])

        changes.sort(key=lambda x: x[0])

        times = []
        qd_total = []
        qd_read = []
        qd_write = []
        depth = 0
        depth_r = 0
        depth_w = 0

        for t, delta, is_read in changes:
            depth += delta
            if is_read:
                depth_r += delta
            else:
                depth_w += delta
            depth = max(0, depth)
            depth_r = max(0, depth_r)
            depth_w = max(0, depth_w)
            times.append(t)
            qd_total.append(depth)
            qd_read.append(depth_r)
            qd_write.append(depth_w)

        return np.array(times), np.array(qd_total), np.array(qd_read), np.array(qd_write)

    def compute_queue_depth_bucketed(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Average queue depth per time bucket."""
        times, qd_total, qd_read, qd_write = self.compute_queue_depth()
        if len(times) == 0:
            return np.array([0]), np.array([0]), np.array([0]), np.array([0])

        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        bucket_times = np.linspace(0, self.duration, n_buckets)
        avg_total = np.zeros(n_buckets)
        avg_read = np.zeros(n_buckets)
        avg_write = np.zeros(n_buckets)

        for i in range(n_buckets):
            t_lo = i * self.time_bucket
            t_hi = (i + 1) * self.time_bucket
            mask = (times >= t_lo) & (times < t_hi)
            if np.any(mask):
                avg_total[i] = np.mean(qd_total[mask])
                avg_read[i] = np.mean(qd_read[mask])
                avg_write[i] = np.mean(qd_write[mask])

        return bucket_times, avg_total, avg_read, avg_write

    # ── LBA Hotspot Analysis ──────────────────────────────────────────────
    def compute_lba_heatmap(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """2D heatmap: time (x) vs LBA range (y), values = I/O count."""
        io_events = [e for e in self.events
                     if e.action in (ACTION_QUEUE, ACTION_ISSUE) and e.sector > 0]
        if not io_events:
            return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

        sectors = np.array([e.sector for e in io_events])
        max_sector = sectors.max()

        n_time_bins = max(1, min(200, int(self.duration / self.time_bucket)))
        n_lba_bins = self.lba_bins

        heatmap_all = np.zeros((n_lba_bins, n_time_bins))
        heatmap_read = np.zeros((n_lba_bins, n_time_bins))
        heatmap_write = np.zeros((n_lba_bins, n_time_bins))

        for ev in io_events:
            t_idx = min(int((ev.timestamp - self.t0) / self.duration * n_time_bins),
                        n_time_bins - 1)
            s_idx = min(int(ev.sector / (max_sector + 1) * n_lba_bins),
                        n_lba_bins - 1)
            heatmap_all[s_idx, t_idx] += 1
            if ev.is_read:
                heatmap_read[s_idx, t_idx] += 1
            elif ev.is_write:
                heatmap_write[s_idx, t_idx] += 1

        return heatmap_all, heatmap_read, heatmap_write, max_sector

    def compute_lba_histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LBA access frequency histogram (read vs write)."""
        io_events = [e for e in self.events
                     if e.action in (ACTION_QUEUE,) and e.sector > 0]
        if not io_events:
            return np.array([0]), np.array([0]), np.array([0])

        sectors = np.array([e.sector for e in io_events])
        max_sector = sectors.max()
        bins = np.linspace(0, max_sector, self.lba_bins + 1)

        read_sectors = [e.sector for e in io_events if e.is_read]
        write_sectors = [e.sector for e in io_events if e.is_write]

        hist_r, _ = np.histogram(read_sectors, bins=bins) if read_sectors else (np.zeros(self.lba_bins), None)
        hist_w, _ = np.histogram(write_sectors, bins=bins) if write_sectors else (np.zeros(self.lba_bins), None)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, hist_r, hist_w

    # ── I/O Size Distribution ─────────────────────────────────────────────
    def compute_io_sizes(self) -> Tuple[List[int], List[int], List[int]]:
        """I/O size distribution in bytes (reads, writes, all)."""
        q_events = [e for e in self.events if e.action == ACTION_QUEUE and e.nblocks > 0]
        read_sizes = [e.size_bytes for e in q_events if e.is_read]
        write_sizes = [e.size_bytes for e in q_events if e.is_write]
        all_sizes = [e.size_bytes for e in q_events]
        return read_sizes, write_sizes, all_sizes

    # ── Throughput Over Time ──────────────────────────────────────────────
    def compute_throughput(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Throughput in MB/s per time bucket (based on C events)."""
        c_events = [e for e in self.events if e.action == ACTION_COMPLETE]
        if not c_events:
            return np.array([0]), np.array([0]), np.array([0])

        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        read_bytes = np.zeros(n_buckets)
        write_bytes = np.zeros(n_buckets)

        for ev in c_events:
            idx = min(int((ev.timestamp - self.t0) / self.time_bucket), n_buckets - 1)
            if ev.is_read:
                read_bytes[idx] += ev.size_bytes
            else:
                write_bytes[idx] += ev.size_bytes

        bucket_times = np.arange(n_buckets) * self.time_bucket
        read_mbps = read_bytes / self.time_bucket / MB
        write_mbps = write_bytes / self.time_bucket / MB

        return bucket_times, read_mbps, write_mbps

    # ── IOPS Over Time ────────────────────────────────────────────────────
    def compute_iops(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """IOPS per time bucket (based on C events)."""
        c_events = [e for e in self.events if e.action == ACTION_COMPLETE]
        if not c_events:
            return np.array([0]), np.array([0]), np.array([0])

        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        read_count = np.zeros(n_buckets)
        write_count = np.zeros(n_buckets)

        for ev in c_events:
            idx = min(int((ev.timestamp - self.t0) / self.time_bucket), n_buckets - 1)
            if ev.is_read:
                read_count[idx] += 1
            else:
                write_count[idx] += 1

        bucket_times = np.arange(n_buckets) * self.time_bucket
        read_iops = read_count / self.time_bucket
        write_iops = write_count / self.time_bucket

        return bucket_times, read_iops, write_iops

    # ── Queued I/O Distribution (Q Events) ──────────────────────────────
    def compute_queued_iops(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Queued IOPS per time bucket (based on Q events)."""
        q_events = [e for e in self.events if e.action == ACTION_QUEUE]
        if not q_events:
            return np.array([0]), np.array([0]), np.array([0])

        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        read_count = np.zeros(n_buckets)
        write_count = np.zeros(n_buckets)

        for ev in q_events:
            idx = min(int((ev.timestamp - self.t0) / self.time_bucket), n_buckets - 1)
            if ev.is_read:
                read_count[idx] += 1
            elif ev.is_write:
                write_count[idx] += 1

        bucket_times = np.arange(n_buckets) * self.time_bucket
        read_iops = read_count / self.time_bucket
        write_iops = write_count / self.time_bucket

        return bucket_times, read_iops, write_iops

    def compute_queued_count_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Distribution of total queued I/Os per time bucket from Q events."""
        q_events = [e for e in self.events if e.action == ACTION_QUEUE]
        if not q_events:
            return np.array([0]), np.array([0])

        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        bucket_counts = np.zeros(n_buckets)

        for ev in q_events:
            idx = min(int((ev.timestamp - self.t0) / self.time_bucket), n_buckets - 1)
            bucket_counts[idx] += 1

        if np.all(bucket_counts == 0):
            return np.array([0]), np.array([0])

        max_count = int(bucket_counts.max())
        bins = np.arange(0, max_count + 2) - 0.5
        hist, edges = np.histogram(bucket_counts, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, hist

    # ── Latency Distributions ─────────────────────────────────────────────
    def compute_latency_distributions(self) -> Dict[str, np.ndarray]:
        """Return latency arrays for Q2C, D2C, Q2D (in microseconds)."""
        q2c_read, q2c_write = [], []
        d2c_read, d2c_write = [], []
        q2d_read, q2d_write = [], []

        for req in self.completed_ios:
            lat = req.q2c_latency_us
            if lat is not None and lat > 0:
                (q2c_read if req.is_read else q2c_write).append(lat)
            lat = req.d2c_latency_us
            if lat is not None and lat > 0:
                (d2c_read if req.is_read else d2c_write).append(lat)
            lat = req.q2d_latency_us
            if lat is not None and lat > 0:
                (q2d_read if req.is_read else q2d_write).append(lat)

        return {
            "q2c_read": np.array(q2c_read) if q2c_read else np.array([]),
            "q2c_write": np.array(q2c_write) if q2c_write else np.array([]),
            "d2c_read": np.array(d2c_read) if d2c_read else np.array([]),
            "d2c_write": np.array(d2c_write) if d2c_write else np.array([]),
            "q2d_read": np.array(q2d_read) if q2d_read else np.array([]),
            "q2d_write": np.array(q2d_write) if q2d_write else np.array([]),
        }

    def compute_latency_over_time(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
        """Latency percentiles (p50, p99) over time buckets."""
        n_buckets = max(1, int(np.ceil(self.duration / self.time_bucket)))
        bucket_lats_read = [[] for _ in range(n_buckets)]
        bucket_lats_write = [[] for _ in range(n_buckets)]

        for req in self.completed_ios:
            lat = req.d2c_latency_us or req.q2c_latency_us
            if lat is None or lat <= 0:
                continue
            t_ref = req.issue_time or req.queue_time
            if t_ref is None:
                continue
            idx = min(int((t_ref - self.t0) / self.time_bucket), n_buckets - 1)
            if req.is_read:
                bucket_lats_read[idx].append(lat)
            else:
                bucket_lats_write[idx].append(lat)

        times = np.arange(n_buckets) * self.time_bucket
        p50_r = np.array([np.median(b) if b else 0 for b in bucket_lats_read])
        p99_r = np.array([np.percentile(b, 99) if b else 0 for b in bucket_lats_read])
        p50_w = np.array([np.median(b) if b else 0 for b in bucket_lats_write])
        p99_w = np.array([np.percentile(b, 99) if b else 0 for b in bucket_lats_write])

        return times, p50_r, p99_r, p50_w, p99_w

    # ── Summary Statistics ────────────────────────────────────────────────
    def print_summary(self):
        """Print a human-readable summary of the trace."""
        print("\n" + "=" * 72)
        print("  BLKTRACE ANALYSIS SUMMARY")
        print("=" * 72)

        print(f"\n  Duration        : {self.duration:.3f} s")
        print(f"  Total events    : {len(self.events):,}")
        print(f"  Read events     : {len(self.reads):,}")
        print(f"  Write events    : {len(self.writes):,}")
        print(f"  Discard events  : {len(self.discards):,}")
        print(f"  Completed I/Os  : {len(self.completed_ios):,}")

        # Throughput
        c_events = [e for e in self.events if e.action == ACTION_COMPLETE]
        total_bytes_r = sum(e.size_bytes for e in c_events if e.is_read)
        total_bytes_w = sum(e.size_bytes for e in c_events if e.is_write)
        if self.duration > 0:
            print(f"\n  Read throughput  : {total_bytes_r / MB / self.duration:.2f} MB/s "
                  f"({total_bytes_r / GB:.2f} GB total)")
            print(f"  Write throughput : {total_bytes_w / MB / self.duration:.2f} MB/s "
                  f"({total_bytes_w / GB:.2f} GB total)")
            print(f"  Total IOPS       : {len(c_events) / self.duration:,.0f}")

        # I/O sizes
        r_sizes, w_sizes, all_sizes = self.compute_io_sizes()
        if all_sizes:
            print(f"\n  I/O size (all)   : min={min(all_sizes)/KB:.1f}K  "
                  f"median={np.median(all_sizes)/KB:.1f}K  "
                  f"max={max(all_sizes)/KB:.1f}K  "
                  f"mean={np.mean(all_sizes)/KB:.1f}K")

        # Latencies
        lats = self.compute_latency_distributions()
        for name, arr in lats.items():
            if len(arr) > 0:
                label = name.replace("_", " ").upper()
                print(f"\n  {label}:")
                print(f"    min={arr.min():.1f}µs  p50={np.median(arr):.1f}µs  "
                      f"p95={np.percentile(arr, 95):.1f}µs  "
                      f"p99={np.percentile(arr, 99):.1f}µs  "
                      f"max={arr.max():.1f}µs")

        # Queue depth
        _, qd_total, _, _ = self.compute_queue_depth()
        if len(qd_total) > 0:
            print(f"\n  Queue depth      : max={qd_total.max()}  "
                  f"mean={qd_total.mean():.1f}")

        print("\n" + "=" * 72)


# ─── Plotting ─────────────────────────────────────────────────────────────────
COLORS = {
    "read": "#2196F3",
    "write": "#FF5722",
    "total": "#4CAF50",
    "p50": "#2196F3",
    "p99": "#F44336",
    "bg": "#FAFAFA",
    "grid": "#E0E0E0",
}


def setup_ax(ax, title, xlabel, ylabel):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.set_facecolor(COLORS["bg"])


def plot_queue_depth(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot queue depth over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    fig.suptitle("Queue Depth Analysis", fontsize=15, fontweight="bold")

    # Instantaneous QD
    times, qd_total, qd_read, qd_write = analyzer.compute_queue_depth()
    if len(times) > 10000:
        # Downsample for plotting
        step = len(times) // 5000
        idx = np.arange(0, len(times), step)
        times_ds, qd_t, qd_r, qd_w = times[idx], qd_total[idx], qd_read[idx], qd_write[idx]
    else:
        times_ds, qd_t, qd_r, qd_w = times, qd_total, qd_read, qd_write

    ax1.fill_between(times_ds, qd_t, alpha=0.2, color=COLORS["total"])
    ax1.plot(times_ds, qd_t, linewidth=0.5, color=COLORS["total"], label="Total")
    ax1.plot(times_ds, qd_r, linewidth=0.5, color=COLORS["read"], alpha=0.7, label="Read")
    ax1.plot(times_ds, qd_w, linewidth=0.5, color=COLORS["write"], alpha=0.7, label="Write")
    setup_ax(ax1, "Instantaneous Queue Depth", "", "Queue Depth")
    ax1.legend(loc="upper right")

    # Bucketed average QD
    bt, avg_t, avg_r, avg_w = analyzer.compute_queue_depth_bucketed()
    ax2.bar(bt, avg_r, width=analyzer.time_bucket * 0.9, color=COLORS["read"],
            alpha=0.7, label="Read")
    ax2.bar(bt, avg_w, width=analyzer.time_bucket * 0.9, bottom=avg_r,
            color=COLORS["write"], alpha=0.7, label="Write")
    setup_ax(ax2, f"Average Queue Depth (bucket={analyzer.time_bucket}s)", "Time (s)", "Avg QD")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    path = os.path.join(output_dir, "01_queue_depth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lba_hotspots(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot LBA access heatmap and histogram."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    heatmap_all, heatmap_read, heatmap_write, max_sector = analyzer.compute_lba_heatmap()

    # Combined heatmap
    ax1 = fig.add_subplot(gs[0, :])
    if heatmap_all.max() > 0:
        im = ax1.imshow(heatmap_all, aspect="auto", origin="lower",
                        norm=LogNorm(vmin=max(1, heatmap_all[heatmap_all > 0].min()),
                                     vmax=heatmap_all.max()),
                        cmap="inferno", interpolation="nearest")
        plt.colorbar(im, ax=ax1, label="I/O Count (log scale)", shrink=0.8)
    lba_gb = max_sector * SECTOR_SIZE / GB if isinstance(max_sector, (int, float, np.integer)) else 0
    setup_ax(ax1, f"LBA Access Heatmap (0–{lba_gb:.1f} GB)", "Time →", "LBA Range →")
    n_yticks = min(8, analyzer.lba_bins)
    ytick_pos = np.linspace(0, heatmap_all.shape[0] - 1, n_yticks)
    ytick_labels = [f"{v * SECTOR_SIZE / GB:.1f}G"
                    for v in np.linspace(0, max_sector if isinstance(max_sector, (int, float, np.integer)) else 0,
                                         n_yticks)]
    ax1.set_yticks(ytick_pos)
    ax1.set_yticklabels(ytick_labels, fontsize=8)

    # LBA frequency histogram
    bin_centers, hist_r, hist_w = analyzer.compute_lba_histogram()
    ax2 = fig.add_subplot(gs[1, 0])
    bin_gb = bin_centers * SECTOR_SIZE / GB
    if hist_r.sum() > 0:
        ax2.bar(bin_gb, hist_r, width=(bin_gb[1] - bin_gb[0]) * 0.9 if len(bin_gb) > 1 else 0.1,
                color=COLORS["read"], alpha=0.7, label="Read")
    if hist_w.sum() > 0:
        ax2.bar(bin_gb, hist_w, width=(bin_gb[1] - bin_gb[0]) * 0.9 if len(bin_gb) > 1 else 0.1,
                bottom=hist_r, color=COLORS["write"], alpha=0.7, label="Write")
    setup_ax(ax2, "LBA Access Frequency", "LBA Offset (GB)", "I/O Count")
    ax2.legend(loc="upper right", fontsize=9)

    # Top hotspot regions table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    combined = hist_r + hist_w
    top_n = min(10, len(combined))
    top_idx = np.argsort(combined)[-top_n:][::-1]
    table_data = []
    for i, idx in enumerate(top_idx):
        if combined[idx] == 0:
            break
        table_data.append([
            f"#{i+1}",
            f"{bin_gb[idx]:.2f} GB",
            f"{int(hist_r[idx]):,}",
            f"{int(hist_w[idx]):,}",
            f"{int(combined[idx]):,}",
        ])
    if table_data:
        table = ax3.table(cellText=table_data,
                          colLabels=["Rank", "LBA Offset", "Reads", "Writes", "Total"],
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax3.set_title("Top LBA Hotspots", fontsize=13, fontweight="bold", pad=10)

    path = os.path.join(output_dir, "02_lba_hotspots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_io_sizes(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot I/O size distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("I/O Size Distribution", fontsize=15, fontweight="bold")

    r_sizes, w_sizes, all_sizes = analyzer.compute_io_sizes()

    # Common power-of-2 bins
    if all_sizes:
        max_size = max(all_sizes)
        bin_edges = [2**i for i in range(9, 25) if 2**i <= max_size * 2]  # 512B to 16MB
        if not bin_edges:
            bin_edges = [512, 1024, 4096, 8192, 16384, 65536, 131072, 524288, 1048576]
        bin_labels = []
        for b in bin_edges:
            if b >= MB:
                bin_labels.append(f"{b//MB}M")
            elif b >= KB:
                bin_labels.append(f"{b//KB}K")
            else:
                bin_labels.append(f"{b}B")

    datasets = [
        (all_sizes, "All I/O", COLORS["total"]),
        (r_sizes, "Reads", COLORS["read"]),
        (w_sizes, "Writes", COLORS["write"]),
    ]

    for ax, (sizes, label, color) in zip(axes, datasets):
        if not sizes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            setup_ax(ax, label, "I/O Size", "Count")
            continue

        # Bucket into power-of-2 bins
        counts = defaultdict(int)
        for s in sizes:
            # Find closest bin
            for i, b in enumerate(bin_edges):
                if s <= b:
                    counts[i] = counts.get(i, 0) + 1
                    break
            else:
                counts[len(bin_edges) - 1] = counts.get(len(bin_edges) - 1, 0) + 1

        x_pos = range(len(bin_edges))
        heights = [counts.get(i, 0) for i in x_pos]
        ax.bar(x_pos, heights, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
        setup_ax(ax, label, "I/O Size", "Count")

        # Annotate top bar
        if heights:
            max_idx = np.argmax(heights)
            ax.annotate(f"{heights[max_idx]:,}",
                        xy=(max_idx, heights[max_idx]),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "03_io_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_throughput(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot throughput and IOPS over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Storage Throughput & IOPS", fontsize=15, fontweight="bold")

    # Throughput
    bt, read_mbps, write_mbps = analyzer.compute_throughput()
    ax1.fill_between(bt, read_mbps, alpha=0.3, color=COLORS["read"])
    ax1.plot(bt, read_mbps, linewidth=1, color=COLORS["read"], label="Read")
    ax1.fill_between(bt, write_mbps, alpha=0.3, color=COLORS["write"])
    ax1.plot(bt, write_mbps, linewidth=1, color=COLORS["write"], label="Write")
    total_mbps = read_mbps + write_mbps
    ax1.plot(bt, total_mbps, linewidth=1, color=COLORS["total"],
             linestyle="--", alpha=0.7, label="Total")
    setup_ax(ax1, f"Throughput (bucket={analyzer.time_bucket}s)", "", "MB/s")
    ax1.legend(loc="upper right")

    # IOPS
    bt, read_iops, write_iops = analyzer.compute_iops()
    ax2.fill_between(bt, read_iops, alpha=0.3, color=COLORS["read"])
    ax2.plot(bt, read_iops, linewidth=1, color=COLORS["read"], label="Read IOPS")
    ax2.fill_between(bt, write_iops, alpha=0.3, color=COLORS["write"])
    ax2.plot(bt, write_iops, linewidth=1, color=COLORS["write"], label="Write IOPS")
    setup_ax(ax2, f"IOPS (bucket={analyzer.time_bucket}s)", "Time (s)", "IOPS")
    ax2.legend(loc="upper right")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    path = os.path.join(output_dir, "04_throughput_iops.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_queued_ios_distribution(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot queued I/O rate over time and its per-bucket count distribution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    fig.suptitle("Queued I/O Distribution (Q Events)", fontsize=15, fontweight="bold")

    # Queued IOPS timeline
    bt, read_qiops, write_qiops = analyzer.compute_queued_iops()
    total_qiops = read_qiops + write_qiops
    ax1.fill_between(bt, read_qiops, alpha=0.25, color=COLORS["read"])
    ax1.plot(bt, read_qiops, linewidth=1, color=COLORS["read"], label="Queued Read IOPS")
    ax1.fill_between(bt, write_qiops, alpha=0.25, color=COLORS["write"])
    ax1.plot(bt, write_qiops, linewidth=1, color=COLORS["write"], label="Queued Write IOPS")
    ax1.plot(bt, total_qiops, linewidth=1, linestyle="--", color=COLORS["total"], label="Queued Total IOPS")
    setup_ax(ax1, f"Queued IOPS Over Time (bucket={analyzer.time_bucket}s)", "", "Queued IOPS")
    ax1.legend(loc="upper right")

    # Distribution of queued I/O counts per time bucket
    centers, hist = analyzer.compute_queued_count_distribution()
    ax2.bar(centers, hist, width=0.9, color=COLORS["total"], alpha=0.8, edgecolor="white")
    setup_ax(ax2, "Distribution of Total Queued I/Os per Time Bucket", "Queued I/Os in Bucket", "Number of Buckets")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(output_dir, "06_queued_io_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_latency(analyzer: BlktraceAnalyzer, output_dir: str):
    """Plot latency distributions and latency over time."""
    lats = analyzer.compute_latency_distributions()

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Row 1: Histograms for Q2C and D2C ──
    for col, (prefix, title) in enumerate([("q2c", "Total Latency (Q→C)"),
                                            ("d2c", "Device Latency (D→C)")]):
        ax = fig.add_subplot(gs[0, col])
        r_data = lats[f"{prefix}_read"]
        w_data = lats[f"{prefix}_write"]

        if len(r_data) > 0:
            ax.hist(r_data, bins=100, alpha=0.6, color=COLORS["read"],
                    label=f"Read (n={len(r_data):,})", density=True)
        if len(w_data) > 0:
            ax.hist(w_data, bins=100, alpha=0.6, color=COLORS["write"],
                    label=f"Write (n={len(w_data):,})", density=True)
        setup_ax(ax, title, "Latency (µs)", "Density")
        ax.legend(fontsize=9)
        # Clip x-axis to p99.5 for readability
        all_data = np.concatenate([d for d in [r_data, w_data] if len(d) > 0]) if (len(r_data) + len(w_data)) > 0 else np.array([0])
        if len(all_data) > 10:
            ax.set_xlim(0, np.percentile(all_data, 99.5))

    # ── Row 2: CDF for Q2C and D2C ──
    for col, (prefix, title) in enumerate([("q2c", "Q→C Latency CDF"),
                                            ("d2c", "D→C Latency CDF")]):
        ax = fig.add_subplot(gs[1, col])
        for data, label, color in [
            (lats[f"{prefix}_read"], "Read", COLORS["read"]),
            (lats[f"{prefix}_write"], "Write", COLORS["write"]),
        ]:
            if len(data) > 0:
                sorted_d = np.sort(data)
                cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                ax.plot(sorted_d, cdf * 100, linewidth=1.5, color=color, label=label)
                # Mark p50, p99
                for pct, ls in [(50, "--"), (99, ":")]:
                    val = np.percentile(data, pct)
                    ax.axvline(val, color=color, linestyle=ls, alpha=0.5, linewidth=0.8)
                    ax.annotate(f"p{pct}={val:.0f}µs", xy=(val, pct),
                                fontsize=7, color=color, alpha=0.8)
        setup_ax(ax, title, "Latency (µs)", "Percentile (%)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
        if len(all_data) > 10:
            ax.set_xlim(0, np.percentile(all_data, 99.9))

    # ── Row 3: Latency over time ──
    ax = fig.add_subplot(gs[2, :])
    times, p50_r, p99_r, p50_w, p99_w = analyzer.compute_latency_over_time()
    mask_r = p50_r > 0
    mask_w = p50_w > 0

    if mask_r.any():
        ax.plot(times[mask_r], p50_r[mask_r], linewidth=1, color=COLORS["read"],
                label="Read p50")
        ax.plot(times[mask_r], p99_r[mask_r], linewidth=1, color=COLORS["read"],
                linestyle="--", alpha=0.6, label="Read p99")
    if mask_w.any():
        ax.plot(times[mask_w], p50_w[mask_w], linewidth=1, color=COLORS["write"],
                label="Write p50")
        ax.plot(times[mask_w], p99_w[mask_w], linewidth=1, color=COLORS["write"],
                linestyle="--", alpha=0.6, label="Write p99")
    setup_ax(ax, "Latency Over Time (Device Latency)", "Time (s)", "Latency (µs)")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Latency Analysis", fontsize=15, fontweight="bold", y=1.01)
    path = os.path.join(output_dir, "05_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_dashboard(analyzer: BlktraceAnalyzer, output_dir: str):
    """Single-page overview dashboard combining key metrics."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Queue depth
    ax = fig.add_subplot(gs[0, 0])
    bt, avg_t, avg_r, avg_w = analyzer.compute_queue_depth_bucketed()
    ax.plot(bt, avg_t, linewidth=1, color=COLORS["total"])
    ax.fill_between(bt, avg_t, alpha=0.2, color=COLORS["total"])
    setup_ax(ax, "Queue Depth", "Time (s)", "QD")

    # 2. Throughput
    ax = fig.add_subplot(gs[0, 1])
    bt, r_mbps, w_mbps = analyzer.compute_throughput()
    ax.plot(bt, r_mbps, color=COLORS["read"], linewidth=1, label="Read")
    ax.plot(bt, w_mbps, color=COLORS["write"], linewidth=1, label="Write")
    setup_ax(ax, "Throughput", "Time (s)", "MB/s")
    ax.legend(fontsize=8)

    # 3. IOPS
    ax = fig.add_subplot(gs[0, 2])
    bt, r_iops, w_iops = analyzer.compute_iops()
    ax.plot(bt, r_iops, color=COLORS["read"], linewidth=1, label="Read")
    ax.plot(bt, w_iops, color=COLORS["write"], linewidth=1, label="Write")
    setup_ax(ax, "IOPS", "Time (s)", "IOPS")
    ax.legend(fontsize=8)

    # 4. LBA heatmap
    ax = fig.add_subplot(gs[1, :2])
    heatmap_all, _, _, max_sector = analyzer.compute_lba_heatmap()
    if isinstance(heatmap_all, np.ndarray) and heatmap_all.max() > 0:
        im = ax.imshow(heatmap_all, aspect="auto", origin="lower",
                        norm=LogNorm(vmin=max(1, heatmap_all[heatmap_all > 0].min()),
                                     vmax=heatmap_all.max()),
                        cmap="inferno", interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.8)
    setup_ax(ax, "LBA Heatmap", "Time →", "LBA →")

    # 5. I/O size pie
    ax = fig.add_subplot(gs[1, 2])
    r_sizes, w_sizes, all_sizes = analyzer.compute_io_sizes()
    if all_sizes:
        size_buckets = {"≤4K": 0, "4K–16K": 0, "16K–128K": 0, "128K–1M": 0, ">1M": 0}
        for s in all_sizes:
            if s <= 4 * KB:
                size_buckets["≤4K"] += 1
            elif s <= 16 * KB:
                size_buckets["4K–16K"] += 1
            elif s <= 128 * KB:
                size_buckets["16K–128K"] += 1
            elif s <= MB:
                size_buckets["128K–1M"] += 1
            else:
                size_buckets[">1M"] += 1
        non_zero = {k: v for k, v in size_buckets.items() if v > 0}
        if non_zero:
            ax.pie(non_zero.values(), labels=non_zero.keys(), autopct="%1.1f%%",
                   textprops={"fontsize": 9})
    ax.set_title("I/O Size Distribution", fontsize=13, fontweight="bold")

    # 6. Latency CDF
    ax = fig.add_subplot(gs[2, 0])
    lats = analyzer.compute_latency_distributions()
    for data, label, color in [
        (lats["d2c_read"], "Read D2C", COLORS["read"]),
        (lats["d2c_write"], "Write D2C", COLORS["write"]),
    ]:
        if len(data) > 0:
            sorted_d = np.sort(data)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d, cdf * 100, linewidth=1.5, color=color, label=label)
    setup_ax(ax, "Latency CDF (D→C)", "Latency (µs)", "Percentile")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # 7. Latency over time
    ax = fig.add_subplot(gs[2, 1:])
    times, p50_r, p99_r, p50_w, p99_w = analyzer.compute_latency_over_time()
    mask_r = p50_r > 0
    mask_w = p50_w > 0
    if mask_r.any():
        ax.plot(times[mask_r], p50_r[mask_r], color=COLORS["read"], linewidth=1, label="Read p50")
        ax.fill_between(times[mask_r], p50_r[mask_r], p99_r[mask_r],
                         color=COLORS["read"], alpha=0.15)
    if mask_w.any():
        ax.plot(times[mask_w], p50_w[mask_w], color=COLORS["write"], linewidth=1, label="Write p50")
        ax.fill_between(times[mask_w], p50_w[mask_w], p99_w[mask_w],
                         color=COLORS["write"], alpha=0.15)
    setup_ax(ax, "Latency Over Time (shaded=p50→p99)", "Time (s)", "Latency (µs)")
    ax.legend(fontsize=8)

    fig.suptitle("blktrace Analysis Dashboard", fontsize=18, fontweight="bold", y=1.01)
    path = os.path.join(output_dir, "00_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="blktrace Analyzer — Queue Depth, LBA Hotspots, I/O Sizes, "
                    "Throughput & Latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture and analyze (requires root):
  sudo blktrace -d /dev/nvme0n1 -w 30 -o nvme_trace
  blkparse -i nvme_trace -o trace.txt
  python3 blktrace_analyzer.py trace.txt

  # Analyze with custom settings:
  python3 blktrace_analyzer.py trace.txt --time-bucket 0.05 --lba-bins 512

  # Pipe directly from blkparse:
  blkparse -i nvme_trace -o - | python3 blktrace_analyzer.py /dev/stdin
        """,
    )
    parser.add_argument("trace_file", help="Path to blkparse text output file")
    parser.add_argument("--time-bucket", type=float, default=0.1,
                        help="Time bucket size in seconds (default: 0.1)")
    parser.add_argument("--lba-bins", type=int, default=256,
                        help="Number of LBA bins for heatmap (default: 256)")
    parser.add_argument("--output-dir", default="./blktrace_results",
                        help="Output directory for charts (default: ./blktrace_results)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip the combined dashboard chart")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary only, skip chart generation")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  blktrace Analyzer")
    print(f"{'='*72}")
    print(f"\n  Loading: {args.trace_file}")

    # Load & parse
    events = load_trace(args.trace_file)
    if not events:
        print("\n  ERROR: No parseable events found. Ensure the input is blkparse text output.")
        print("  Run:  blkparse -i <blktrace_output> -o trace.txt")
        sys.exit(1)

    # Analyze
    analyzer = BlktraceAnalyzer(events, time_bucket=args.time_bucket,
                                 lba_bins=args.lba_bins)
    analyzer.print_summary()

    if args.summary_only:
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n  Generating charts in: {args.output_dir}/\n")

    # Generate all charts
    if not args.no_dashboard:
        plot_combined_dashboard(analyzer, args.output_dir)
    plot_queue_depth(analyzer, args.output_dir)
    plot_lba_hotspots(analyzer, args.output_dir)
    plot_io_sizes(analyzer, args.output_dir)
    plot_throughput(analyzer, args.output_dir)
    plot_latency(analyzer, args.output_dir)
    plot_queued_ios_distribution(analyzer, args.output_dir)

    print(f"\n  All charts saved to: {args.output_dir}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            fpath = os.path.join(args.output_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
