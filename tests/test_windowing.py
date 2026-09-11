"""Unit tests for the windowing layer."""
import numpy as np
import pytest

from ats.data.windowing import concat_windowsets, make_windows


def test_window_count_and_timestamps():
    """20 s at 25 Hz, 64-sample windows, 32-sample hop -> 14 windows."""
    n = 500
    values = np.random.default_rng(0).normal(size=(n, 6))
    ws = make_windows(values, np.ones(n, bool), t0=0.0, fs=25.0,
                      window_samples=64, hop_samples=32)
    assert len(ws) == 14
    assert ws.values.shape == (14, 64, 6)
    assert ws.t_start[0] == pytest.approx(0.0)
    assert ws.t_start[1] == pytest.approx(32 / 25.0)
    assert ws.t_end[0] == pytest.approx(64 / 25.0)
    assert ws.t_end[-1] <= n / 25.0 + 1e-9


def test_t0_offset_puts_windows_on_the_recording_axis():
    values = np.zeros((500, 6))
    ws = make_windows(values, np.ones(500, bool), t0=3600.0, fs=25.0)
    assert ws.t_start[0] == pytest.approx(3600.0)
    assert ws.t_start[-1] == pytest.approx(3600.0 + 13 * 32 / 25.0)


def test_validity_threshold_flags_partially_missing_windows():
    n = 500
    valid = np.ones(n, bool)
    valid[100:140] = False          # 40 missing samples
    ws = make_windows(np.zeros((n, 6)), valid, t0=0.0, fs=25.0,
                      window_samples=64, hop_samples=32, min_validity=0.90)
    touched = np.flatnonzero(~ws.valid)
    assert touched.size > 0
    # Windows far from the dropout are untouched.
    assert ws.valid[0] and ws.valid[-1]
    assert ws.validity_fraction[touched].max() < 0.90


def test_filter_valid_keeps_arrays_aligned():
    valid = np.ones(500, bool)
    valid[:100] = False
    ws = make_windows(np.zeros((500, 6)), valid, t0=0.0, fs=25.0)
    kept = ws.filter_valid()
    assert len(kept) == int(ws.valid.sum())
    assert kept.t_start.shape[0] == kept.values.shape[0]
    assert kept.valid.all()


def test_short_block_yields_no_windows():
    ws = make_windows(np.zeros((10, 6)), np.ones(10, bool), t0=0.0)
    assert len(ws) == 0


def test_concat_preserves_order_and_time():
    a = make_windows(np.zeros((500, 6)), np.ones(500, bool), t0=0.0)
    b = make_windows(np.zeros((500, 6)), np.ones(500, bool), t0=60.0)
    merged = concat_windowsets([a, b])
    assert len(merged) == len(a) + len(b)
    assert merged.t_start[len(a)] == pytest.approx(60.0)
    assert np.all(np.diff(merged.t_start) > 0)
