"""Unit tests for the resampling layer."""
import numpy as np
import pytest

from ats.data.resample import align_streams, resample_to_grid


def test_uniform_signal_is_preserved():
    """A clean 40 Hz sinusoid resampled to 25 Hz keeps its shape."""
    fs_in, freq, dur = 40.0, 2.0, 20.0
    t = np.arange(int(dur * fs_in)) / fs_in
    x = np.sin(2 * np.pi * freq * t)[:, None]
    out = resample_to_grid(t, x, fs=25.0, duration=dur, t0=0.0)
    assert out.n_samples == 500
    assert out.coverage == pytest.approx(1.0)
    expected = np.sin(2 * np.pi * freq * out.times)
    # Linear interpolation of a 2 Hz sinusoid at 40 Hz: error is small but real.
    assert np.max(np.abs(out.values[:, 0] - expected)) < 0.05


def test_dropout_is_marked_invalid_not_interpolated_over():
    """The regression this guards: a 2 s gap must not become 2 s of smooth motion."""
    fs_in, dur = 40.0, 20.0
    t = np.arange(int(dur * fs_in)) / fs_in
    x = np.ones((t.size, 3))
    keep = ~((t >= 8.0) & (t < 10.0))
    out = resample_to_grid(t[keep], x[keep], fs=25.0, duration=dur, t0=0.0,
                           max_gap=0.08)
    gap = (out.times >= 8.1) & (out.times < 9.9)
    assert not out.valid[gap].any()          # nothing in the gap is trusted
    assert out.valid[out.times < 7.5].all()  # everything outside it is
    assert 0.85 < out.coverage < 0.95


def test_irregular_timestamps_are_sorted_and_deduplicated():
    t = np.array([0.0, 0.1, 0.1, 0.05, 0.2])
    x = np.arange(5, dtype=float)[:, None]
    out = resample_to_grid(t, x, fs=25.0, duration=0.2, t0=0.0, max_gap=0.1)
    assert out.n_samples == 5
    assert np.isfinite(out.values).all()


def test_short_and_empty_inputs_do_not_crash():
    out = resample_to_grid(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]),
                           fs=25.0, duration=1.0, t0=1.0)
    assert out.n_samples == 25
    assert out.valid[:3].all() and not out.valid[-1]

    empty = resample_to_grid(
        np.array([]), np.empty((0, 3)), fs=25.0, duration=20.0, t0=0.0
    )
    assert empty.values.shape == (0, 3)
    assert empty.valid.shape == (0,)
    assert empty.coverage == 0.0


def test_align_streams_requires_both_modalities_valid():
    t = np.arange(800) / 40.0
    acc = resample_to_grid(t, np.ones((800, 3)), fs=25.0, duration=20.0, t0=0.0)
    keep = ~((t >= 5.0) & (t < 6.0))
    gyro = resample_to_grid(t[keep], np.zeros((keep.sum(), 3)), fs=25.0,
                            duration=20.0, t0=0.0)
    values, valid, t0 = align_streams(acc, gyro)
    assert values.shape == (500, 6)
    assert t0 == 0.0
    assert not valid[(np.arange(500) / 25.0 >= 5.1) & (np.arange(500) / 25.0 < 5.9)].any()


def test_mismatched_grids_are_rejected():
    t = np.arange(100) / 40.0
    a = resample_to_grid(t, np.ones((100, 3)), fs=25.0, duration=2.0, t0=0.0)
    b = resample_to_grid(t, np.ones((100, 3)), fs=25.0, duration=2.0, t0=60.0)
    with pytest.raises(ValueError):
        align_streams(a, b)
