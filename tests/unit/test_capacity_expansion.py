from types import SimpleNamespace

import numpy as np

from firm_ce.optimisation.capacity_expansion import (
    _active_vintages,
    _build_effective_year_bounds,
    _effective_build_bounds,
)


def test_effective_build_bounds_subtracts_active_endogenous_capacity():
    effective_min, effective_max = _effective_build_bounds(2.0, 10.0, 4.0)

    assert effective_min == 0.0
    assert effective_max == 6.0


def test_effective_build_bounds_clamps_when_active_capacity_exceeds_gross_ceiling():
    effective_min, effective_max = _effective_build_bounds(1.0, 3.0, 5.0)

    assert effective_min == 0.0
    assert effective_max == 0.0


def test_active_vintages_retire_when_lifetime_is_reached():
    vintages = [
        (2025, 4.0, 2),
        (2024, 1.0, 1),
        (2026, 2.0, 0),
    ]

    active_2026, total_2026 = _active_vintages(vintages, 2026)
    active_2027, total_2027 = _active_vintages(vintages, 2027)

    assert total_2026 == 6.0
    assert active_2026 == [(2025, 4.0, 2), (2026, 2.0, 0)]
    assert total_2027 == 2.0
    assert active_2027 == [(2026, 2.0, 0)]


def test_build_effective_year_bounds_respects_asset_blocks_and_durations():
    fleet = SimpleNamespace(
        generators={
            0: SimpleNamespace(order=0),
            1: SimpleNamespace(order=1),
        },
        storages={
            0: SimpleNamespace(order=0, duration={0: 0}),
            1: SimpleNamespace(order=1, duration={0: 4}),
        },
    )
    network = SimpleNamespace(
        major_lines={
            0: SimpleNamespace(order=0),
        }
    )

    raw_lower_bounds = np.array([2.0, 1.0, 3.0, 4.0, 5.0, 0.0, 6.0], dtype=np.float64)
    raw_upper_bounds = np.array([10.0, 8.0, 9.0, 7.0, 11.0, 0.0, 12.0], dtype=np.float64)

    effective_lower_bounds, effective_upper_bounds = _build_effective_year_bounds(
        fleet,
        network,
        0,
        raw_lower_bounds,
        raw_upper_bounds,
        np.array([4.0, 0.5], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([2.0, 0.0], dtype=np.float64),
        np.array([3.0], dtype=np.float64),
    )

    np.testing.assert_allclose(effective_lower_bounds, np.array([0.0, 0.5, 2.0, 2.0, 3.0, 0.0, 3.0]))
    np.testing.assert_allclose(effective_upper_bounds, np.array([6.0, 7.5, 8.0, 5.0, 9.0, 0.0, 9.0]))
