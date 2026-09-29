import numpy as np
import pytest

from geopulse import baseline

from .conftest import CONTROL, EVENT, synthetic_inputs


@pytest.mark.parametrize("task", ["flood", "wildfire", "vegetation"])
def test_event_mapped_lookalike_not(task):
    p = baseline.predict(synthetic_inputs(task).arrays, task)
    assert p["target"][EVENT].mean() > 0.9
    assert p["target"][CONTROL].mean() < 0.15  # permanent water / fuel-less darkening / senescing grass
    assert p["target"][0:40, 0:40].mean() < 0.05  # untouched background
    assert p["uncertainty"][EVENT].mean() < 0.3


@pytest.mark.parametrize(
    ("task", "sensor"),
    [("flood", "s1"), ("flood", "s2"), ("wildfire", "s2"), ("vegetation", "s1"), ("vegetation", "s2")],
)
def test_each_sensor_alone_still_maps_the_event(task, sensor):
    p = baseline.predict(synthetic_inputs(task, sensors=(sensor,)).arrays, task)
    assert p["target"][EVENT].mean() > 0.8


def test_sar_only_burn_is_never_confident():
    p = baseline.predict(synthetic_inputs("wildfire", sensors=("s1",)).arrays, "wildfire")
    assert p["target"][EVENT].mean() > 0.5  # canopy volume loss is visible in VH...
    assert np.nanmin(p["uncertainty"]) >= baseline.SAR_ONLY_BURN_UNCERTAINTY  # ...but flagged as weak evidence


def test_sar_fills_cloud_gaps_in_burn_map():
    arrays = synthetic_inputs("wildfire").arrays
    arrays["s2_post"][:, 60:80, 70:90] = np.nan  # smoke/cloud over part of the burn
    p = baseline.predict(arrays, "wildfire")
    assert np.isfinite(p["target"][60:80, 70:90]).all() and p["target"][60:80, 70:90].mean() > 0.5


def test_burn_severity_classes():
    sev = baseline.predict(synthetic_inputs("wildfire").arrays, "wildfire")["severity"]
    assert (sev[EVENT] == 4).mean() > 0.95  # dNBR ≈ 0.88 -> high
    assert (sev[0:40, 0:40] == 0).mean() > 0.95


def test_wind_roughened_floodwater_found_by_sar_change():
    arrays = synthetic_inputs(sensors=("s1",)).arrays
    arrays["s1_post"][(0, *EVENT)] = -14.0  # rough water: above the -18 dB threshold...
    arrays["s1_pre"][(0, *EVENT)] = -8.0  # ...but a 6 dB drop from before
    assert baseline.predict(arrays)["target"][EVENT].mean() > 0.8


def test_weak_labels_are_consensus_only():
    arrays = synthetic_inputs().arrays
    lab = baseline.weak_labels(arrays)
    assert (lab[0][EVENT] == 1).mean() > 0.95
    assert (lab[0][CONTROL] == 0).all()
    arrays["s2_post"][:, :20, :20] = np.nan  # clouds: no consensus -> ignored
    assert (baseline.weak_labels(arrays)[0][:20, :20] == baseline.IGNORE).all()


def test_weak_labels_need_both_sensors():
    lab = baseline.weak_labels(synthetic_inputs(sensors=("s1",)).arrays)
    assert (lab == baseline.IGNORE).all()


def test_generic_change_sees_vegetation_loss():
    ch = baseline.weak_change(synthetic_inputs("vegetation").arrays)  # conservative: only confident pixels
    assert (ch[EVENT] == 1).mean() > 0.1 and not (ch[EVENT] == 0).any() and (ch[0:40, 0:40] == 0).mean() > 0.9


def test_sparse_shrubland_can_burn():
    """Regression: a 0.2 NDVI fuel gate hid burned sagebrush steppe (pre-fire NDVI ~0.17)."""
    from .conftest import SCENARIOS

    SCENARIOS["sagebrush"] = ("sagebrush", ("sagebrush", "sagebrush_burned"), ("bare", "bare_dark"))
    try:
        p = baseline.predict(synthetic_inputs("sagebrush").arrays, "wildfire")
    finally:
        del SCENARIOS["sagebrush"]
    assert p["target"][EVENT].mean() > 0.8 and p["target"][CONTROL].mean() < 0.15
