"""Regression for a source outside a non-degenerate sensor bounding box."""
import numpy as np
import pandas as pd

from FastGaussianPuff import GaussianPuff


def test_outside_source_reaches_downwind_sensor():
    # The source is north of the sensor box (y ranges from -40 to -4 m).
    # Nearly westerly wind carries methane to the first sensor, 40 m east
    # and only 4 m crosswind. Continuous emission must not yield all zeros.
    # Before the bounds fix this case returns zeros on macOS; other builds
    # can instead raise an allocation error from an invalid plume travel time.
    model = GaussianPuff(
        obs_dt=60,
        sim_dt=1,
        puff_dt=1,
        output_dt=60,
        simulation_start=pd.Timestamp("2026-02-01T12:00:00Z"),
        simulation_end=pd.Timestamp("2026-02-01T12:05:00Z"),
        time_zone="America/New_York",
        source_coordinates=np.array([[0.0, 0.0, 2.0]]),
        emission_rates=np.array([3.6]),  # 1 g/s
        wind_speeds=np.full(6, 2.25),
        wind_directions=np.full(6, 269.99),
        using_sensors=True,
        sensor_coordinates=np.array([[40.0, -4.0, 2.0], [-40.0, -40.0, 2.0]]),
        skip_low_wind=True,
        low_wind_cutoff=0.1,
        quiet=True,
    )
    concentrations = np.asarray(model.simulate())

    assert concentrations.shape == (6, 2)
    assert np.isfinite(concentrations).all()
    assert (concentrations >= 0).all()
    # After the first minute the nearby downwind sensor should have a clear
    # signal. This loose lower bound rejects zeros/underflow, not rounding error;
    # it does not prescribe a particular implementation's exact prediction.
    assert (concentrations[1:, 0] > 0.01).all(), "Missing methane at the downwind sensor"


if __name__ == "__main__":
    test_outside_source_reaches_downwind_sensor()
