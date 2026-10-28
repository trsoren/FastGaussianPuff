"""A source outside the sensor box must still reach a downwind sensor."""
import numpy as np
import pandas as pd

from FastGaussianPuff import GaussianPuff


def test_source_outside_sensor_bounds():
    # The source is north of both sensors; the first sensor is downwind.
    model = GaussianPuff(
        obs_dt=60,
        sim_dt=1,
        puff_dt=1,
        simulation_start=pd.Timestamp("2000-01-01T12:00:00Z"),
        simulation_end=pd.Timestamp("2000-01-01T12:05:00Z"),
        time_zone="UTC",
        source_coordinates=np.array([[0.0, 0.0, 1.0]]),
        sensor_coordinates=np.array([[50.0, -5.0, 1.0], [-50.0, -50.0, 1.0]]),
        emission_rates=np.array([3.6]),
        wind_speeds=np.full(6, 2.0),
        wind_directions=np.full(6, 269.99),
        using_sensors=True,
    )
    concentrations = np.asarray(model.simulate())

    assert concentrations.shape == (6, 2)
    assert np.isfinite(concentrations).all()
    assert (concentrations >= 0).all()
    # Allow travel time, then require a clear signal at the downwind sensor.
    assert (concentrations[1:, 0] > 0.01).all()


if __name__ == "__main__":
    test_source_outside_sensor_bounds()
