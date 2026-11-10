import numpy as np
import pytest

from qprep.encoding import encode_observation
from qprep.quantum import zero_state
from qprep.targets import build_target_catalogue


def test_observation_layout_and_target_phase():
    targets = build_target_catalogue()
    current = zero_state(2)

    obs_e = encode_observation(current, targets["E"].state, 6, 8)
    obs_f = encode_observation(current, targets["F"].state, 6, 8)

    expected_f = [
        1,
        0,
        0,
        0,  # current real
        0,
        0,
        0,
        0,  # current imaginary
        2**-0.5,
        0,
        0,
        0,  # target real
        0,
        0,
        0,
        2**-0.5,  # target imaginary
        0.75,
    ]

    assert obs_f.shape == (17,)
    assert obs_f.dtype == np.float32
    assert np.isfinite(obs_f).all()
    np.testing.assert_allclose(obs_f, expected_f, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(obs_e[:8], obs_f[:8])
    assert not np.allclose(obs_e[8:16], obs_f[8:16])


@pytest.mark.parametrize("remaining, expected", [(0, 0.0), (8, 1.0)])
def test_observation_budget_endpoints(remaining, expected):
    state = zero_state(2)
    assert encode_observation(state, state, remaining, 8)[-1] == expected


@pytest.mark.parametrize("remaining, maximum", [(-1, 8), (9, 8), (0, 0)])
def test_invalid_budget_is_rejected(remaining, maximum):
    state = zero_state(2)
    with pytest.raises(ValueError, match="budget"):
        encode_observation(state, state, remaining, maximum)
