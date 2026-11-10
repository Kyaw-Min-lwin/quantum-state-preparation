from pathlib import Path
import numpy as np
import pytest
from qprep.config import load_config
from qprep.encoding import encode_observation
from qprep.quantum import apply_gate, fidelity, zero_state
from qprep.targets import build_target_catalogue, sample_target

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo1.json"


@pytest.mark.parametrize("target_id", ["A", "B", "C", "D", "E", "F", "zero"])
def test_reference_actions_reach_target_and_encode_result(target_id):
    config = load_config(CONFIG_PATH)
    catalogue = build_target_catalogue()
    rng = np.random.default_rng(config["seeds"][0])
    target = sample_target(rng, catalogue, target_id=target_id)

    state = zero_state(config["n_qubits"])
    remaining = config["max_gates"]

    for action_id in target.reference_action_ids:
        assert remaining > 0
        state = apply_gate(state, action_id)
        remaining -= 1

    assert fidelity(state, target.state) >= config["success_fidelity"]

    obs = encode_observation(state, target.state, remaining, config["max_gates"])

    assert obs.shape == (config["observation_dim"],)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    # If the reference sequence succeeds, the two state blocks agree
    np.testing.assert_allclose(obs[:8], obs[8:16], atol=1e-7, rtol=0)
    assert obs[-1] == pytest.approx(remaining / config["max_gates"])

    if target_id == "zero":
        assert target.reference_action_ids == ()
        assert remaining == config["max_gates"]
