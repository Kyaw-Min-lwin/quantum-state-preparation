from numbers import Integral
import numpy as np
from qprep.contracts import Observation
from qprep.quantum import canonicalise_phase


def encode_observation(
    current_state, target_state, remaining_gates: int, max_gates: int
) -> Observation:
    # Encode the current state, target state, and remaining gate budget
    for count in (remaining_gates, max_gates):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("Gate counts must be integers.")

    if max_gates <= 0 or not 0 <= remaining_gates <= max_gates:
        raise ValueError(
            "Gate budget must satisfy 0 <= remaining <= max, with max > 0."
        )

    current = canonicalise_phase(current_state)
    target = canonicalise_phase(target_state)
    observation = np.concatenate(
        (
            current.real,
            current.imag,
            target.real,
            target.imag,
            [remaining_gates / max_gates],
        )
    )
    return observation.astype(np.float32)
