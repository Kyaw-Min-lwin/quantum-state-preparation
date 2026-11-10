from qprep.contracts import Target
from qprep.quantum import apply_gate, zero_state

REFERENCE_ACTIONS = {
    "A": (0,),  # H(q0)
    "B": (1,),  # H(q1)
    "C": (0, 1),  # H(q0), H(q1)
    "D": (0, 4),  # H(q0), T(q0)
    "E": (0, 6),  # H(q0), CX(q0, q1)
    "F": (0, 2, 6),  # H(q0), S(q0), CX(q0, q1)
}


def build_target_catalogue() -> dict[str, Target]:
    # generate the Demo 1 targets from their reference gate sequences
    catalogue = {}
    for target_id, action_ids in REFERENCE_ACTIONS.items():
        state = zero_state(2)
        for action_id in action_ids:
            state = apply_gate(state, action_id)

        catalogue[target_id] = Target(
            target_id=target_id,
            state=state,
            reference_action_ids=action_ids,
        )

    return catalogue


def sample_target(rng, catalogue: dict[str, Target], target_id=None) -> Target:
    # sample a training target or return a specific evaluation target
    if target_id is None:
        target_id = str(rng.choice(list(REFERENCE_ACTIONS)))

    if target_id == "zero":
        return Target(
            target_id="zero",
            state=zero_state(2),
            reference_action_ids=(),
        )

    if target_id not in catalogue:
        raise ValueError(f"Unknown target ID: {target_id!r}")

    return catalogue[target_id]
