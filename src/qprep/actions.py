from numbers import Integral

from qprep.contracts import ACTION_SPECS, Action


def build_action_catalogue() -> tuple[Action, ...]:
    # Return the eight Demo 1 actions in their fixed ID order
    return tuple(
        Action(
            action_id=action_id,
            gate=gate,
            qubits=qubits,
        )
        for action_id, (gate, qubits) in enumerate(ACTION_SPECS)
    )


ACTION_CATALOGUE = build_action_catalogue()


def get_action(action_id: int) -> Action:
    """Resolve a legal integer ID without silently converting inputs."""
    if isinstance(action_id, bool) or not isinstance(action_id, Integral):
        raise TypeError("action_id must be an integer, excluding bool.")

    action_id = int(action_id)

    if not 0 <= action_id < len(ACTION_CATALOGUE):
        raise ValueError(
            f"action_id must be between 0 and "
            f"{len(ACTION_CATALOGUE) - 1}; received {action_id}."
        )

    return ACTION_CATALOGUE[action_id]


def serialise_action_catalogue() -> list[dict]:
    """Return a JSON-compatible catalogue for run records/checkpoints."""
    return [
        {
            "action_id": action.action_id,
            "gate": action.gate,
            "qubits": list(action.qubits),
        }
        for action in ACTION_CATALOGUE
    ]
