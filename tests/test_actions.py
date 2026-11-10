import json
import numpy as np
import pytest

from qprep.actions import (
    build_action_catalogue,
    get_action,
    serialise_action_catalogue,
)


def test_catalogue_has_exact_demo1_mapping():
    catalogue = build_action_catalogue()

    actual = [(action.action_id, action.gate, action.qubits) for action in catalogue]

    expected = [
        (0, "h", (0,)),
        (1, "h", (1,)),
        (2, "s", (0,)),
        (3, "s", (1,)),
        (4, "t", (0,)),
        (5, "t", (1,)),
        (6, "cx", (0, 1)),
        (7, "cx", (1, 0)),
    ]

    assert actual == expected
    assert len({(a.gate, a.qubits) for a in catalogue}) == 8


@pytest.mark.parametrize("action_id", range(8))
def test_lookup_returns_the_requested_action(action_id):
    action = get_action(action_id)

    assert action.action_id == action_id
    assert action == build_action_catalogue()[action_id]


def test_numpy_integer_is_accepted():
    action = get_action(np.int64(6))

    assert action.gate == "cx"
    assert action.qubits == (0, 1)


@pytest.mark.parametrize("action_id", [-1, 8, 100])
def test_out_of_range_ids_are_rejected(action_id):
    with pytest.raises(ValueError, match="between"):
        get_action(action_id)


@pytest.mark.parametrize("action_id", [True, False, 1.0, "1", None])
def test_non_integer_ids_are_rejected(action_id):
    with pytest.raises(TypeError, match="integer"):
        get_action(action_id)


def test_catalogue_survives_json_round_trip():
    records = serialise_action_catalogue()
    restored = json.loads(json.dumps(records))

    assert restored == records
    assert restored[6] == {
        "action_id": 6,
        "gate": "cx",
        "qubits": [0, 1],
    }
    assert restored[7]["qubits"] == [1, 0]
