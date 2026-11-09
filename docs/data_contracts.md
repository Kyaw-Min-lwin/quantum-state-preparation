# Demo 1 data contracts

## Quantum representation

- Two noiseless qubits
- Computational basis ordering: |q1 q0>
- Statevector order: |00>, |01>, |10>, |11>
- Quantum arithmetic uses complex128
- State validation rejects incorrect shape, non-finite amplitudes,
  and non-unit norm. It does not silently normalise input
- State validation is implemented in section B

## Actions

Action IDs are stable:

| ID | Gate | Qubits |
| --- | --- | --- |
| 0 | H | q0 |
| 1 | H | q1 |
| 2 | S | q0 |
| 3 | S | q1 |
| 4 | T | q0 |
| 5 | T | q1 |
| 6 | CX | control q0, target q1 |
| 7 | CX | control q1, target q0 |

ACTION_SPECS defines the ordering
The configuration action labels must agree with it
B1 builds Action objects from these specifications

## Target

A Target contains:
- target_id
- statevector
- reference_action_ids

Reference circuits are evaluation metadata
They must never enter the observation or policy input
They demonstrate reachability, not proven optimality

## Observation

The model receives a one-dimensional float32 array of length 17

| Slice | Content |
| --- | --- |
| 0:4 | Real parts of current state |
| 4:8 | Imaginary parts of current state |
| 8:12 | Real parts of target state |
| 12:16 | Imaginary parts of target state |
| 16 | Remaining gates divided by maximum gates |

Current and target states are independently canonicalised for global phase
before encoding, using the rule specified in section B

At reset, remaining fraction is 1
After eight applied gates, remaining fraction is 0

Neither target ID nor reference circuit is included in the observation

## Transition

A Transition records:
- observation before the action
- action ID
- reward for that action
- observation after the action
- terminated
- truncated

Stored observations are independent copies

Success and exhaustion of the eight-gate task budget are terminal:
terminated=True, truncated=False

A separate external interruption is truncation:
terminated=False, truncated=True

The DQN target masks future value using terminated
Truncation may bootstrap only from a valid final next observation,
never from the next episode's reset observation

## Rewards and already-solved targets

Every applied gate earns -0.05
First reaching success adds 10, giving 9.95 for that successful step

An already-solved target ends before any action
It has zero gates, zero transitions, and total reward 0

## Episode trace

The trace includes the target ID, initial fidelity, ordered step records,
and an explicit end reason

Each step records:
- cumulative gate count, starting at 1
- action ID
- fidelity after the gate
- reward for that gate

End reasons:
- already_solved
- success
- budget_exhausted
- external_truncation

Success on gate eight is success, not budget_exhausted

Failures remain in evaluation results

## Checkpoints

schema_version starts at 1

Required checkpoint information:
- Resolved configuration and ordered action catalogue
- Online and target network states
- Optimiser state
- Replay contents and sampling position/state
- Environment-step and gradient-step counters
- Completed-episode count and current epsilon
- Random-generator states
- Environment state and rollout state
- Code revision, package versions, device, and other provenance

Random-generator states include the target sampler, exploration,
replay sampling, and relevant Python/PyTorch CPU/CUDA states

For a mid-episode checkpoint, environment and rollout state include
the current state, target, remaining budget, action history, and
partial trace

Evaluation-only loading can use a subset of these fields
Full training resumption requires the complete checkpoint
Exact training reproduction across different hardware or software
versions is not guaranteed