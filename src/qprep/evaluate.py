from dataclasses import asdict
from math import isclose
from qiskit.quantum_info import Statevector
from qprep.circuits import rebuild_circuit
from qprep.dqn import select_action
from qprep.environment import QuantumEnv
from qprep.quantum import fidelity
from qprep.rollout import rollout
from time import perf_counter

def evaluate_policy(network, config, rng, *, measure_inference=False):
    # Evaluate one greedy rollout per fixture and verify each circuit
    env = QuantumEnv(config)
    evaluation_seed = int(rng.integers(0, 2**32))
    results = []

    inference_seconds = 0.0

    def greedy_policy(observation):
        nonlocal inference_seconds

        if not measure_inference:
            return select_action(network, observation, epsilon=0, rng=rng)

        started = perf_counter()
        action = select_action(network, observation, epsilon=0, rng=rng)
        inference_seconds += perf_counter() - started
        return action

    was_training = network.training
    network.eval()

    try:
        for target_id in config["target_ids"]:
            inference_seconds = 0.0
            trace = rollout(
                env,
                greedy_policy,
                target_id=target_id,
                seed=evaluation_seed,
            )

            actions = [step.action_id for step in trace.steps]
            circuit = rebuild_circuit(actions)
            state = Statevector.from_instruction(circuit).data
            score = fidelity(state, env.catalogue[target_id].state)
            success = score >= config["success_fidelity"]

            if not isclose(score, trace.final_fidelity, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Evaluation replay differs for {target_id}.")
            if success != trace.success:
                raise ValueError(f"Evaluation outcome differs for {target_id}.")

            results.append(
                {
                    "target_id": target_id,
                    "success": success,
                    "final_fidelity": score,
                    "gate_count": trace.gate_count,
                    "depth": circuit.depth(),
                    "cx_count": int(circuit.count_ops().get("cx", 0)),
                    "trace": asdict(trace),
                }
            )
            if measure_inference:
                results[-1]["inference_seconds"] = inference_seconds
    finally:
        network.train(was_training)
    return results
