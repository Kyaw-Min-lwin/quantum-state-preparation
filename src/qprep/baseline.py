import json
from dataclasses import asdict
from math import isclose
import numpy as np
from qiskit.quantum_info import Statevector
from qprep.circuits import rebuild_circuit
from qprep.environment import QuantumEnv
from qprep.quantum import fidelity
from qprep.rollout import rollout
from qprep.runtime import write_json
from qprep.targets import build_target_catalogue


def run_random_baseline(run):
    # run the configured number of random episodes for each target
    config = run.config
    env = QuantumEnv(config)
    seed_rng = run.rngs["baseline"]
    records = []

    for target_id in config["target_ids"]:
        for attempt in range(1, config["random_attempts_per_target"] + 1):
            policy_seed = int(seed_rng.integers(0, 2**63))
            policy_rng = np.random.default_rng(policy_seed)

            def random_policy(_observation):
                return int(policy_rng.integers(0, env.action_space.n))

            trace = rollout(env, random_policy, target_id=target_id)

            record = asdict(trace)
            record["attempt"] = attempt
            record["policy_seed"] = policy_seed
            record["success"] = trace.success
            record["final_fidelity"] = trace.final_fidelity
            records.append(record)

    path = run.run_dir / "random_baseline_traces.json"
    write_json(path, records)
    return path

def verify_and_summarise_baseline(path, config):
    # replay saved circuits and report success rates by target
    records = json.loads(path.read_text(encoding="utf-8"))
    catalogue = build_target_catalogue()
    summary = {
        target_id: {"attempts": 0, "successes": 0}
        for target_id in config["target_ids"]
    }

    for record in records:
        target_id = record["target_id"]
        action_ids = [step["action_id"] for step in record["steps"]]
        circuit = rebuild_circuit(action_ids)
        state = Statevector.from_instruction(circuit).data
        score = fidelity(state, catalogue[target_id].state)
        label = f"{target_id}, attempt {record['attempt']}"

        if not isclose(score, record["final_fidelity"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Replayed fidelity differs for {label}.")

        success = score >= config["success_fidelity"]
        if success != record["success"]:
            raise ValueError(f"Replayed success flag differs for {label}.")

        summary[target_id]["attempts"] += 1
        summary[target_id]["successes"] += int(success)

    expected = config["random_attempts_per_target"]
    for target_id, result in summary.items():
        if result["attempts"] != expected:
            raise ValueError(f"Expected {expected} attempts for {target_id}.")
        result["success_rate"] = result["successes"] / result["attempts"]

    return summary