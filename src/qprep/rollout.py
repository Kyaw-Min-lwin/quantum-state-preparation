from qprep.contracts import EpisodeTrace, TraceStep


def rollout(env, policy, target_id=None, seed=None) -> EpisodeTrace:
    # Run one episode and record its actions, rewards, and fidelities
    observation, reset_info = env.reset(
        seed=seed, options={"target_id": target_id}
    )
    steps = []
    end_reason = reset_info["end_reason"]

    while end_reason is None:
        action_id = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action_id)

        steps.append(
            TraceStep(
                gate_count=info["gate_count"],
                action_id=int(action_id),
                fidelity=info["fidelity"],
                reward=reward,
            )
        )

        if terminated:
            end_reason = info["end_reason"]
        elif truncated:
            end_reason = "external_truncation"

    return EpisodeTrace(
        target_id=reset_info["target_id"],
        initial_fidelity=reset_info["initial_fidelity"],
        steps=tuple(steps),
        end_reason=end_reason,
    )