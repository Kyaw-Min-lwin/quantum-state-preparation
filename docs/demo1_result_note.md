# Demo 1 result note

**Project:** Reinforcement learning for quantum state preparation  
**Scope:** Two noiseless qubits, starting at $|00\rangle$. Eight legal actions place H, S, T or a directed CX gate. Each attempt uses up to eight gates and succeeds at fidelity of at least 0.999999.

Three independently seeded DQNs trained for 100,000 environment steps each. The final checkpoint from each seed was selected in advance. Each checkpoint was evaluated greedily once per target on the six training fixtures. The random policy used 100 attempts on each fixture under the same task rules.

| Policy | Successful attempts | Success rate |
| --- | ---: | ---: |
| DQN | 18/18 | 100.0% |
| Random | 142/600 | 23.7% |

The DQN exceeded the random baseline by 76.3 percentage points and met the planned learning criterion. Seeds 0, 1 and 2 each solved all six targets. Every final fidelity rounded to 1.000000, and all 18 saved evaluations matched after a clean-kernel reload. The random baseline recorded 458 unsuccessful attempts.

The DQN used 6.0 gates per successful attempt on average. Seventeen of 18 circuits were longer than their known generating references. For the Bell target E, seeds 0, 1 and 2 used 8, 8 and 7 gates respectively, against a two-gate reference. Training took 330.1, 588.3 and 587.2 seconds respectively. Median measured action-selection time per evaluated attempt was 1.51 ms in the evaluation runtime.

These results show reliable preparation on the two-qubit training suite. The reference circuits give known constructions for comparing gate counts. The strongest next question is whether a policy trained on some reachable target states can solve independently selected, unseen target states within the same budget.

