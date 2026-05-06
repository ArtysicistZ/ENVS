# Diagnosis: SFT init vs ARPO step_35 (86 trainable, n=1, T=0)

**Caveat:** n=1 deterministic per model. Counts reflect modal policy, not full success-rate over 8 rollouts.

## 1. Task-level buckets
- SFT solved: **44/86** (51.2%)
- ARPO step_35 solved: **47/86** (54.7%)
- Net coverage change: **+3 tasks**

| Bucket | Count | % |
|---|---|---|
| SFT_win | 14 | 16.3% |
| ARPO_win | 17 | 19.8% |
| both_solve | 30 | 34.9% |
| both_fail | 25 | 29.1% |

## 2. Trajectory length (success-conditioned)
| Model | success rate | avg steps | median | %>=10 | %hit max(15) |
|---|---|---|---|---|---|
| SFT init | 51.2% | 11.11 | 11.0 | 68.2% | 22.7% |
| ARPO step_35 | 54.7% | 9.81 | 9 | 46.8% | 19.1% |

Length buckets among **successful** trajectories:
| Model | short(1-5) | med(6-9) | long(10-15) |
|---|---|---|---|
| SFT init | 0 (0%) | 14 (32%) | 30 (68%) |
| ARPO step_35 | 4 (9%) | 21 (45%) | 22 (47%) |

## 3. Failure modes
| Mode | SFT (n=42) | ARPO (n=39) |
|---|---|---|
| max_steps_no_progress | 32 (76%) | 35 (90%) |
| premature_termination | 10 (24%) | 4 (10%) |

Avg failed-traj length: SFT 13.79, ARPO 14.38

## 5. Action-type distribution (all steps, all tasks)
| action | SFT count | SFT % | ARPO count | ARPO % | Δ(pp) |
|---|---|---|---|---|---|
| click | 668 | 62.5% | 581 | 56.8% | -5.7 |
| wait | 97 | 9.1% | 110 | 10.8% | +1.7 |
| hotkey | 84 | 7.9% | 98 | 9.6% | +1.7 |
| type | 65 | 6.1% | 64 | 6.3% | +0.2 |
| drag | 37 | 3.5% | 56 | 5.5% | +2.0 |
| left_double | 39 | 3.7% | 46 | 4.5% | +0.8 |
| finished | 43 | 4.0% | 40 | 3.9% | -0.1 |
| scroll | 22 | 2.1% | 14 | 1.4% | -0.7 |
| right_single | 12 | 1.1% | 8 | 0.8% | -0.3 |
| call_user | 1 | 0.1% | 5 | 0.5% | +0.4 |

- Repeated-action rate (consecutive identical Action): SFT **17.1%** (168/982), ARPO **20.3%** (190/936)
- First FINISH step (avg): SFT -, ARPO - | n_finish: SFT=0, ARPO=0