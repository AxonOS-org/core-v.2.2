# EDF Scheduler

## Theorem 5.2 (Liu & Layland, 1973)

A set of n periodic tasks with D_i = T_i is schedulable on a uniprocessor under EDF iff U ≤ 1.

## Conservative Ceiling

AxonOS uses U_max = 0.25 to account for interrupt jitter and cache effects.

## Busy Period Bound

L = Σ_j ceil(L / T_j) * C_j

For AxonOS: L = 972 µs [L2].

## Deadline Slack

S_1 = D_1 - R_1 = 4000 - 972 = 3028 µs [L1].

## References

- Liu, C. L., & Layland, J. W. (1973). "Scheduling algorithms for multiprogramming in a hard-real-time environment." JACM 20(1), 46–61.
- Buttazzo, G. C. (2011). *Hard Real-Time Computing Systems* (3rd ed.). Springer. Section 5.5.1.
- Yermakou, D. (2026). "AxonOS: Analytical Real-Time Schedulability." arXiv preprint.
