# EDF Scheduler

## Theorem 5.2 (Liu-Layland)

A set of n periodic tasks with D_i = T_i is schedulable on a uniprocessor under EDF iff U ≤ 1.

## Conservative Ceiling

AxonOS uses U_max = 0.25 to account for interrupt jitter and cache effects.

## Busy Period Bound

L = Σ_j ceil(L / T_j) * C_j

For AxonOS: L = 972 µs [L2].

## Deadline Slack

S_1 = D_1 - R_1 = 4000 - 972 = 3028 µs [L1].
