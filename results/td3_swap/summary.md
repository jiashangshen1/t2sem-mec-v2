# DDPG vs TD3 long-agent swap (v4.3 protocol)

seeds=[42, 123, 456, 789, 1011], eval_eps=30, episodes=300, device=cpu

| algo | SLA % | std (pp) | delay (s) | energy (J) | critical % | interactive % | best_effort % |
|---|---:|---:|---:|---:|---:|---:|---:|
| ddpg | 90.21 | 2.09 | 0.138 | 19326 | 83.73 | 86.49 | 100.00 |
| td3 | 90.79 | 2.37 | 0.124 | 13928 | 76.37 | 95.86 | 99.97 |
