# SOMA v3.1.1 Benchmark

## C Native vs Python Interpreter

| Program        | C Native (avg) | Python  | Speedup |
|----------------|---------------|---------|---------|
| hello_agent    | 5ms           | 3,444ms | ~689×   |
| swarm_cluster  | 10ms          | —       | —       |
| online_learner | 5ms           | —       | —       |

Platform: WSL2 / Ubuntu, gcc -O3 -march=native -ffast-math
Date: 2026-02-20
Tag: v3.1.1
