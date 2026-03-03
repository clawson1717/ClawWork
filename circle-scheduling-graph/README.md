# CIRCLE-Scheduling-Graph (CSG)

A self-evolving framework for **Flexible Job Shop Scheduling** (FJSP) that uses **Heterogeneous Graph Networks** to model production constraints (buffers, kitting) while implementing a **Prospective Validation Protocol** (CIRCLE) to bridge the gap between theoretical makespan metrics and real-world deployment performance.

## Overview
CSG combines three cutting-edge techniques from recent ArXiv research:
1.  **Heterogeneous Graph Networks:** To model complex dependencies among machines, operations, and buffers in a real production line.
2.  **CIRCLE (Validation Framework):** A six-stage protocol to translate qualitative stakeholder concerns into measurable quantitative rewards.
3.  **Process-Aware Evaluation (DARE-bench):** To measure "Instruction Fidelity" and "Process Integrity" during complex scheduling tasks.

## 12-Step Roadmap

1.  **Step 1: Project Scaffold** - `src/`, `tests/`, `requirements.txt`
2.  **Step 2: FJSP Environment Model** - Limited buffers & kitting simulation
3.  **Step 3: Heterogeneous Graph Constructor** - Machine, Operation, Buffer nodes
4.  **Step 4: Heterogeneous Graph DRL Agent** - Global state modeling (HGT)
5.  **Step 5: CIRCLE Context Capture** - Translating "Stakeholder Concerns" to signals
6.  **Step 6: Process-Aware Reward Function** - DARE-derived verifiable rewards
7.  **Step 7: CIRCLE Validation Loop** - Field testing and user variability
8.  **Step 8: Dataset Preparation** - Synthetic + Real-world production data
9.  **Step 9: Training Pipeline** - PPO/DQN training on HGT agent
10. **Step 10: Fidelity vs. Efficiency Benchmark** - Comparative analysis
11. **Step 11: CLI & Production Visualization** - ASCII scheduling board
12. **Step 12: Final Documentation** - Implementation write-up & results

---
*Created by Clawson (🦞) via arxiv-project-ideator.*
