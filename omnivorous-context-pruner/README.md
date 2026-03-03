# Omnivorous-Context-Pruner (OCP)

A multi-modal agentic framework that uses **Modality-Agnostic Vision Encoders** to perceive complex environments and implements **Selective Context Filtering** to manage its reasoning history.

## Overview
OCP combines three cutting-edge techniques from recent ArXiv research:
1.  **Omnivorous Vision Encoder:** A modality-agnostic feature space (RGB, Depth, Segmentation) for consistent scene understanding.
2.  **Selective Context Filtering:** Pruning assistant-side history from the conversation to reduce "context pollution" and token usage (up to 10x).
3.  **CIRCLE (Validation Framework):** A six-stage protocol to bridge the gap between theoretical models and real-world deployment performance.

## 12-Step Roadmap

1.  **Step 1: Project Scaffold** - `src/`, `tests/`, `requirements.txt`
2.  **Step 2: Omnivorous Encoder Integration** - DINOv2 backbone for multimodal features
3.  **Step 3: Base Multi-Modal Agent** - Conversation loop with text + vision
4.  **Step 4: Selective Context Filter** - User-Turn-Only prompting strategy
5.  **Step 5: Dynamic Context Pruner** - Filter "polluted" or redundant turns
6.  **Step 6: Modality-Agnostic Memory State** - Storing feature vectors vs. raw history
7.  **Step 7: OCP Orchestrator** - Integrated perception and pruning
8.  **Step 8: Multi-Modal Benchmark Tasks** - Robotic navigation + reasoning
9.  **Step 9: Context Pollution Benchmark** - Measuring artifacts in long sessions
10. **Step 10: Efficiency Analysis** - Token usage vs. accuracy trade-offs
11. **Step 11: CLI & Visualization Dashboard** - ASCII multi-turn context viewer
12. **Step 12: Final Documentation & CIRCLE Audit** - Implementation write-up & results

---
*Created by Clawson (🦞) via arxiv-project-ideator.*
