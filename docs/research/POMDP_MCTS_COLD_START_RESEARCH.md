# POMDP + MCTS for SFT Cold-Start in GUI Agent Training: Research Survey

> **Research Goal**: Investigate the feasibility and novelty of using POMDP + MCTS to generate diverse SFT cold-start data for GUI agent training, as an alternative to (a) direct LLM labeling for cold-start, or (b) using MCTS during the RL phase.

---

## Table of Contents

1. [Executive Summary & Innovation Analysis](#1-executive-summary--innovation-analysis)
2. [MCTS + LLM: Foundations and Key Papers](#2-mcts--llm-foundations-and-key-papers)
3. [GUI Agent Training Pipelines](#3-gui-agent-training-pipelines)
4. [POMDP Formulation for GUI Agents](#4-pomdp-formulation-for-gui-agents)
5. [Cold-Start Methods: Current Landscape](#5-cold-start-methods-current-landscape)
6. [Data Diversity and Its Impact on RL](#6-data-diversity-and-its-impact-on-rl)
7. [Proposed Method: POMDP-MCTS Cold-Start](#7-proposed-method-pomdp-mcts-cold-start)
8. [Key Technical Challenges](#8-key-technical-challenges)
9. [Related Work Positioning](#9-related-work-positioning)

---

## 1. Executive Summary & Innovation Analysis

### The Proposed Idea

Use POMDP + MCTS during the **SFT cold-start phase** (before RL training) to generate diverse, high-quality training trajectories for GUI agents.

### Why This Is Novel

| Aspect | Previous Approaches | Our Proposal |
|--------|-------------------|--------------|
| **Cold-start data** | Direct LLM annotation (GPT-4 labeling) | MCTS-guided exploration generates diverse trajectories |
| **MCTS usage** | Applied during RL phase (expensive, slow) | Applied during cold-start only (one-time cost) |
| **Data diversity** | Limited by LLM's own distribution | Tree search explores multiple action paths |
| **Dataset contribution** | Proprietary or limited | Public dataset as byproduct |

### Three Key Benefits

1. **Diversity for RL**: MCTS-generated cold-start data covers more of the action space, enabling better exploration during RL rollouts
2. **Dataset contribution**: The MCTS-generated trajectories form a reusable, high-quality dataset
3. **Efficiency**: Avoids the prohibitive cost of running MCTS during online RL training

---

## 2. MCTS + LLM: Foundations and Key Papers

### 2.1 rStar-Math (Guan et al., Jan 2025)

**"rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking"**

- **Key idea**: Uses MCTS to enable small LLMs (e.g., 7B) to achieve math reasoning performance comparable to OpenAI o1
- **Method**: Self-evolved deep thinking via MCTS — the LLM generates candidate reasoning steps, MCTS selects the best paths using a process reward model (PRM), and successful trajectories are used for iterative self-training
- **Training loop**: Generate → Search (MCTS) → Filter → SFT → Repeat
- **Relevance**: **Directly demonstrates MCTS can generate high-quality SFT data**. Their approach is iterative (MCTS during each round of self-training), while we propose using MCTS only at cold-start. rStar achieves remarkable results showing MCTS-generated data is superior to simple sampling

### 2.2 ReST-MCTS* (Zhang et al., Jun 2024)

**"ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search"**

- **Key idea**: Combines Reinforced Self-Training (ReST) with MCTS* (a variant of MCTS using process rewards)
- **Method**: Uses tree search guided by a process reward model to generate high-quality reasoning traces for self-training SFT data
- **Key insight**: MCTS* generates traces that are significantly better than standard sampling or best-of-N, because the tree search can backtrack from failed paths and explore alternatives
- **Results**: Outperforms standard ReST and best-of-N rejection sampling
- **Relevance**: **Core precedent for using MCTS to generate SFT training data**. The key difference from our proposal is domain (math reasoning vs. GUI agents) and the additional POMDP formulation we introduce

### 2.3 V-STaR (Hosseini et al., Feb 2024)

**"V-STaR: Training Verifiers for Self-Taught Reasoners"**

- **Key idea**: Extends STaR by training a verifier using DPO on both correct and incorrect self-generated solutions
- **Method**: Self-improvement loop where incorrect solutions are not discarded but used to train a discriminative verifier
- **Relevance**: Shows the value of leveraging **failed trajectories** (not just successful ones) — important for MCTS where many explored branches fail. In GUI agent MCTS, failed paths can train a verifier/critic

### 2.4 Value-Guided MCTS / PPO-MCTS (Liu et al., Sep 2023)

**"Don't throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding"**

- **Key idea**: Reuses the value network from PPO training as an MCTS guide during decoding
- **Method**: Token-level MCTS where the value model (trained during PPO) guides tree expansion
- **Relevance**: Shows how value models can be integrated with MCTS. For GUI agents, we could train a lightweight value model on initial data, then use it to guide MCTS exploration during cold-start

### 2.5 AlphaLLM (2024)

**"AlphaLLM: Training LLMs with MCTS Self-Play"**

- **Key idea**: Adapts AlphaGo-style self-play training to LLMs
- **Method**: MCTS with a learned value function and policy, iteratively improving both through self-play
- **Training pipeline**: Policy network (LLM) → MCTS search → collect trajectories → train policy + value → repeat
- **Relevance**: Demonstrates the full loop of MCTS → data collection → training, but applied during RL. Our innovation is doing this only at cold-start

### 2.6 LLM-MCTS (Zhao et al., 2023)

**"Large Language Models as Commonsense Knowledge for Large-Scale Task Planning"**

- **Key idea**: Uses LLM as a world model / heuristic within MCTS for task planning
- **Method**: LLM provides action proposals and state evaluations; MCTS uses these to search over plans
- **Relevance**: Directly relevant — GUI agent interaction IS task planning. LLM can serve as the prior policy in MCTS, proposing likely GUI actions, while MCTS explores alternatives

### 2.7 Marco-o1 (2024)

**"Marco-o1: Towards Open Reasoning Models"**

- **Key idea**: Open-source attempt to replicate o1-style reasoning using MCTS
- **Method**: MCTS-based reasoning with reflection and self-evaluation at each step
- **Relevance**: Shows MCTS can be applied to general reasoning, not just math. Validates the approach of using MCTS to explore reasoning paths

### 2.8 Math-Shepherd (Wang et al., Dec 2023)

**"Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"**

- **Key idea**: Process reward model (PRM) that scores individual reasoning steps
- **Method**: Automatic construction of step-level supervision data without manual annotation; used for both reranking and PPO
- **Results**: Significant improvements (Mistral-7B: 77.9% → 84.1% on GSM8K)
- **Relevance**: PRM concept is essential for MCTS in GUI agents — we need step-level (action-level) rewards to guide tree search. This paper shows PRMs can be constructed automatically

### 2.9 MCTS-DPO (Xie et al., 2024) [arXiv: 2405.00451]

**"Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning"**

- **Key idea**: Uses MCTS to generate **step-level preference data** (not just instance-level), combining outcome validation and self-evaluation
- **Method**: MCTS explores reasoning paths → DPO refines the policy using step-level preferences from the tree
- **Results**: GSM8K 81.8% (+5.9%), MATH 34.7% (+5.8%) over Mistral-7B
- **Relevance**: Shows MCTS can generate preference data for DPO training. In our GUI agent context, MCTS at cold-start generates both SFT data AND preference pairs for subsequent DPO/RL

### 2.10 RAP (Hao et al., 2023, EMNLP) [arXiv: 2305.14992]

**"Reasoning with Language Model is Planning with World Model"**

- **Key idea**: Repurposes the LLM as both a reasoning agent AND a world model. Uses MCTS to explore a reasoning tree with LLM simulating state transitions
- **Results**: LLaMA-33B surpassed GPT-4 in plan generation by 33% relative improvement
- **Relevance**: Foundational paper showing LLM can serve dual roles (policy + world model) in MCTS — directly applicable to GUI agents where LLM predicts both actions and state transitions

### 2.11 AlphaMath Almost Zero (Chen et al., 2024) [arXiv: 2405.03553]

**"AlphaMath Almost Zero: Process Supervision without Process"**

- **Key idea**: Uses MCTS to **autonomously generate process supervision signals without any human or GPT-4 annotations**
- **Method**: MCTS generates step-level supervision signals; value model trained from MCTS statistics
- **Results**: Comparable to or exceeding SOTA on both in-domain and out-of-domain math benchmarks
- **Relevance**: **Demonstrates zero-annotation MCTS cold-start** — directly applicable to GUI agent cold-start where we want to avoid expensive human annotation

### 2.12 OmegaPRM (Luo et al., 2024) [arXiv: 2406.06592]

**"OmegaPRM: Improve Mathematical Reasoning by Automated Process Supervision"**

- **Key idea**: Uses MCTS with binary search to automatically locate reasoning errors and generate balanced process supervision data
- **Scale**: **1.5M+ process supervision annotations without human labeling**
- **Results**: Gemini Pro: 51% → 69.4% on MATH500, 86.4% → 93.6% on GSM8K
- **Relevance**: Shows MCTS can generate massive-scale supervision data automatically — provides evidence for scalability of MCTS cold-start

### 2.13 "Distilling System 2 into System 1" (Yu et al., 2024) [arXiv: 2407.06023]

**Key idea**: Transfers outputs from expensive inference-time reasoning (System 2, including MCTS search) back into base model (System 1) via self-supervised distillation
- **Relevance**: **Defines the exact paradigm we propose** — use expensive search (MCTS) to generate data, then distill into a cheaper model via SFT. Our contribution is applying this specifically to GUI agents at cold-start with POMDP formulation

### 2.14 "Compute-Optimal Sampling" (Bansal et al., 2024) [arXiv: 2408.16737]

**"Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling"**

- **Key finding**: Weaker/cheaper models generate **more diverse** training data than expensive models. Higher coverage compensates for higher false positive rates
- **Relevance**: Suggests using a smaller LLM as the MCTS policy could produce MORE diverse cold-start data than using a stronger model. Cost-efficiency argument for MCTS cold-start

### 2.15 SEEA-R1 (Tian et al., 2025)

**"SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents"**

- **Key idea**: Integrates MCTS into RL fine-tuning for embodied agents
- **Method**: Tree-GRPO (Tree-based Group Relative Policy Optimization) using MCTS for denser learning signals + Multi-modal Generative Reward Model (MGRM)
- **Results**: 85.07% on ALFWorld (text), surpassing GPT-4o
- **Relevance**: **Highly relevant** — applies MCTS to agent training, but in the RL phase. This is exactly the approach we want to avoid (MCTS in RL is expensive). We can contrast: SEEA-R1 uses MCTS during RL → expensive; we use MCTS during cold-start → efficient

---

## 3. GUI Agent Training Pipelines

### 3.1 AgentQ (Putta et al., 2024) [arXiv: 2408.07199]

**"Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents"**

- **Key idea**: Combines guided MCTS with self-critique and iterative off-policy DPO for training web agents
- **MCTS Details**:
  - Each node = agent state (history summary + DOM tree). Edges = executed actions
  - At each node, samples K possible actions from base LLM
  - Selection via UCB1 (Q-value + exploration bonus based on visit counts)
  - Rollouts use current policy until terminal state; reward = 1 (success) or 0 (failure)
  - Values backpropagate bottom-up
- **Self-critique**: Same base LLM serves as critic, ranking generated actions by perceived utility (process-level supervision)
- **DPO pipeline**: Step-level DPO constructs preference pairs from branches where value differences exceed threshold. Q-value = α × empirical MCTS value + (1-α) × AI ranking value. Replay buffer stores trajectories. Each iteration: sample tasks → MCTS search → collect trajectories → construct pairs → DPO optimize
- **Results**: Llama-3 70B on OpenTable (real website): 18.6% → 81.7% (without search) → 95.4% (with MCTS at test time). Outperformed GPT-4 (62.6%)
- **Critical analysis for our work**: AgentQ uses MCTS **during every DPO iteration**, meaning each training round requires expensive MCTS rollouts in the live environment. **Our proposal moves MCTS to cold-start**, making it a one-time cost. AgentQ is our closest competitor and key paper to differentiate from

### 3.2 DigiRL (2024)

**"DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning"**

- **Key idea**: Autonomous RL for device control agents in real Android environments
- **Method**:
  1. **Cold-start**: SFT on offline demonstrations (human-collected or LLM-labeled)
  2. **Online RL**: Autonomous RL with reward from task completion, using filtered behavior cloning + offline-to-online RL
- **Cold-start approach**: Uses offline demonstrations — limited diversity, relies on human annotation or LLM labeling
- **Relevance**: Exemplifies the standard cold-start problem. Their SFT phase uses static demonstrations, which lack diversity. MCTS cold-start could generate much more diverse initialization data

### 3.3 WebRL (2024)

**"WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning"**

- **Key idea**: Self-evolving curriculum for online RL training of web agents
- **Method**:
  1. **SFT initialization**: Train on existing web interaction data
  2. **Online RL**: Self-evolving curriculum — automatically adjusts task difficulty based on agent performance
  3. **Reward model**: Outcome-based reward (task success/failure)
- **Cold-start**: Standard SFT on available demonstrations
- **Relevance**: Shows importance of curriculum in RL. MCTS cold-start data could provide a better initialization for the curriculum, covering more task types and interaction patterns

### 3.4 UI-TARS / UI-TARS-2 (2024-2025)

**"UI-TARS: Pioneering Automated GUI Interaction with Native Agents"**
**"UI-TARS-2: Advancing GUI Agent with Multi-Turn Reinforcement Learning"**

- **UI-TARS**: Large-scale SFT on GUI interaction data (screenshots + actions), achieving strong zero-shot performance
- **UI-TARS-2**: Extends with multi-turn RL training
  - Uses group-relative policy optimization
  - Trains on multi-step interaction trajectories
  - Cold-start: extensive SFT on curated GUI datasets
- **Cold-start approach**: Massive data collection effort — web scraping, human annotation, synthetic generation via GPT-4V
- **Relevance**: Represents the "brute force data collection" approach to cold-start. MCTS could achieve similar diversity with less manual effort

### 3.5 WebPilot (Zhang et al., Aug 2024)

**"WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration"**

- **Key idea**: Uses customized MCTS for complex web environments at **inference time**
- **Method**:
  1. **Global Optimization**: High-level planning via task decomposition
  2. **Local Optimization**: MCTS for each subtask, handling uncertainty and incomplete information
- **Results**: 93% relative improvement over concurrent tree search methods on WebArena
- **Relevance**: **Demonstrates MCTS works well for GUI/web agent environments**. However, WebPilot uses MCTS only at inference time (expensive per-query). We propose using it at training time for data generation — amortizing the cost

### 3.6 OS-ATLAS (2024)

**"OS-ATLAS: A Foundation Action Model for Generalist GUI Agents"**

- **Key idea**: Foundation model for cross-platform GUI interaction
- **Method**: Large-scale pre-training on GUI screenshots with action annotations across multiple OS platforms
- **Cold-start**: Extensive data collection from multiple platforms (Windows, macOS, Linux, Android, iOS, Web)
- **Relevance**: Data-centric approach to cold-start; MCTS could augment this with more diverse exploration trajectories

### 3.7 GUI-Shift (2025)

**"GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning"**

- **Key idea**: Self-supervised RL for GUI agents, reducing need for labeled data
- **Method**: Uses environment feedback as self-supervised signal for RL training
- **Relevance**: Complements our approach — MCTS cold-start generates SFT data, then GUI-Shift-style self-supervised RL can follow

### 3.8 InSTA (2025)

**"InSTA: Towards Internet-Scale Training For Agents"**

- **Key idea**: Scaling agent training to internet-scale using diverse web environments
- **Method**: Large-scale data collection and training pipeline for web agents
- **Relevance**: Demonstrates the appetite for large, diverse training data in agent training — MCTS cold-start could be a scalable way to generate such data

### 3.9 Tree Search for Language Model Agents (Koh et al., 2024)

**"Tree Search for Language Model Agents"**
- **arXiv**: 2407.01476
- **Key idea**: Best-first tree search operating within the **actual environment** (not a simulator)
- **Results**: GPT-4o on VisualWebArena: 39.7% relative improvement; WebArena: 28.0% relative improvement. **Scales with test-time compute**
- **Relevance**: **Directly demonstrates tree search works in real interactive web environments where rollbacks require environment snapshots** — validates our approach's feasibility

### 3.10 LATS (Zhou et al., 2023)

**"Language Agent Tree Search (LATS): Unifying Reasoning, Acting, and Planning"**
- **arXiv**: 2310.04406
- **Key idea**: Unifies LM reasoning, acting, and planning via MCTS with LM-powered value functions and self-reflection
- **Results**: 92.7% pass@1 on HumanEval with GPT-4; competitive with fine-tuning on web navigation using GPT-3.5
- **Relevance**: Foundational framework for MCTS + LLM agents. Demonstrates LLM can serve as both policy and value function in tree search

### 3.11 ExACT / R-MCTS (Yu et al., 2024)

**"ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning"**
- **arXiv**: 2410.02052
- **Key idea**: Reflective MCTS (R-MCTS) incorporating contrastive reflection from past interactions + multi-agent debate for state evaluation
- **Key innovation**: **Exploratory Learning distills search capabilities into the model**, reducing compute at deployment while retaining search benefits
- **Results**: GPT-4o achieves 6-30% relative improvement on VisualWebArena
- **Relevance**: **Directly relevant** — shows how to distill MCTS search knowledge into SFT data. This is essentially what our cold-start approach does: run MCTS → collect trajectories → SFT

### 3.12 Plan-MCTS (Zhang et al., 2026)

**"Plan-MCTS: Plan Exploration for Action Exploitation in Web Navigation"**
- **arXiv**: 2602.14083
- **Key idea**: Searches in **plan space** rather than action space. Converts sparse action space into Dense Plan Tree for efficient exploration
- **Method**: Abstracted Semantic History for precise state awareness + dual validation and subplan repair
- **Relevance**: **State-of-the-art for MCTS in web navigation**. Addresses the key challenge that most actions in web/GUI environments are invalid, making naive MCTS wasteful. Plan-level search is much more efficient

### 3.13 ProAct (Yu et al., 2026)

**"ProAct: Agentic Lookahead in Interactive Environments"**
- **arXiv**: 2602.05327
- **Key idea**: Grounded LookAhead Distillation (GLAD) converts search trees into causal chains for supervised training
- **Results**: 4B model outperforms all open-source baselines via MCTS → SFT distillation
- **Relevance**: **Strongest precedent for our approach** — explicitly distills tree search into SFT training data for interactive environments. Difference: ProAct does this iteratively; we propose doing it at cold-start only

### 3.14 TSR (Djuhera et al., 2026)

**"TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents"**
- **arXiv**: 2602.11767
- **Key idea**: Performs lightweight tree-style search **during RL training rollouts** (not inference). Compatible with PPO and GRPO
- **Results**: Up to 15% gains on Sokoban, FrozenLake, and WebShop
- **Relevance**: Shifts search from inference to training, but still during RL. Our approach goes further — shifting search to cold-start phase before RL even begins

### 3.15 EvoCUA (Xue et al., 2026) [arXiv: 2601.15876]

**"EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience"**

- **Key idea**: Self-sustaining evolutionary cycle with synthetic task generation, thousands of async sandbox rollouts, iterative policy updates
- **Results**: **56.7% on OSWorld** (surpassing UI-TARS-2's 53.1%) — current SOTA
- **Relevance**: Strong baseline for comparison. Uses evolutionary data generation but not structured tree search. MCTS could replace or augment their random rollout strategy

### 3.16 OS-Genesis (Sun et al., 2024, ACL 2025) [arXiv: 2412.19723]

**"OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis"**

- **Key idea**: Reverses conventional data collection — agents first **explore environments** and perform interactions, then **retrospectively derive tasks** from successful trajectories
- **Method**: Trajectory reward model ensures quality. No manual task specification needed
- **Relevance**: Alternative cold-start strategy — explore first, label later. MCTS could improve the "explore" phase by being more systematic than random exploration

### 3.17 AgentTrek (Xu et al., 2024, ICLR 2025 Spotlight) [arXiv: 2412.09605]

**"AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials"**

- **Key idea**: Synthesizes trajectories from publicly available web tutorials. VLM agent executes tutorial instructions in live environments; VLM evaluator verifies correctness
- **Cost**: $0.55 per high-quality trajectory, fully automated
- **Results**: SOTA on WebArena, ScreenSpot Web, Multimodal Mind2Web
- **Relevance**: Tutorial-guided data generation is complementary to MCTS. MCTS explores diverse strategies per task; AgentTrek provides diverse tasks from tutorials

### 3.18 Key Observation: MDP vs POMDP in Existing Work

**Critical finding from survey**: Most existing GUI agent papers formulate the problem as an **MDP** (not POMDP):
- **DigiRL**: "finite-horizon MDP with states = last two screenshots"
- **AgentQ**: Implicit MDP with DOM tree as state
- **WebRL**: MDP with screenshot observations
- **UI-TARS**: MDP with last N observations as state

**No existing GUI agent training paper uses an explicit POMDP formulation.** The partial observability is handled implicitly by limiting input to recent screenshots. This confirms the novelty of our POMDP formulation.

---

## 4. POMDP Formulation for GUI Agents

### 4.1 Why GUI Interaction is a POMDP

A GUI agent interaction naturally fits the POMDP (Partially Observable Markov Decision Process) framework:

| POMDP Component | GUI Agent Mapping |
|----------------|-------------------|
| **State (S)** | Full system state (all processes, memory, file system, network state) |
| **Observation (O)** | Current screenshot (+ optionally accessibility tree / HTML DOM) |
| **Action (A)** | Click, type, scroll, keyboard shortcut, etc. |
| **Transition (T)** | System response to action (deterministic given full state, but stochastic from agent's perspective) |
| **Reward (R)** | Task completion signal (binary or graded) |
| **Belief (B)** | Agent's estimate of the true system state given observation history |

**Key partial observability aspects:**
- Agent sees only the current screen, not background processes
- System state includes hidden elements (minimized windows, background apps, network state)
- Previous actions' effects may not be immediately visible
- Same visual observation can correspond to different underlying states

### 4.2 POMCP: Partially Observable Monte Carlo Planning (Silver & Veness, 2010)

The foundational algorithm for MCTS in POMDPs:

- **Key idea**: Extends UCT (Upper Confidence bounds applied to Trees) to POMDPs by maintaining a belief (particle set) at each node
- **Method**:
  1. Maintain a tree where nodes represent action-observation histories
  2. Use particle filters to approximate beliefs at each node
  3. Simulate forward using particles to estimate values
  4. UCB1 for action selection balancing exploration vs exploitation
- **Complexity**: O(|A|^d × |O|^d) per planning step, where d is depth
- **Relevance**: **Core algorithm for our POMDP-MCTS cold-start**. However, standard POMCP assumes a known transition model — in GUI environments, we need the LLM as an approximate world model or use real environment interaction

### 4.3 BA-POMCP (Katt et al., 2018)

**"Learning in POMDPs with Monte Carlo Tree Search"**

- **Key idea**: Extends POMCP to Bayes-Adaptive POMDPs where the transition model is unknown and must be learned
- **Method**: Maintains beliefs over both states AND model parameters, enabling simultaneous learning and planning
- **Relevance**: In GUI environments, the "world model" (how the system responds to actions) is unknown and must be learned from interaction — BA-POMCP provides the theoretical foundation

### 4.4 NeoPlanner (Paul, 2023)

**"Sequential Planning in Large Partially Observable Environments guided by LLMs"**

- **Key idea**: Combines state space search with LLM queries for planning in large POMDPs
- **Method**: Reward signals direct exploration; LLM generates action proposals when random exploration is needed; learnings captured as entity relationships
- **Results**: 124% improvement over previous best on Scienceworld
- **Relevance**: **Directly demonstrates LLM + tree search in partially observable interactive environments**. Closest to our proposed approach in spirit

### 4.5 LOOP (Chen et al., 2025)

**"LOOP: Reinforcement Learning for Long-Horizon Interactive LLM Agents"**
- **arXiv**: 2502.01600
- **Key idea**: **Directly formalizes interactive digital agents as POMDPs**. Introduces a data- and memory-efficient PPO variant (no value network, single LLM copy)
- **Results**: 32B agent trained with LOOP outperforms OpenAI o1 by 9 percentage points on AppWorld
- **Key findings**: Agents learn to consult documentation, avoid assumptions, reduce hallucinations, and recover from failures
- **Relevance**: **First reported RL application using explicit POMDP formulation for stateful, multi-domain LLM agents**. Validates that POMDP framing leads to better agent behavior

### 4.6 From Words to Actions (He et al., 2024, ICML)

**"From Words to Actions: Theoretical Underpinnings of LLM-Driven Autonomous Systems"**

- **Key idea**: Theoretical hierarchical RL framework where an LLM Planner navigates POMDPs through language-based subgoal generation
- **Key theorem**: Naively following Bayesian aggregated imitation learning (BAIL) subgoals produces **linear regret**; epsilon-greedy exploration achieves sublinear regret
- **Relevance**: **Provides theoretical proof that imitation-based cold-start (LLM labeling) is suboptimal in POMDPs** — exploration (e.g., MCTS) is theoretically necessary for good performance

### 4.7 PIANIST (Light et al., 2024, NeurIPS Workshop)

**"PIANIST: Learning Partially Observable World Models with LLMs"**

- **Key idea**: Decomposes world models into seven components for zero-shot LLM generation supporting MCTS simulation
- **Method**: Enables planning in partially observable multi-agent environments without domain-specific training
- **Relevance**: Shows how LLMs can serve as world models in POMDPs, enabling MCTS without an explicit simulator — critical for our approach where we need the LLM to predict GUI state transitions

### 4.8 Generative World Explorer / Genex (Lu et al., 2024)

**"Generative World Explorer"**

- **Key idea**: Agents "mentally explore" unseen world areas via generative imagination, updating beliefs before making decisions
- **Method**: Creates synthetic observations to refine incomplete world understanding
- **Relevance**: Directly addresses partial observability by letting agents mentally simulate what they cannot currently see — could be used in MCTS to simulate unobserved branches without requiring actual environment interaction

### 4.9 AR-Bench (Zhou et al., 2025, ICML)

**"From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?"**

- **Key idea**: Benchmarks LLMs' ability to actively gather information under partial observability
- **Key finding**: LLMs struggle significantly with active reasoning compared to passive reasoning; tree-based search yields only modest improvements
- **Relevance**: Validates that partial observability is a real challenge for LLM agents, motivating the POMDP formulation. Also suggests MCTS alone may not suffice — we need good heuristics (the LLM prior policy)

### 4.10 DESPOT (Ye et al., 2017, JAIR)

**"DESPOT: Online POMDP Planning with Regularization"**

- **Key idea**: Sparse approximation of belief trees using randomly sampled scenarios, with regularization to balance value and policy size
- **Method**: Provides theoretical regret bounds tied to optimal policy representation size
- **Relevance**: Complementary to POMCP — another foundational online POMDP solver that could be adapted for GUI agent MCTS

---

## 5. Cold-Start Methods: Current Landscape

### 5.1 Method Comparison

| Method | Examples | Pros | Cons |
|--------|----------|------|------|
| **Human annotation** | Mind2Web, AITW | High quality, correct | Expensive, limited scale, low diversity |
| **LLM labeling** | GPT-4V annotation, AgentTrek | Scalable, cheap | Limited by LLM's own distribution, hallucinations |
| **Behavior cloning from demos** | DigiRL cold-start | Natural trajectories | Distribution mismatch, limited diversity |
| **Self-play / self-training** | STaR, ReST | Iterative improvement | Needs initial competence, mode collapse risk |
| **MCTS in RL phase** | AgentQ, SEEA-R1 | Explores well | Very expensive (MCTS per RL step), slow training |
| **MCTS at cold-start (ours)** | This proposal | Diverse data, one-time cost | Requires environment access, MCTS overhead |

### 5.2 STaR (Zelikman et al., 2022)

**"STaR: Self-Taught Reasoner — Bootstrapping Reasoning With Reasoning"**

- **Method**: LLM generates rationales → filter by correctness → SFT on correct rationales → repeat
- **Key insight**: Bootstrapping from the model's own correct solutions creates a virtuous cycle
- **Limitation**: Limited to the model's existing distribution — no active exploration of new strategies
- **Relevance**: Our MCTS approach actively explores beyond the model's current distribution, addressing STaR's diversity limitation

### 5.3 ReST (Gulcehre et al., 2023)

**"Reinforced Self-Training"**

- **Method**: Generate → Filter by reward → SFT → Repeat (similar to STaR but with explicit reward model)
- **Key insight**: Offline RL-style self-training can match online RL performance
- **Relevance**: ReST uses sampling-based generation; MCTS would provide more structured, diverse exploration

### 5.4 FireAct (Chen et al., 2023)

**"FireAct: Toward Language Agent Fine-tuning"**

- **Key idea**: Fine-tune LLMs for agent tasks using GPT-4 generated trajectories
- **Method**: GPT-4 generates action trajectories in ReAct format → SFT smaller models
- **Cold-start**: Pure LLM labeling approach — GPT-4 as teacher
- **Limitation**: Bounded by GPT-4's own capabilities and biases
- **Relevance**: Exemplifies the "LLM labeling" cold-start approach. MCTS can explore strategies that GPT-4 would never generate

### 5.5 Kimi k1.5 (2025)

**"Kimi k1.5: Scaling Reinforcement Learning with LLMs"**

- **Key idea**: Scaling laws for RL training of LLMs
- **Method**: Long-CoT supervised fine-tuning → RL with process rewards
- **Cold-start**: Carefully curated SFT data with long chain-of-thought traces
- **Key insight**: Quality and diversity of cold-start SFT data directly impacts RL training efficiency
- **Relevance**: **Empirically validates that cold-start data quality matters for downstream RL**

### 5.6 ERPO (Liu et al., 2025)

**"Explore Residual Prompts in Policy Optimization"**

- **Key idea**: Addresses "residual prompts" in RL — prompts that yield zero-variance rewards (all correct or all wrong)
- **Method**: Dynamically increases sampling temperature for residual prompts to generate more diverse reasoning traces
- **Key insight**: Diversity of rollouts is critical for RL training signal
- **Relevance**: **Directly supports our thesis** — diverse cold-start data prevents residual prompt problems during RL. MCTS-generated diverse trajectories at cold-start would reduce the number of residual prompts from the start

### 5.7 ASTER (Zhang et al., 2026)

**"ASTER: Agentic Scaling with Tool-integrated Extended Reasoning"**
- **arXiv**: 2602.01204
- **Key finding**: Identifies "interaction collapse" where models abandon tool use during RL. **A small expert cold-start set of just 4K interaction-dense trajectories yields the strongest downstream RL performance**
- **Results**: ASTER-4B reaches 90.0% on AIME 2025
- **Relevance**: **Core empirical evidence** that cold-start SFT data diversity (specifically interaction density/diversity) is critical for preventing collapse during RL training. Supports using MCTS to generate interaction-dense trajectories

### 5.8 Theoretical Perspectives on Data Quality (Javanmard et al., 2026)

**"Theoretical Perspectives on Data Quality and Synergistic Effects in Pre- and Post-Training"**
- **arXiv**: 2603.01293
- **Key theorem**: **SFT learns best from small sets of challenging examples; excessively large SFT datasets may dilute informative pretraining signals. RL performs best on large-scale data of moderate difficulty**
- **Relevance**: **Provides the theoretical justification** for why small, diverse, challenging MCTS-generated cold-start data (not massive LLM-labeled data) is optimal for SFT before RL

### 5.9 ACuRL (Xue et al., 2026)

**"ACuRL: Autonomous Continual Learning of Computer-Use Agents"**
- **arXiv**: 2602.10356
- **Key idea**: Autonomous Curriculum RL with zero human data. Curriculum task generator synthesizes training tasks based on agent capability
- **Results**: 4-22% gains without catastrophic forgetting
- **Relevance**: Alternative cold-start that bypasses trajectory collection entirely. Complementary to our approach — MCTS cold-start could provide better initialization for autonomous curriculum RL

### 5.10 Self-Play SFT / SPA (Chen et al., 2025)

**"Internalizing World Models via Self-Play Finetuning"**
- **arXiv**: 2510.15047
- **Key idea**: Cold-starts policy via Self-Play SFT to learn a world model from environment interaction
- **Results**: Sokoban: 25.6% → 59.8%; FrozenLake: 22.1% → 70.9%
- **Relevance**: Self-play as cold-start — agent generates training data through self-play interaction. MCTS provides more structured exploration than pure self-play

### 5.11 Endless Terminals (Gandhi et al., 2026)

**"Endless Terminals: Scaling RL Environments for Terminal Agents"**
- **arXiv**: 2601.16443
- **Key finding**: "Simple RL succeeds when environments scale." Procedurally generates 3,255 diverse terminal tasks. Llama-3.2-3B: 4.0% → 18.2%, Qwen2.5-7B: 10.7% → 53.3%
- **Relevance**: **Environment diversity may be even more important than trajectory diversity**. For MCTS cold-start, generating diverse task/environment variants is as important as diverse trajectories per task

### 5.12 ASTRA (Tian et al., 2026)

**"ASTRA: Automated Synthesis of Trajectories and Reinforcement Arenas"**
- **arXiv**: 2601.21558
- **Key idea**: Combines trajectory synthesis (using tool-call graph topology for diversity) with environment synthesis (converting traces into verifiable executables)
- **Relevance**: Fully automated cold-start pipeline. MCTS could be integrated into ASTRA-style pipelines for even more diverse trajectory generation

### 5.13 DeepSeek-R1 (2025) [arXiv: 2501.12948]

**"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"**

- **4-stage pipeline**:
  1. **Cold-Start SFT**: Thousands of long CoT examples (few-shot, direct prompting, R1-Zero outputs, human annotation) → fine-tune base model as initial RL actor
  2. **Reasoning-Oriented RL**: GRPO with accuracy + format + language consistency rewards
  3. **Rejection Sampling + Mixed SFT**: ~600k reasoning + ~200k non-reasoning samples
  4. **Universal RL**: All scenarios with rule-based + preference-based rewards
- **Key insight on cold-start**: Cold-start data "prevents the early unstable cold start phase of RL training from the base model." Provides initial behavioral priors, improves readability
- **Relevance**: **Gold-standard example of cold-start → RL pipeline**. Shows cold-start is essential (R1-Zero without it had readability and language mixing issues)

### 5.14 "SFT Memorizes, RL Generalizes" (Chu et al., 2025) [arXiv: 2501.17161]

**Critical findings for cold-start design:**
- RL generalizes across OOD variants; **SFT memorizes and fails to generalize** (-8% to -79% OOD degradation)
- **SFT is essential for bootstrapping**: RL directly on base model fails completely — base model has "poor instruction following capability"
- **But excessive SFT hurts RL**: Overly-tuned SFT creates local optima that RL cannot escape
- SFT functions as a **"format teacher"** — stabilizes output format so RL can optimize task performance
- **Relevance**: **Directly validates our approach** — cold-start SFT should be diverse (not overfit), and MCTS-generated data provides exactly this: diverse strategies that don't overfit to a single pattern

### 5.15 "Teaching LLMs to Reason with RL" (Havrilla et al., 2024) [arXiv: 2403.04642]

**Critical findings on exploration and cold-start:**
- **RL algorithms do NOT explore significantly beyond SFT-discovered solutions**. Pass@96 saturates early
- Expert Iteration outperforms PPO in most cases (~10^6 samples to converge)
- **SFT trained for 4 epochs produced only 2.9 unique correct solutions vs. 3.7 for 2-epoch SFT** — overfitting limits solution diversity and blocks RL improvements
- There is a **"tradeoff between maj@1 and pass@96 during SFT training"** — continued SFT improves greedy accuracy at the expense of sampling diversity
- **Relevance**: **Strongest argument for MCTS cold-start**. Since RL does NOT discover new strategies beyond SFT, the diversity of cold-start data directly determines the diversity of strategies available during RL. MCTS maximizes this diversity

### 5.16 "Cognitive Behaviors that Enable Self-Improving Reasoners" (Gandhi et al., 2025) [arXiv: 2503.01307]

**Key findings on SFT initialization for RL:**
- Four critical cognitive behaviors: backtracking, verification, subgoal setting, backward chaining
- **Structure over correctness**: Models primed with *incorrect* solutions containing proper reasoning structures achieved equivalent RL performance to correct-solution priming
- Standard pretraining corpora contain these behaviors "infrequently"
- **Relevance**: Cold-start data should emphasize **reasoning structure diversity** (backtracking, verification), not just correctness. MCTS naturally produces these: it backtracks, explores subgoals, and verifies outcomes — the tree structure itself encodes these cognitive behaviors

### 5.17 "Scaling of Search and Learning" Roadmap (2024) [arXiv: 2412.14135]

**Key insights for our approach:**
- Policy initialization must balance **"sampling-efficiency and sampling-diversity"** — over-convergence to SFT strategies limits discovery during search
- Search generates **higher-quality training data** than simple sampling
- Process rewards (step-level) are more effective than outcome rewards
- **Circular enhancement cycle**: better search → higher-quality trajectories → better policy → better search
- **Relevance**: Directly frames our approach as part of the "search → training data → better policy" loop, applied at cold-start

---

## 6. Data Diversity and Its Impact on RL

### 6.1 Why Diversity Matters for RL Cold-Start

The core argument for MCTS cold-start rests on the following chain:

```
Diverse cold-start SFT data
  → Model learns multiple strategies per task type
    → RL rollouts sample from a broader distribution
      → More diverse (positive, negative) pairs for training signal
        → Better RL training convergence and final performance
```

**Evidence from literature:**

1. **ERPO (2025)**: Shows that when all rollouts for a prompt are identical (zero variance), no RL learning signal is produced. Diverse initialization prevents this.

2. **DLR (Yang et al., 2025)**: "Discover, Learn, and Reinforce" — demonstrates that diverse RL-generated trajectories lead to better VLA model training. Models trained on diversified data "surpass counterparts trained on equal-sized standard RL datasets" and show "positive data-scaling behavior that single-pattern RL lacks."

3. **TopoCurate (Yang et al., 2026)**: Shows that selecting SFT trajectories with "strategic diversity" improves both SFT (+4.2%) and RL (+6.9%) performance. Confirms trajectory diversity is valuable for training.

4. **Kimi k1.5 (2025)**: Empirically shows cold-start data quality directly impacts RL efficiency.

5. **ASTER (Zhang et al., 2026)**: **Just 4K interaction-dense trajectories outperform larger homogeneous datasets** for cold-start. Interaction density (diversity of tool use patterns) is more important than volume.

6. **Theoretical Perspectives (Javanmard et al., 2026)**: **Theoretically proves** that SFT learns best from small sets of challenging examples, while RL needs large-scale moderate-difficulty data. This means cold-start SFT should prioritize diversity over quantity.

7. **From Words to Actions (He et al., 2024)**: **Theoretically proves** that naive imitation in POMDPs leads to linear regret — exploration-based methods (like MCTS) are necessary for sublinear regret.

8. **Endless Terminals (Gandhi et al., 2026)**: Environment diversity is the key bottleneck — given diverse environments, even simple RL succeeds.

9. **"Teaching LLMs to Reason with RL" (Havrilla et al., 2024)**: **RL does NOT explore beyond SFT-discovered solutions** — pass@96 saturates early. SFT overfitting (4 epochs) produces only 2.9 unique solutions vs 3.7 for 2 epochs. This is the strongest argument: since RL cannot find new strategies, cold-start diversity is the ceiling.

10. **"SFT Memorizes, RL Generalizes" (Chu et al., 2025)**: Excessive SFT creates local optima RL cannot escape. SFT should be diverse (not overfit) — exactly what MCTS produces.

11. **"Cognitive Behaviors" (Gandhi et al., 2025)**: Structure > correctness for cold-start. Models primed with *incorrect* solutions containing proper reasoning structures perform equivalently in RL. MCTS naturally produces backtracking, verification, and subgoal exploration.

12. **"Scaling Search and Learning" (2024)**: Policy initialization must balance "sampling-efficiency and sampling-diversity." Search produces higher-quality training data than sampling.

### 6.2 MCTS as a Diversity Engine

MCTS naturally generates diverse trajectories because:

1. **Exploration via UCB**: UCB1 formula balances exploitation of known-good actions with exploration of under-visited actions
2. **Multiple rollouts**: Each MCTS iteration explores a different branch of the action tree
3. **Backpropagation**: Failed branches are deprioritized but still recorded (useful as negative examples)
4. **Stochasticity**: Random rollouts in MCTS explore the space beyond the policy's mode

In the GUI agent context:
- For the same task (e.g., "open settings"), MCTS would discover multiple valid paths (menu bar → settings, keyboard shortcut, system tray → settings, search bar → settings)
- Failed attempts (wrong clicks, dead ends) provide valuable negative examples for DPO/RLHF
- The tree structure naturally records the decision points where strategies diverge

---

## 7. Proposed Method: POMDP-MCTS Cold-Start

### 7.1 Algorithm Overview

```
Input: GUI environment E, task set T, base LLM M
Output: Diverse SFT dataset D

1. For each task t in T:
   a. Initialize belief b₀ from initial screenshot observation o₀
   b. Run MCTS with M as prior policy:
      - Selection: UCB1 over action-observation histories
      - Expansion: M proposes candidate actions given (history, observation)
      - Simulation: Execute action in E, observe new screenshot
      - Backpropagation: Update values based on task progress signal
   c. Collect all explored trajectories (successful and failed)
   d. Filter and annotate trajectories:
      - Successful trajectories → positive SFT examples
      - Failed trajectories → negative examples (for DPO)
      - Near-miss trajectories → hard negative examples

2. Aggregate and deduplicate dataset D
3. SFT train model M on D → M_cold
4. Proceed to RL phase with M_cold as initialization
```

### 7.2 Key Design Decisions

**MCTS Configuration:**
- **Action space**: Discretized GUI actions (click at coordinates, type text, scroll, shortcuts)
- **Observation**: Screenshot (+ optional accessibility tree for efficiency)
- **Prior policy**: Base LLM M provides action probabilities (reduces branching factor from thousands to ~10-50 plausible actions)
- **Value function**: Can be trained online during MCTS, or use a simpler heuristic (visual similarity to goal state, task completion estimate)
- **Search depth**: Limited to reasonable episode lengths (e.g., 10-30 steps)
- **Branching factor**: Top-K actions from LLM prior (K=5-10)

**POMDP-specific adaptations:**
- **Belief tracking**: Maintain history of (action, observation) pairs as implicit belief state
- **Observation-based branching**: Tree nodes are action-observation histories, not states
- **Partial observability handling**: LLM processes the full history of screenshots to maintain context

### 7.3 Advantages Over Alternatives

| vs. LLM Labeling | vs. MCTS in RL |
|-------------------|----------------|
| Explores beyond LLM's distribution | One-time cost, not per-RL-iteration |
| Discovers non-obvious strategies | Can be parallelized offline |
| Generates both positive and negative examples | Doesn't slow down RL training loop |
| Less susceptible to LLM hallucination | Same quality of exploration |

---

## 8. Key Technical Challenges

### 8.1 Environment Reset and Branching

**Challenge**: GUI environments are stateful and hard to reset. MCTS requires exploring multiple branches from the same state.

**Possible solutions**:
- **VM snapshots**: Take VM snapshots at decision points, restore for different branches (OSWorld already uses VMs)
- **Parallel VMs**: Run multiple VM instances for parallel MCTS exploration
- **Approximate branching**: Instead of exact state restoration, re-execute the prefix of actions to reach the branching point
- **Limited backtracking**: Only backtrack 1-2 steps rather than full tree traversal

### 8.2 Observation Space Complexity

**Challenge**: Screenshots are high-dimensional observations, making belief tracking expensive.

**Possible solutions**:
- **VLM embeddings**: Use VLM to encode screenshots into compact representations
- **Accessibility tree**: Use structured accessibility tree as auxiliary observation (lower dimensional)
- **History compression**: Summarize observation history using the LLM itself

### 8.3 Reward Signal Design

**Challenge**: Task completion reward is sparse and binary (success/failure at episode end).

**Possible solutions**:
- **Progress-based rewards**: Detect intermediate progress (correct window opened, right text field selected)
- **LLM-as-judge**: Use LLM to evaluate whether an action made progress toward the goal
- **Visual similarity**: Compare current screenshot to expected intermediate/final states
- **Process reward model**: Train a PRM on initial data, then use it to guide MCTS (following Math-Shepherd approach)

### 8.4 Computational Cost

**Challenge**: MCTS in GUI environments is expensive (each action requires rendering a screenshot).

**Mitigation strategies**:
- **Cold-start only**: The entire MCTS phase runs once, not during RL
- **Parallel execution**: Run MCTS across many VMs simultaneously
- **Efficient search**: Use strong LLM prior to reduce effective branching factor
- **Time budget**: Set a fixed time budget per task, stop MCTS when budget is exhausted
- **Shared exploration**: Reuse exploration data across similar tasks

---

## 9. Related Work Positioning

### 9.1 Differentiation from AgentQ

AgentQ is the closest related work. Key differences:

| Aspect | AgentQ | Our Proposal |
|--------|--------|-------------|
| **When MCTS is used** | During online DPO training (every iteration) | During cold-start only (one-time) |
| **Cost** | Proportional to RL training iterations | Fixed, one-time cost |
| **Purpose of MCTS** | Generate preference pairs for DPO | Generate diverse SFT data for initialization |
| **Environment** | WebShop (simulated) | Real OS environments (OSWorld) |
| **POMDP formulation** | Implicit | Explicit POMDP modeling |
| **Dataset contribution** | No dataset released | Dataset as a contribution |

### 9.2 Differentiation from SEEA-R1

| Aspect | SEEA-R1 | Our Proposal |
|--------|---------|-------------|
| **Domain** | Embodied agents (ALFWorld) | GUI agents (OSWorld) |
| **When MCTS is used** | During RL (Tree-GRPO) | During cold-start |
| **Tree structure** | Action trees for policy optimization | Action-observation trees for data generation |
| **Reward** | Multi-modal generative reward model | Task completion + progress signals |

### 9.3 Differentiation from ReST-MCTS*

| Aspect | ReST-MCTS* | Our Proposal |
|--------|-----------|-------------|
| **Domain** | Math reasoning | GUI agent interaction |
| **State space** | Token sequences (fully observable) | GUI states (partially observable) |
| **Observability** | Full | Partial (POMDP) |
| **MCTS integration** | Iterative self-training loop | One-time cold-start |
| **Action space** | Next token | GUI actions (click, type, scroll) |

### 9.4 Differentiation from WebPilot

| Aspect | WebPilot | Our Proposal |
|--------|---------|-------------|
| **When MCTS is used** | Inference time (per query) | Training time (cold-start) |
| **Purpose** | Improve test-time performance | Generate training data |
| **Cost amortization** | Per-query cost | One-time cost, amortized over all training |
| **Model improvement** | No model update | Trains a better model |

### 9.5 Differentiation from ProAct/ExACT

| Aspect | ProAct / ExACT | Our Proposal |
|--------|---------------|-------------|
| **Search → SFT pipeline** | Iterative (search → distill → repeat) | One-time cold-start before RL |
| **Purpose** | Replace inference-time search with learned behavior | Initialize model for RL training |
| **Downstream** | No RL phase — SFT only | SFT → RL pipeline |
| **Formulation** | Implicit MDP | Explicit POMDP |
| **Focus** | Test-time compute reduction | Training data diversity for RL |

### 9.6 Differentiation from TSR

| Aspect | TSR | Our Proposal |
|--------|-----|-------------|
| **When search runs** | During RL rollouts | Before RL (cold-start) |
| **Integration** | Tree search augments each rollout | MCTS generates offline dataset |
| **RL compatibility** | PPO/GRPO with search-augmented rollouts | Any RL algorithm (clean separation) |
| **Overhead** | Per-rollout search cost during RL | Zero overhead during RL |

### 9.7 Overall Novelty Statement

**To our knowledge, no prior work has:**
1. Applied MCTS specifically during the SFT cold-start phase for GUI agents
2. Explicitly formulated GUI agent data generation as a POMDP planning problem
3. Used MCTS-generated trajectories as a diversity-enhancing mechanism for downstream RL initialization
4. Combined POMDP belief tracking with LLM-guided MCTS for GUI environment exploration

The closest works either use MCTS during RL (AgentQ, SEEA-R1 — expensive) or use MCTS for different domains (ReST-MCTS* — math reasoning, fully observable). Our contribution is showing that a one-time MCTS cold-start investment yields better RL training outcomes through improved data diversity.

---

## Appendix: Full Paper List (70+ Papers)

### A. MCTS + LLM Papers
1. rStar-Math (Guan et al., 2025) - MCTS for math self-training [arXiv: 2501.04519]
2. ReST-MCTS* (Zhang et al., 2024) - Process reward guided tree search [arXiv: 2406.03816]
3. V-STaR (Hosseini et al., 2024) - Verifiers for self-taught reasoners [arXiv: 2402.06457]
4. PPO-MCTS (Liu et al., 2023) - Value-guided MCTS decoding [arXiv: 2309.15028]
5. AlphaLLM (2024) - MCTS self-play for LLMs [arXiv: 2404.12253]
6. LLM-MCTS (Zhao et al., 2023) - LLM as world model for MCTS
7. Marco-o1 (2024) - Open reasoning with MCTS [arXiv: 2411.14405]
8. Math-Shepherd (Wang et al., 2023) - Process reward model [arXiv: 2312.08935]
9. MCTS-DPO (Xie et al., 2024) - Step-level preference learning [arXiv: 2405.00451]
10. SEEA-R1 (Tian et al., 2025) - Tree-structured RL for embodied agents
11. Self-Explore (Hwang et al., 2024) - Fine-grained reward for math reasoning
12. RAP (Hao et al., 2023) - LLM as world model in MCTS (EMNLP) [arXiv: 2305.14992]
13. AlphaMath Almost Zero (Chen et al., 2024) - Zero-annotation MCTS [arXiv: 2405.03553]
14. OmegaPRM (Luo et al., 2024) - 1.5M+ auto process supervision [arXiv: 2406.06592]
15. Distilling System 2 into System 1 (Yu et al., 2024) - Search → SFT paradigm [arXiv: 2407.06023]
16. Compute-Optimal Sampling (Bansal et al., 2024) - Weaker models = more diverse data [arXiv: 2408.16737]
17. AlphaLLM-CPL (Wang et al., 2024) - Curriculum preference learning from MCTS [arXiv: 2410.06508]
18. rStar (Qi et al., 2024) - Mutual reasoning + MCTS [arXiv: 2408.06195]
19. MCTSr (Zhang et al., 2024) - Monte Carlo Tree Self-Refine [arXiv: 2406.07394]
20. Tree of Thoughts (Yao et al., 2023) - Multi-path reasoning [arXiv: 2305.10601]

### B. GUI/Web Agent Training Papers
21. AgentQ (Putta et al., 2024) - MCTS + DPO for web agents [arXiv: 2408.07199]
22. DigiRL (2024) - Autonomous RL for device control [arXiv: 2406.11896]
23. WebRL (2024) - Self-evolving online curriculum RL [arXiv: 2411.02337]
24. UI-TARS (2025) / UI-TARS-2 (2025) - GUI agent with multi-turn RL [arXiv: 2501.12326, 2509.02544]
25. WebPilot (Zhang et al., 2024) - MCTS for web task execution [arXiv: 2408.15978]
26. OS-ATLAS (2024) - Foundation action model for GUI agents [arXiv: 2410.23218]
27. GUI-Shift (2025) - Self-supervised RL for GUI agents [arXiv: 2505.12493]
28. InSTA (2025) - Internet-scale training for agents [arXiv: 2502.06776]
29. Mind2Web (2023) - Generalist web agent [arXiv: 2306.06070]
30. WebArena (2023) - Realistic web environment benchmark [arXiv: 2307.13854]
31. Android in the Wild (AITW) (2023) - Large-scale Android dataset [arXiv: 2307.10088]
32. EvoCUA (2026) - Evolutionary computer use agents, 56.7% OSWorld [arXiv: 2601.15876]
33. OS-Genesis (2024) - Reverse task synthesis (ACL 2025) [arXiv: 2412.19723]
34. AgentTrek (2024) - Trajectory synthesis from tutorials (ICLR 2025) [arXiv: 2412.09605]
35. CogAgent (2023) - 18B VLM for GUI (CVPR 2024) [arXiv: 2312.08914]
36. SeeClick (2024) - GUI grounding pre-training [arXiv: 2401.10935]
37. ZeroGUI (2025) - Zero human cost GUI training [arXiv: 2505.23762]

### C. Tree Search in Interactive Environments
22. Tree Search for LM Agents (Koh et al., 2024) - Best-first search in real web envs [arXiv: 2407.01476]
23. LATS (Zhou et al., 2023) - Language Agent Tree Search [arXiv: 2310.04406]
24. ExACT / R-MCTS (Yu et al., 2024) - Reflective MCTS + distillation [arXiv: 2410.02052]
25. Plan-MCTS (Zhang et al., 2026) - Plan-space tree search for web navigation [arXiv: 2602.14083]
26. ProAct (Yu et al., 2026) - GLAD: tree search → SFT distillation [arXiv: 2602.05327]
27. TSR (Djuhera et al., 2026) - Tree search in RL rollouts [arXiv: 2602.11767]
28. Dyna-Mind (Yu et al., 2025) - Learning to simulate from search trees [arXiv: 2509.25189]

### D. POMDP and Planning Papers
29. POMCP (Silver & Veness, 2010) - Monte Carlo planning in POMDPs (NeurIPS)
30. DESPOT (Ye et al., 2017) - Online POMDP planning with regularization (JAIR) [arXiv: 1609.03250]
31. BA-POMCP (Katt et al., 2018) - Learning in POMDPs with MCTS [arXiv: 1806.05631]
32. NeoPlanner (Paul, 2023) - LLM-guided planning in large POMDPs [arXiv: 2312.07368]
33. LOOP (Chen et al., 2025) - RL for long-horizon interactive LLM agents as POMDPs [arXiv: 2502.01600]
34. From Words to Actions (He et al., 2024) - Theoretical POMDP framework for LLM agents (ICML)
35. PIANIST (Light et al., 2024) - LLM world models for MCTS in POMDPs (NeurIPS Workshop)
36. Genex (Lu et al., 2024) - Generative exploration for partial observability
37. AR-Bench (Zhou et al., 2025) - Active reasoning under incomplete information (ICML)

### E. Cold-Start and Data Generation Papers
38. STaR (Zelikman et al., 2022) - Self-taught reasoner
39. ReST (Gulcehre et al., 2023) - Reinforced self-training
40. FireAct (Chen et al., 2023) - Language agent fine-tuning
41. Kimi k1.5 (2025) - Scaling RL with LLMs
42. ERPO (Liu et al., 2025) - Exploring residual prompts in policy optimization [arXiv: 2511.04800]
43. DLR (Yang et al., 2025) - Diverse RL-generated trajectories [arXiv: 2511.19528]
44. TopoCurate (Yang et al., 2026) - Interaction topology for agent training [arXiv: 2603.01714]
45. LIMO (Ye et al., 2025) - Less is more for reasoning (data efficiency)
46. O1 Replication Journey Part 2 (Huang et al., 2024) - Distillation approaches
47. ASTER (Zhang et al., 2026) - 4K diverse trajectories beat larger datasets [arXiv: 2602.01204]
48. ACuRL (Xue et al., 2026) - Autonomous curriculum RL [arXiv: 2602.10356]
49. SPA (Chen et al., 2025) - Self-play SFT cold-start [arXiv: 2510.15047]
50. Endless Terminals (Gandhi et al., 2026) - Environment diversity as bottleneck [arXiv: 2601.16443]
51. ASTRA (Tian et al., 2026) - Automated trajectory + environment synthesis [arXiv: 2601.21558]
52. Theoretical Perspectives (Javanmard et al., 2026) - Data quality theory for SFT/RL [arXiv: 2603.01293]

### F. Cold-Start Dynamics and RL Training Papers
53. DeepSeek-R1 (2025) - 4-stage cold-start → RL pipeline [arXiv: 2501.12948]
54. "SFT Memorizes, RL Generalizes" (Chu et al., 2025) - SFT essential but excessive SFT hurts RL [arXiv: 2501.17161]
55. "Teaching LLMs to Reason with RL" (Havrilla et al., 2024) - RL doesn't explore beyond SFT [arXiv: 2403.04642]
56. "Cognitive Behaviors for Self-Improving Reasoners" (Gandhi et al., 2025) - Structure > correctness [arXiv: 2503.01307]
57. "Scaling of Search and Learning" Roadmap (2024) - Search → data → policy cycle [arXiv: 2412.14135]
58. "Does RL Incentivize Reasoning Beyond Base Model?" (Yue et al., 2025) - RL constrained by base [arXiv: 2504.13837]
59. "Demystifying Long CoT Reasoning" (Yeo et al., 2025) - SFT+RL > RL alone [arXiv: 2502.03373]
60. Scaling LLM Test-Time Compute (Snell et al., 2024) - Compute-optimal search [arXiv: 2408.03314]
61. Large Language Monkeys (2024) - Coverage scales log-linearly with samples [arXiv: 2407.21787]
62. ETO (2024) - Exploration-based trajectory optimization with contrastive DPO [arXiv: 2403.02502]

---

*Document compiled: 2026-03-14*
*Updated with findings from 4 parallel research agents*
*For the ARPO GUI agent training research project*
