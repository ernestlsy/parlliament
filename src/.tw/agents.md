Autonomous ML Research Loop Architecture
For this hackathon, defining the agents around the actual autonomous ML research loop—rather than making every ML task its own independent agent—is highly effective. The key is to make the Evolution Judge the decision-maker for experiments, while the Orchestrator manages execution and the Persistent Judge provides historical memory.
1. Orchestrator Agent
Primary responsibility: Control the entire autonomous ML research process. It should not decide what ML technique is best. Instead, it coordinates the other agents.
Responsibilities:
Initialize the benchmark and create the initial candidate
Spawn appropriate agents and pass information between them
Maintain the global experiment state and decide which agent needs to act next
Launch experiments requested by the Evolution Judge
Handle agent failures and enforce the 50-iteration / 6-hour constraints
Ensure the pipeline eventually produces a valid submission
Track manual interventions and ensure every experiment is properly logged

2. Research Agent
Primary responsibility: Find external knowledge that can inspire better experiments. This directly addresses the hackathon requirement that the agent should draw from papers, industry practices, existing approaches, and public solutions.
Responsibilities:
Search: Academic papers, recommender-system architectures, KuaiRand papers, public implementations, and GitHub repositories.
Explore Industry Practices: Feature-engineering techniques, multi-task learning, debiasing, counterfactual learning, ranking models, training strategies, loss functions, regularization methods, and hyperparameter strategies.
Convert: Translate research into actionable recommendations.
Example Output:
Research Finding: Multi-task learning can exploit auxiliary feedback signals.
Evidence: Paper X, Paper Y, Industry approach Z
Hypothesis: Jointly predicting click + long_view may improve representation quality.
Suggested Experiment: Replace single-task FM with shared-bottom multi-task model.
Expected Benefit: Improved ranking representation.
Risk: Task conflict may reduce long_view performance
Implementation Complexity: Medium

3. Data Scientist Agent
Primary responsibility: Understand the problem and discover opportunities in the data. This agent is responsible for the first part of the MLE loop: Read problem → Inspect data → Understand what matters.
Responsibilities:
Problem understanding: Dataset structure, prediction target (e.g., long_view), evaluation metrics (GAUC, nDCG@5), train/validation split, ranking setup, constraints, submission format, and leakage risks.
Data inspection: Missing values, cardinality, feature/user/item distributions, label imbalance, temporal patterns, user activity, item popularity, feature correlations, duplicate interactions, sparse features, and potential leakage.
Data-driven hypotheses: Discovering actionable insights, such as high correlation between long_view and play_time.
Example Output Structure:
Data Finding
    ↓
Why it matters
    ↓
Hypothesis
    ↓
Potential experiment

4. Feature Engineer Agent
Primary responsibility: Turn data/research hypotheses into feature representations. This is one of the agents that actually modifies the ML pipeline.
Responsibilities:
Design Features: User, item, context, temporal, interaction, cross, historical, popularity, sequence, and aggregated features.
Determine Encoding: Encoding strategy, embedding dimensions, normalization, feature selection, feature crossing, and leakage prevention.
Important: The Feature Engineer should be hypothesis-driven, not randomly generate hundreds of features.

5. Model Agent
Primary responsibility: Design the learning algorithm and training configuration. (Combines Architecture and Training aspects).
Responsibilities:
Choose Architecture: FM, DeepFM, Wide & Deep, DNN, DCN, xDeepFM, DIN, Transformer-based models, Multi-task networks, Shared-bottom architectures, MMOE, PLE.
Choose Loss: Binary cross entropy, Pairwise ranking loss, BPR, Focal loss, Weighted BCE, Multi-task losses, Auxiliary-task losses.
Training Configuration: Learning rate, batch size, embedding dimensions, number of layers, hidden dimensions, optimizer, weight decay, dropout, early stopping, number of epochs.
Example Output:
Hypothesis: DeepFM should capture nonlinear user-item interactions that FM cannot represent.
Architecture: Embedding → FM component + Embedding → DNN → prediction
Loss: BCE
LR: 0.001
Expected improvement: +0.005 primary
Risk: Higher compute cost

6. Experiment / Compute Manager
Primary responsibility: Execute experiments reliably and measure resources. This is the "hands-on execution layer."
Responsibilities:
Generate/run training code, allocate compute, execute training.
Monitor runtime, memory, CPU/GPU, capture stdout/stderr, detect crashes, retry failed experiments.
Validate outputs, run evaluation, record metrics, record iteration count, token usage, and wall-clock time.
Save checkpoints and experiment artifacts.
Enforce iterations < 50 and wall_clock < 6 hours.

7. Evolution Judge Agent
Primary responsibility: Decide what experiment should happen next based on the population of previous experiments. It is the research strategist / evolutionary controller.
Responsibilities:
Determine which candidate survives, is eliminated, reproduces, gets mutated, or should be explored further.
Decide when to exploit a promising direction vs. explore something new.
Determine which hypothesis should be tested next and how lineage evolves.
Inputs: Current candidates + Experiment results + Research recommendations + Data findings + Historical experiment summaries.

8. Persistent Judge Agent
Primary responsibility: Remember the history of the research process. The Evolution Judge is stateless regarding long-term history; the Persistent Judge maintains the research memory.
Stores for every experiment:
Experiment ID, Parent ID, Hypothesis, Research source, Data finding
Feature changes, Model changes, Code diff
Metrics, Runtime, Failure, Recovery, Decision, Lineage
It prevents the system from repeatedly rediscovering the same dead end.
Baseline
   │
   ├── FM + popularity
   │      │
   │      └── FM + popularity + time
   │
   └── DeepFM
          │
          ├── DeepFM + MTL
          │       │
          │       └── DeepFM + MTL + popularity
          │
          └── DeepFM + ranking loss

The Crucial Distinction Between Decision Agents
Agent
Question it answers
 
Persistent Judge
"What have we learned?"
Evolution Judge
"What should we try next?"
Orchestrator
"How do we execute that?"


Complete Autonomous Loop
               ┌───────────────┐
               │  Orchestrator │
               └───────┬───────┘
                       │
            INITIAL UNDERSTANDING
                       │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  Research Agent             Data Scientist Agent
         │                             │
         └──────────────┬──────────────┘
                        ▼
                Feature Engineer
                        │
                        ▼
                   Model Agent
                        │
                        ▼
             Experiment / Compute
                        │
                        ▼
                    Evaluate
                        │
                        ▼
              Persistent Judge
                        │
                        ▼
               Evolution Judge
                        │
              ┌─────────┴─────────┐
              │                   │
           Continue             Stop
              │
              ▼
          New hypothesis
              │
              ▼
          Orchestrator
              │
              └───────────────► LOOP

Unified Experiment Specification
Instead of having the Feature Engineer and Model Agent independently generate solutions every iteration, create a unified Experiment Specification that the Orchestrator turns into actual work.
experiment_id: 23
parent: 17
hypothesis:
  "Adding item popularity and user activity features
   will improve nDCG@5."
research:
  - "Paper X"
  - "Industry approach Y"
data_findings:
  - "Item popularity is highly skewed."
features:
  add:
    - item_impression_count
    - item_long_view_rate
    - user_impression_count
model:
  architecture: DeepFM
  embedding_dim: 32
training:
  optimizer: Adam
  learning_rate: 0.001
  batch_size: 1024
evaluation:
  primary: mean(GAUC, nDCG@5)

Hackathon Requirement Mapping
Challenge requirement
Your architecture
 
Read problem
Data Scientist
Inspect data
Data Scientist
Research existing methods
Research Agent
Engineer features
Feature Engineer
Choose architecture
Model Agent
Train
Compute Manager
Tune
Model Agent + Evolution Judge
Evaluate
Compute Manager
Reflect
Persistent Judge
Decide next experiment
Evolution Judge
Write/revise code
Model/Feature agents + Compute Manager
Recover from failure
Compute Manager + Orchestrator
Maintain history
Persistent Judge
Repeat autonomously
Orchestrator


Key Architectural Decisions
No separate Evaluation Agent: Evaluation is deterministic. There is no reason to spend an LLM call deciding what metrics to run. Instead, the Compute Manager executes the fixed evaluation script, and the Evolution Judge interprets the results.
No separate Mutation/Crossover/Training Agents: These are better treated as operations performed by the Evolution Judge and Model Agent.
This leaves a much cleaner architecture of 8 specialized agents while still covering essentially the entire MLE iteration loop the hackathon is asking you to automate.