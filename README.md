# ParLLiaMent

ParLLiaMent is an autonomous machine learning research agent for the KuaiRand-Pure recommender-system benchmark. It automates the iterative MLE loop of understanding the task, inspecting data, engineering features, training and tuning models, evaluating results, reflecting on evidence, and selecting the next experiment.

The system uses OpenAI GPT-5.6 agents for research, hypothesis generation, review, orchestration, and code generation. A deterministic host process controls the parts that must remain trustworthy: temporal splits, experiment IDs, file ownership, subprocess execution, evaluation, convergence, journaling, and final submission selection.

## Project overview

Each ParLLiaMent generation follows this workflow:

1. Profile the dataset and run leakage-aware feature screening on a temporal holdout contained entirely within the training period.
2. Summarize the screening results and weaknesses from earlier scored experiments.
3. Retrieve relevant cards from a hash-validated recommender-system research knowledge base.
4. Generate a portfolio of evidence-backed hypotheses and compare them in a candidate tournament.
5. Select one interpretable ablation and convert it into an explicit implementation contract.
6. Delegate only the required `data.py`, `model.py`, `train.py`, and `config.json` files to specialized coding roles.
7. Run a contract probe followed by one bounded training process.
8. Evaluate canonical validation predictions with fixed GAUC and nDCG@5 implementations.
9. Record the hypothesis, code diff, metrics, diagnostics, failures, and recovery actions before planning the next experiment.

Only successfully scored validation experiments receive experiment IDs. Planning, screening, repair attempts, and abandoned runs do not consume the experiment budget. The final submission always uses the experiment with the highest observed validation primary score, where `primary = (GAUC + nDCG@5) / 2`.

Additional implementation details are available in [the package README](src/parlliament/README.md) and [the project description](project_description.txt).

## Setup and installation

### Requirements

- Python 3.9 or later
- The KuaiRand-Pure dataset from the organizer's Starter Kit
- An OpenAI API key with access to `gpt-5.6-terra`
- Windows, Linux, or WSL

The core Python dependencies are NumPy, pandas, and scikit-learn. From the repository root, create an isolated environment and install ParLLiaMent in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\src\parlliament
```

On Linux or WSL, activate the environment with:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./src/parlliament
```

Place or retain the dataset at:

```text
src/kuairand-starter-kit/KuaiRand-Pure/data/
```

Set the API key without committing it to the repository:

```powershell
$env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

For Linux or WSL:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Verify the installation:

```powershell
parlliament --help
python -m unittest discover -s .\src\parlliament\tests -v
```

## Reproducing the workflow and results

The original submitted run was launched from `src/parlliament` with the following exact command:

```text
parlliament run --workspace . --data-dir ../kuairand-starter-kit/KuaiRand-Pure/data --model gpt-5.6-terra --run_name run_2 --seed-model simple
```

To reproduce it:

```powershell
cd .\src\parlliament
parlliament run --workspace . --data-dir ../kuairand-starter-kit/KuaiRand-Pure/data --model gpt-5.6-terra --run_name run_2 --seed-model simple
```

`--run_name` is retained as an alias of the canonical `--run-name` option so the original command remains executable. The run directory is created at `src/parlliament/runs/run_2`. ParLLiaMent can resume an interrupted run by invoking the same command again with the same workspace, run name, and seed model.

Inspect progress or the final journal with:

```powershell
parlliament status .\runs\run_2
```

When the run has stopped, generate or refresh its submission bundle with:

```powershell
parlliament submit .\runs\run_2
```

The bundle contains the iteration log, validation-best result summary, resource accounting, manifest, and canonical KuaiRand-Pure submission CSV. LLM generation and model training can be nondeterministic, so a new run may not reproduce every intermediate hypothesis or metric bit-for-bit. The command reproduces the agent configuration, data split, evaluation protocol, seed scaffold, and autonomous workflow.

## Limitations and future improvements

ParLLiaMent still makes official experiment decisions from one public validation period, so repeated adaptation can overfit that period even though train-only screening reduces unnecessary validation trials. Given more time, we would add stronger multi-seed confirmation, paired uncertainty estimates, and a frozen secondary robustness gate before promoting a final model.

The system is sequential by design and can consume substantial LLM tokens and wall-clock time. Retrieval caching, shorter role prompts, reusable structured summaries, and early rejection of low-information candidates could reduce cost without weakening the audit trail.

Generated experiment Python runs as an ordinary local subprocess inside a guarded filesystem directory, not a hardened operating-system sandbox. A production version should execute generated code inside an isolated container with explicit CPU, memory, network, and filesystem limits.

The current implementation is specialized for KuaiRand-Pure and within-user long-view ranking. With additional development, the task contract, dataset schema, evaluator, and knowledge priors could become benchmark plugins supporting the optional KuaiRand-1k and KuaiRand-27k tasks or other public recommendation datasets.

The research catalog identifies promising ranking, causal-history, multitask, watch-time, and temporal approaches, but the final quality remains constrained by the experiment budget and model-generated implementation choices. More time would allow deeper work on efficient sequence models, censored watch-time objectives, exposure-robust validation, and small diverse ensembles.

## Team member contributions

- `[Team member name]`: `[Role and contributions]`
- `[Team member name]`: `[Role and contributions]`
- `[Team member name]`: `[Role and contributions]`
