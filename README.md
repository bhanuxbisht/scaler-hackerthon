---
title: Support Triage OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Support Triage OpenEnv

## Overview

`support_triage_env` is a production-oriented OpenEnv simulation for customer support operations.
The environment evaluates agent performance on ticket triage workflows that teams run in real support organizations:
- queue and priority classification
- escalation decisions
- response drafting under policy constraints
- resolution handling under SLA pressure

The environment is API-first and compatible with the OpenEnv `reset/step/state` interaction contract.

## OpenEnv Compliance

This repository implements the required OpenEnv components:
- typed action, observation, reward, and state models (Pydantic): [models.py](./models.py)
- environment runtime: [server/triage_environment.py](./server/triage_environment.py)
- HTTP app entrypoint: [server/app.py](./server/app.py)
- OpenEnv metadata: [openenv.yaml](./openenv.yaml)
- OpenEnv validation support (`openenv validate`) with multi-mode readiness

## Action Space

`TriageAction` supports:
- `list_tickets`
- `open_ticket`
- `classify`
- `draft_reply`
- `resolve`
- `advance_time`
- `finish`

Primary fields:
- `ticket_id`
- `priority`: `low | medium | high | critical`
- `queue`: `account_access | billing | technical | safety`
- `escalate`: `true | false`
- `tags`
- `reply_subject`, `reply_body`
- `resolution`: `answer_and_close | approve_refund | decline_refund | escalate_to_specialist | escalate_to_engineering`
- `minutes` for `advance_time`

## Observation Space

`TriageObservation` returns:
- task metadata (`task_id`, `objective`, `difficulty`)
- simulated time and active ticket context
- complete ticket snapshots
- reward breakdown (`progress_delta`, `quality_bonus`, `penalty`, `total`)
- `progress_score` for trajectory shaping
- `grader_score` in `[0.0, 1.0]`
- standard OpenEnv fields (`reward`, `done`, `metadata`)

## Task Suite and Difficulty Progression

Task definitions are in [tasks.py](./tasks.py).  
Three deterministic tasks are provided:

1. `support_triage_easy` (easy)  
   Single account-access issue with compliant closure.
2. `support_triage_medium` (medium)  
   Concurrent platform incident and billing correction.
3. `support_triage_hard` (hard)  
   Mixed fraud/policy/technical tickets with stricter policy and ordering constraints.

Each task specifies:
- expected classification (priority, queue, escalation)
- expected tagging
- expected resolution action
- required/forbidden response phrases
- SLA target
- optional ordering constraints

## Graders and Scoring

Grader implementation: [graders.py](./graders.py)

Properties:
- deterministic and reproducible
- continuous score output in `[0.0, 1.0]`
- weighted scoring across:
  - classification correctness
  - tag coverage
  - response quality (required/forbidden phrase checks)
  - resolution correctness
  - SLA and resolve-order behavior

## Reward Function

Reward shaping is implemented in [server/triage_environment.py](./server/triage_environment.py).

Per-step reward includes:
- positive signal from partial progress delta
- bonuses for correct triage and policy-compliant execution
- penalties for invalid actions, repetitive loops, policy-unsafe responses, and inefficient time progression

This provides dense learning signal across the full trajectory, not only terminal outcomes.

## Baseline Inference

Required baseline script: [inference.py](./inference.py) (project root).

LLM calls are executed through the OpenAI Python client using:
- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (fallback for key handling in this project)

Run baseline:

```bash
python inference.py --agent openai --output outputs/evals/baseline_scores.json
```

Offline deterministic fallback:

```bash
python inference.py --agent rule --output outputs/evals/baseline_scores.json
```

Reference rule baseline:

| Task | Difficulty | Score |
|---|---|---|
| `support_triage_easy` | easy | `1.0000` |
| `support_triage_medium` | medium | `0.9869` |
| `support_triage_hard` | hard | `0.9387` |
| **Average** | - | **`0.9752`** |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -U pip
pip install -r server/requirements.txt
pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
```

Start server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Local validation:

```bash
openenv validate --verbose
openenv validate --url http://localhost:8000
```

## Docker

```bash
docker build -t support-triage-openenv .
docker run --rm -p 8000:8000 support-triage-openenv
```

## Hugging Face Space Deployment

Authenticate:

```bash
hf auth login --token <HF_TOKEN>
```

Push:

```bash
openenv push --repo-id <hf-username>/<space-name>
```

Validate deployed runtime:

```bash
openenv validate --url https://<space-name>.hf.space
```

Required Space variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY`



## Repository Structure


.
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── inference.py
├── models.py
├── tasks.py
├── graders.py
├── client.py
├── openenv_compat.py
├── validate_submission.py
├── Dockerfile
└── server/
    ├── app.py
    ├── triage_environment.py
    ├── requirements.txt
    └── Dockerfile

