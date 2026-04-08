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

## 1. Project Summary

`support_triage_env` is a real-world OpenEnv simulation for customer support ticket operations.

It evaluates an agent on tasks that support teams perform daily:
- classify priority and queue correctly
- decide whether escalation is required
- draft policy-compliant replies
- resolve tickets under SLA constraints
- avoid repetitive/invalid behavior

This repository is built for OpenEnv Round 1 evaluation and follows the required `reset()/step()/state()` runtime contract.

## 2. OpenEnv Specification Coverage

This project includes:
- Typed models (Pydantic): `models.py`
- Environment logic: `server/triage_environment.py`
- HTTP service entrypoint: `server/app.py`
- OpenEnv metadata: `openenv.yaml`
- Root baseline script required by hackathon: `inference.py`
- Dockerized runtime: `Dockerfile`

Core runtime endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`
- `GET /schema`

## 3. Action Space

Action model: `TriageAction`

Supported action types:
- `list_tickets`
- `open_ticket`
- `classify`
- `draft_reply`
- `resolve`
- `advance_time`
- `finish`

Key action fields:
- `ticket_id`
- `priority`: `low | medium | high | critical`
- `queue`: `account_access | billing | technical | safety`
- `escalate`: `true | false`
- `tags: list[str]`
- `reply_subject`, `reply_body`
- `resolution`: `answer_and_close | approve_refund | decline_refund | escalate_to_specialist | escalate_to_engineering`
- `minutes` (for `advance_time`)

## 4. Observation and State Space

Observation model: `TriageObservation`

Important fields:
- `task_id`, `difficulty`, `objective`
- `current_time_min`, `active_ticket_id`
- full ticket snapshots (`tickets`)
- `reward_breakdown` with `progress_delta`, `quality_bonus`, `penalty`, `total`
- `progress_score` (dense shaping signal)
- `grader_score` (task-level evaluation score)
- `reward`, `done`, `metadata`

State model: `SupportTriageState` (returned by `state()`).

## 5. Task Suite (Easy to Hard)

Task definitions are deterministic and stored in `tasks.py`.

1. `support_triage_easy` (easy)
- Single account-access issue
- Correct classification, compliant response, on-time closure

2. `support_triage_medium` (medium)
- Two concurrent enterprise tickets (incident + billing correction)
- Requires proper prioritization and escalation handling

3. `support_triage_hard` (hard)
- Security takeover + refund policy edge case + technical export bug
- Requires strong policy discipline and resolve-order control

## 6. Grader Design

Grader implementation: `graders.py`

Properties:
- Deterministic and reproducible
- Programmatic scoring with clear criteria
- Weighted evaluation across:
  - classification correctness
  - tag coverage
  - required/forbidden phrase compliance
  - resolution correctness
  - SLA adherence
  - resolve-order consistency

Important validator rule compliance:
- Final and intermediate exposed score fields are enforced to be strictly inside `(0, 1)` (never `0.0`, never `1.0`).

## 7. Reward Function

Reward shaping is implemented in `server/triage_environment.py`.

Per-step reward includes:
- positive reward from progress improvement
- quality bonuses for correct and policy-safe actions
- penalties for:
  - invalid actions
  - repetitive loops
  - forbidden phrasing
  - poor sequencing/time-wasting behavior

This gives dense trajectory feedback rather than sparse terminal-only reward.

## 8. Baseline Inference (`inference.py`)

The baseline script supports:
- `--agent openai` (OpenAI-compatible API endpoint)
- `--agent rule` (deterministic offline baseline)

### Required env vars for openai mode
- `OPENAI_API_KEY` (required; if missing, `HF_TOKEN` is used as fallback only)
- `API_BASE_URL` (default exists)
- `MODEL_NAME` (default exists)
- `HF_TOKEN` (required by hackathon infra; also used for fallback key handling)

### Structured stdout contract

`inference.py` prints required blocks to stdout:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

This format is used for validator parsing.

### Run baseline

OpenAI-compatible mode:

```bash
python inference.py --agent openai --output outputs/evals/baseline_scores.json
```

Offline deterministic mode:

```bash
python inference.py --agent rule --output outputs/evals/baseline_scores.json
```

Latest deterministic reference:

| Task | Difficulty | Score |
|---|---|---|
| `support_triage_easy` | easy | `0.9999` |
| `support_triage_medium` | medium | `0.9869` |
| `support_triage_hard` | hard | `0.9387` |
| **Average** | - | **`0.9752`** |

## 9. Local Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -U pip
pip install -r server/requirements.txt
pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
```

Run server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate locally:

```bash
openenv validate --verbose
openenv validate --url http://localhost:8000
python validate_submission.py
```

## 10. Docker

Build:

```bash
docker build -t support-triage-openenv .
```

Run:

```bash
docker run --rm -p 8000:8000 support-triage-openenv
```

Validate running container:

```bash
openenv validate --url http://localhost:8000
```

## 11. Hugging Face Space Deployment

Login:

```bash
hf auth login --token <HF_TOKEN>
```

Push:

```bash
openenv push --repo-id <hf-username>/<space-name>
```

Validate deployed runtime:

```bash
openenv validate --url https://<hf-user>-<space-name>.hf.space
```

Set Space Variables/Secrets:
- Variable: `API_BASE_URL`
- Variable: `MODEL_NAME`
- Secret: `OPENAI_API_KEY`
- Secret: `HF_TOKEN`

## 12. Submission Checklist

Before resubmitting, verify:
- `openenv validate --verbose` passes
- Space endpoint validation passes
- `python inference.py --agent rule ...` completes and prints `[START]/[STEP]/[END]`
- each task score is strictly within `(0, 1)`
- `python validate_submission.py` passes (includes reset/probe score-boundary checks)
- `docker build -t support-triage-openenv .` succeeds locally

## 13. Repository Layout

```text
.
|-- openenv.yaml
|-- pyproject.toml
|-- uv.lock
|-- inference.py
|-- models.py
|-- tasks.py
|-- graders.py
|-- client.py
|-- openenv_compat.py
|-- validate_submission.py
|-- Dockerfile
`-- server
    |-- app.py
    |-- triage_environment.py
    |-- requirements.txt
    `-- Dockerfile
```
