# MTG Deck Builder Public Calibration Runner

This is a stripped-down public runner for the AI Commander Deck Builder v0.5 calibration experiment.

Its purpose is to offload Forge simulation shards to free GitHub Actions runners on a public repository. It contains only code and public-source experiment inputs needed to reproduce the Atraxa calibration run.

## Included

- Python package source under `src/deckbuilder/`
- Alembic migrations
- Tests and fixtures
- `data/raw/atraxa_corpus.csv`, a public-source Atraxa corpus with Archidekt deck URLs and card names
- A manual GitHub Actions workflow at `.github/workflows/calibration-shards.yml`

## Excluded

- Local `.env`
- Virtual environments and caches
- Progress notes from the private/local working session
- Raw Scryfall bulk JSON cache
- Raw Archidekt API payloads
- Prior smoke/calibration reports
- Any local Forge install or database state

## Local Check

```bash
cp .env.example .env
uv sync
docker compose up -d postgres
uv run deckbuilder db init
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## GitHub Actions Calibration

Publish this repository as public, then run:

`Actions` -> `Calibration shards` -> `Run workflow`

Recommended full calibration inputs:

- `shards`: `25`
- `decks_per_shard`: `4`
- `matches`: `100`
- `attempt_cap`: `500`

That evaluates 100 generated decks total. The workflow uses a different `--seed-start` per shard to avoid duplicate generated decks.

Each shard uploads one markdown artifact named `calibration-shard-N`.

## What Each Shard Does

1. Starts a pgvector Postgres service.
2. Installs Python dependencies with `uv`.
3. Installs Java 17 and Forge 2.0.12 into the Actions workspace.
4. Downloads Scryfall oracle cards if the cache is empty.
5. Loads `cards`, ingests `data/raw/atraxa_corpus.csv`, and fits AWR.
6. Runs a Forge smoke test.
7. Runs its assigned calibration shard and uploads the markdown report.

## Cost Notes

GitHub-hosted standard runners are free for public repositories, subject to GitHub Actions usage limits and fair-use behavior. Jobs are capped at 6 hours, so this workflow shards the experiment into smaller jobs and limits parallelism to 5.

## Results

After all shards finish, download the uploaded shard reports from the workflow run. The v0.5 decision should be made from the combined shard results, not from a single shard.

If the artifacts are extracted into `artifacts/`, merge the shard metrics with:

```bash
uv run python scripts/merge_reports.py artifacts
```
