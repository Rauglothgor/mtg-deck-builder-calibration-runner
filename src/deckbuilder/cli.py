"""CLI entrypoints for the deck builder experiment."""

from pathlib import Path
from typing import Annotated, cast

import typer
from alembic import command
from alembic.config import Config

from deckbuilder.config import get_settings
from deckbuilder.db.session import reset_database
from deckbuilder.experiment.forge_calibrator import (
    ScoreField,
    evaluate_outcome_model,
    fit_empirical_calibrator,
    fit_forge_outcome_model,
    load_empirical_calibrator,
    load_forge_outcome_model,
    load_observations_from_artifacts,
    load_outcome_observations_from_artifacts,
    write_empirical_calibrator,
    write_forge_outcome_model,
)
from deckbuilder.experiment.orchestrator import run_experiment, run_score_band_experiment
from deckbuilder.forge.runner import run_sim
from deckbuilder.ingest.archidekt import collect_archidekt_corpus
from deckbuilder.ingest.corpus import ingest_corpus, write_corpus_ingest_report
from deckbuilder.ingest.scryfall import ingest_scryfall
from deckbuilder.surrogate.awr import fit_awr

app = typer.Typer(help="AI Commander Deck Builder v0.5 CLI")
db_app = typer.Typer(help="Database management commands")
ingest_app = typer.Typer(help="Data ingestion commands")
embed_app = typer.Typer(help="Embedding commands")
surrogate_app = typer.Typer(help="Surrogate model commands")
forge_app = typer.Typer(help="Forge native runtime commands")
experiment_app = typer.Typer(help="Experiment orchestration commands")

app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(embed_app, name="embed")
app.add_typer(surrogate_app, name="surrogate")
app.add_typer(forge_app, name="forge")
app.add_typer(experiment_app, name="experiment")


@app.callback()
def main_callback() -> None:
    """Run the deck builder CLI."""


def get_alembic_config() -> Config:
    """Load the local Alembic configuration file."""
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    return config


@db_app.command("init")
def db_init() -> None:
    """Apply all Alembic migrations to the configured database."""
    command.upgrade(get_alembic_config(), "head")


@db_app.command("reset")
def db_reset() -> None:
    """Drop and recreate the public schema, then reapply migrations."""
    reset_database()
    command.upgrade(get_alembic_config(), "head")


@ingest_app.command("scryfall")
def ingest_scryfall_command() -> None:
    """Fetch and load Scryfall bulk oracle cards into the database."""
    path, count = ingest_scryfall()
    typer.echo(f"Ingested {count} cards from {path}")


@ingest_app.command("archidekt")
def ingest_archidekt_command(
    commander: Annotated[
        str,
        typer.Option(help="Commander name to validate against."),
    ],
    target: Annotated[
        int,
        typer.Option(help="Maximum accepted deck rows to collect."),
    ],
    output: Annotated[
        Path,
        typer.Option(help="Output CSV path for the collected corpus."),
    ],
) -> None:
    """Collect a bounded Commander corpus from Archidekt into a CSV file."""
    artifacts = collect_archidekt_corpus(
        commander_name=commander,
        target=target,
        output_path=output,
    )
    typer.echo(f"Collected Archidekt corpus to {artifacts.csv_path}")
    typer.echo(f"State saved to {artifacts.state_path}")
    typer.echo(f"Report saved to {artifacts.report_path}")


@ingest_app.command("corpus")
def ingest_corpus_command(csv_path: Path) -> None:
    """Load a collected CSV corpus into training_decks."""
    report = ingest_corpus(csv_path)
    report_path = write_corpus_ingest_report(
        report=report,
        training_deck_row_count=report.inserted_row_count,
    )
    typer.echo(f"Ingested {report.inserted_row_count} deck rows from {csv_path}")
    typer.echo(f"Report saved to {report_path}")


@surrogate_app.command("fit")
def surrogate_fit_command(commander_name: str) -> None:
    """Fit the AWR surrogate for one commander corpus."""
    result = fit_awr(commander_name)
    typer.echo(f"Fit commander {result.commander_name}")
    typer.echo(f"fit_run_id={result.fit_run_id}")
    typer.echo(f"decks={result.deck_count}")
    typer.echo(f"unique_cards={result.unique_card_count}")
    typer.echo(f"synergy_pairs={len(result.synergy_scores)}")


@embed_app.command("cards")
def embed_cards_command() -> None:
    """Generate embeddings for all cards missing them for the configured model."""
    from deckbuilder.embedding.encoder import MODEL_NAME, embed_all_cards

    inserted, total = embed_all_cards()
    typer.echo(f"Embedded {inserted} cards with {MODEL_NAME}; total embeddings now {total}")


@embed_app.command("neighbors")
def embed_neighbors_command(card_name: str, limit: int = 10) -> None:
    """Print nearest-neighbor cards for a named card."""
    from deckbuilder.embedding.encoder import nearest_neighbors

    for neighbor_name, distance in nearest_neighbors(card_name, limit=limit):
        typer.echo(f"{neighbor_name}\t{distance:.6f}")


def _resolve_forge_deck(deck: str | Path, bundled_deck_dir: Path) -> Path:
    """Resolve a Forge deck path from either a direct path or bundled filename."""
    deck_path = Path(deck).expanduser()
    if deck_path.is_file():
        return deck_path

    bundled_path = bundled_deck_dir / str(deck)
    if bundled_path.is_file():
        return bundled_path

    msg = f"Forge deck not found: {deck}"
    raise FileNotFoundError(msg)


@forge_app.command("smoke-test")
def forge_smoke_test_command(
    deck_a: Annotated[
        str,
        typer.Option(help="Candidate-slot deck path or bundled Forge deck filename."),
    ] = "atraxa.dck",
    deck_b: Annotated[
        str,
        typer.Option(help="Opponent-slot deck path or bundled Forge deck filename."),
    ] = "alela.dck",
) -> None:
    """Run one Forge commander simulation between bundled stock decks."""
    settings = get_settings()
    try:
        deck_a_path = _resolve_forge_deck(deck_a, settings.forge_bundled_deck_dir)
        deck_b_path = _resolve_forge_deck(deck_b, settings.forge_bundled_deck_dir)
        result = run_sim(
            deck_a_path,
            deck_b_path,
            n_matches=1,
            seed=settings.default_seed,
        )
    except Exception as exc:
        typer.echo(f"Forge smoke-test failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    winner = result.game_winners[0]
    if winner is None:
        typer.echo("Forge smoke-test winner: draw")
    else:
        typer.echo(f"Forge smoke-test winner: {winner}")


@experiment_app.command("run")
def experiment_run_command(
    commander: Annotated[str, typer.Option(help="Commander name to evaluate.")],
    n_decks: Annotated[
        int,
        typer.Option("--n-decks", help="Number of elite decks to evaluate."),
    ],
    matches: Annotated[int, typer.Option(help="Matches per deck.")],
    opponent: Annotated[
        str,
        typer.Option(help="Opponent deck path or bundled deck filename."),
    ] = "alela.dck",
    output: Annotated[
        Path,
        typer.Option(help="Markdown report output path."),
    ] = Path("/tmp/deckbuilder-calibration.md"),
    seed_start: Annotated[
        int,
        typer.Option("--seed-start", help="First generation seed for this run or shard."),
    ] = 42,
    attempt_cap: Annotated[
        int,
        typer.Option(help="Maximum generation attempts before failing the run."),
    ] = 500,
) -> None:
    """Run the experiment orchestration pipeline."""
    outcome = run_experiment(
        commander_name=commander,
        n_decks=n_decks,
        matches=matches,
        opponent=opponent,
        output=output,
        seed_start=seed_start,
        attempt_cap=attempt_cap,
    )
    typer.echo(f"experiment_run_id={outcome.experiment_run_id}")
    typer.echo(f"decision={outcome.calibration.decision}")
    typer.echo(f"retry_count={outcome.retry_count}")
    typer.echo(f"report_path={outcome.report_path}")


@experiment_app.command("run-score-bands")
def experiment_run_score_bands_command(
    commander: Annotated[str, typer.Option(help="Commander name to evaluate.")],
    n_decks: Annotated[
        int,
        typer.Option("--n-decks", help="Number of score-band decks to evaluate."),
    ],
    matches: Annotated[int, typer.Option(help="Matches per deck.")],
    opponent: Annotated[
        str,
        typer.Option(help="Opponent deck path or bundled deck filename."),
    ] = "alela.dck",
    output: Annotated[
        Path,
        typer.Option(help="Markdown report output path."),
    ] = Path("/tmp/deckbuilder-score-band-calibration.md"),
    seed_start: Annotated[
        int,
        typer.Option("--seed-start", help="First generation seed for this run or shard."),
    ] = 42,
    candidate_pool_size: Annotated[
        int,
        typer.Option(help="Generated candidate pool size before band selection."),
    ] = 100,
    band_count: Annotated[
        int,
        typer.Option(help="Number of rank-based score bands to sample from."),
    ] = 5,
    forge_calibrator: Annotated[
        Path | None,
        typer.Option(
            "--forge-calibrator",
            help="Optional empirical Forge calibrator JSON to apply during selection.",
        ),
    ] = None,
    forge_outcome_model: Annotated[
        Path | None,
        typer.Option(
            "--forge-outcome-model",
            help="Optional learned Forge-outcome model JSON to apply during selection.",
        ),
    ] = None,
    rerank_shortlist_size: Annotated[
        int,
        typer.Option(
            "--rerank-shortlist-size",
            help="Top model-scored candidates to rerank with lightweight Forge sims.",
        ),
    ] = 0,
    rerank_matches: Annotated[
        int,
        typer.Option(
            "--rerank-matches",
            help="Lightweight Forge matches per shortlisted rerank candidate.",
        ),
    ] = 0,
    rerank_prior_weight: Annotated[
        float,
        typer.Option(
            "--rerank-prior-weight",
            help="Equivalent prior match count assigned to the model score during reranking.",
        ),
    ] = 20.0,
) -> None:
    """Run calibration with rank-based surrogate score-band sampling."""
    calibrator = load_empirical_calibrator(forge_calibrator) if forge_calibrator else None
    outcome_model = load_forge_outcome_model(forge_outcome_model) if forge_outcome_model else None
    outcome = run_score_band_experiment(
        commander_name=commander,
        n_decks=n_decks,
        matches=matches,
        opponent=opponent,
        output=output,
        seed_start=seed_start,
        candidate_pool_size=candidate_pool_size,
        band_count=band_count,
        forge_calibrator=calibrator,
        forge_outcome_model=outcome_model,
        rerank_shortlist_size=rerank_shortlist_size,
        rerank_matches=rerank_matches,
        rerank_prior_weight=rerank_prior_weight,
    )
    typer.echo(f"experiment_run_id={outcome.experiment_run_id}")
    typer.echo(f"decision={outcome.calibration.decision}")
    typer.echo(f"retry_count={outcome.retry_count}")
    typer.echo(f"report_path={outcome.report_path}")


@experiment_app.command("fit-calibrator")
def experiment_fit_calibrator_command(
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help=(
                "Directory containing downloaded markdown, .selection.csv, "
                "and .structure.csv artifacts."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="Output path for the empirical Forge calibrator JSON artifact."),
    ],
    score_field: Annotated[
        str, typer.Option(help="Score field to calibrate against Forge outcomes.")
    ] = ("selection_score"),
    bin_count: Annotated[
        int,
        typer.Option("--bin-count", help="Number of empirical score bins."),
    ] = 5,
) -> None:
    """Fit an empirical Forge calibrator from validation artifacts."""
    observations = load_observations_from_artifacts(artifacts_dir)
    parsed_score_field = _parse_score_field(score_field)
    calibrator = fit_empirical_calibrator(
        observations,
        score_field=parsed_score_field,
        bin_count=bin_count,
    )
    write_empirical_calibrator(calibrator, output)
    typer.echo(f"observations={calibrator.source_case_count}")
    typer.echo(f"score_field={calibrator.score_field}")
    typer.echo(f"bins={len(calibrator.bins)}")
    typer.echo(f"calibrator_path={output}")


@experiment_app.command("fit-outcome-model")
def experiment_fit_outcome_model_command(
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help=(
                "Directory containing downloaded markdown, .selection.csv, "
                "and .structure.csv artifacts."
            ),
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="Output path for the learned Forge-outcome model JSON artifact."),
    ],
    base_calibrator: Annotated[
        Path | None,
        typer.Option(
            "--base-calibrator",
            help="Optional empirical Forge calibrator JSON used as the residual base.",
        ),
    ] = None,
    l2_regularization: Annotated[
        float,
        typer.Option("--l2-regularization", help="Ridge regularization strength."),
    ] = 5.0,
) -> None:
    """Fit a learned Forge-outcome model from validation artifacts."""
    observations = load_outcome_observations_from_artifacts(artifacts_dir)
    calibrator = load_empirical_calibrator(base_calibrator) if base_calibrator else None
    model = fit_forge_outcome_model(
        observations,
        base_calibrator=calibrator,
        l2_regularization=l2_regularization,
    )
    evaluation = evaluate_outcome_model(model, observations)
    write_forge_outcome_model(model, output)
    typer.echo(f"observations={model.source_case_count}")
    typer.echo(f"features={len(model.feature_names)}")
    typer.echo(f"training_mad={evaluation.mean_absolute_deviation:.4f}")
    typer.echo(f"training_bias={evaluation.bias:.4f}")
    typer.echo(f"model_path={output}")


@experiment_app.command("evaluate-outcome-model")
def experiment_evaluate_outcome_model_command(
    artifacts_dir: Annotated[
        Path,
        typer.Option(
            "--artifacts-dir",
            help=(
                "Directory containing downloaded markdown, .selection.csv, "
                "and .structure.csv artifacts."
            ),
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option("--model", help="Forge-outcome model JSON artifact to evaluate."),
    ],
) -> None:
    """Evaluate a learned Forge-outcome model against validation artifacts."""
    observations = load_outcome_observations_from_artifacts(artifacts_dir)
    model = load_forge_outcome_model(model_path)
    evaluation = evaluate_outcome_model(model, observations)
    typer.echo(f"observations={evaluation.case_count}")
    typer.echo(f"mean_prediction={evaluation.mean_prediction:.4f}")
    typer.echo(f"mean_actual={evaluation.mean_actual:.4f}")
    typer.echo(f"mad={evaluation.mean_absolute_deviation:.4f}")
    typer.echo(f"bias={evaluation.bias:.4f}")
    typer.echo(f"overconfidence_rate_20={evaluation.overconfidence_rate_20:.4f}")
    typer.echo(f"overconfidence_rate_30={evaluation.overconfidence_rate_30:.4f}")
    typer.echo(f"underconfidence_rate_20={evaluation.underconfidence_rate_20:.4f}")
    typer.echo(f"underconfidence_rate_30={evaluation.underconfidence_rate_30:.4f}")
    typer.echo(f"pearson={evaluation.pearson:.4f}")
    typer.echo(f"spearman={evaluation.spearman:.4f}")


def _parse_score_field(score_field: str) -> ScoreField:
    if score_field not in {"predicted_win_rate", "selection_score"}:
        msg = "score_field must be one of: predicted_win_rate, selection_score"
        raise typer.BadParameter(msg)
    return cast(ScoreField, score_field)


def main() -> None:
    """Console script entrypoint."""
    app()
