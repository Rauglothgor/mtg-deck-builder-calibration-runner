"""Acceptance test for the fitted AWR surrogate."""

from deckbuilder.surrogate.awr import run_acceptance_check


def test_top_quartile_corpus_deck_scores_above_random_sample() -> None:
    """A top-quartile training deck should beat a deterministic random 99-card sample."""
    result = run_acceptance_check("Atraxa, Praetors' Voice")
    assert result.passed, (
        f"expected top-quartile corpus deck score {result.top_quartile_score:.6f} "
        f"to exceed random sample score {result.random_score:.6f} "
        f"(seed={result.random_seed}, source={result.top_quartile_source})"
    )
