from pathlib import Path
from uuid import uuid4

from deckbuilder.experiment.structure import (
    DeckStructureDiagnostics,
    StructureCard,
    analyze_structure_cards,
    expected_compounded_mana_spent,
    structural_adjusted_score,
    structural_score_penalty,
    structure_manifest_row,
    write_structure_manifest,
)


def _card(
    name: str,
    *,
    cmc: float,
    type_line: str = "Creature",
    oracle_text: str = "",
) -> StructureCard:
    return StructureCard(
        oracle_id=uuid4(),
        name=name,
        cmc=cmc,
        type_line=type_line,
        oracle_text=oracle_text,
    )


def test_expected_compounded_mana_spent_rewards_castable_curve() -> None:
    curved_deck = [
        *[_card(f"Land {index}", cmc=0, type_line="Basic Land") for index in range(1, 18)],
        *[_card(f"Two Drop {index}", cmc=2) for index in range(1, 13)],
        *[
            _card(f"Ramp {index}", cmc=2, oracle_text="Add one mana of any color.")
            for index in range(1, 6)
        ],
    ]
    clunky_deck = [
        *[_card(f"Land {index}", cmc=0, type_line="Basic Land") for index in range(1, 18)],
        *[_card(f"Seven Drop {index}", cmc=7) for index in range(1, 18)],
    ]

    curved = expected_compounded_mana_spent(curved_deck, trials=50, turns=5, seed=7)
    clunky = expected_compounded_mana_spent(clunky_deck, trials=50, turns=5, seed=7)

    assert curved > clunky


def test_analyze_structure_cards_counts_roles_and_curve() -> None:
    cards = [
        *[_card(f"Land {index}", cmc=0, type_line="Basic Land") for index in range(1, 4)],
        _card("Rampant Growth", cmc=2, oracle_text="Search your library for a basic land card."),
        _card("Divination", cmc=3, oracle_text="Draw two cards."),
        _card("Murder", cmc=3, oracle_text="Destroy target creature."),
        _card("Wrath", cmc=4, oracle_text="Destroy all creatures."),
        _card("Expensive Threat", cmc=7),
    ]

    diagnostics = analyze_structure_cards(cards, "Atraxa, Praetors' Voice", ecms_trials=10)

    assert diagnostics.card_count == 8
    assert diagnostics.land_count == 3
    assert diagnostics.ramp_count == 1
    assert diagnostics.card_draw_count == 1
    assert diagnostics.removal_count == 2
    assert diagnostics.board_wipe_count == 1
    assert diagnostics.low_curve_nonland_count == 1
    assert diagnostics.high_curve_nonland_count == 1
    assert diagnostics.expected_compounded_mana_spent >= 0


def test_write_structure_manifest(tmp_path: Path) -> None:
    cards = [
        _card("Forest", cmc=0, type_line="Basic Land"),
        _card("Rampant Growth", cmc=2, oracle_text="Search your library for a basic land card."),
    ]
    diagnostics = analyze_structure_cards(cards, "Atraxa, Praetors' Voice", ecms_trials=1)
    row = structure_manifest_row(uuid4(), 42, 0.75, diagnostics)

    manifest_path = write_structure_manifest(tmp_path / "report.md", [row])

    text = manifest_path.read_text(encoding="utf-8")
    assert manifest_path.name == "report.structure.csv"
    assert "expected_compounded_mana_spent" in text
    assert "predicted_win_rate" in text
    assert "0.75" in text


def test_structural_score_penalty_flags_extreme_ramp_pile() -> None:
    diagnostics = DeckStructureDiagnostics(
        card_count=99,
        land_count=48,
        nonland_count=51,
        ramp_count=38,
        card_draw_count=8,
        removal_count=6,
        board_wipe_count=2,
        win_condition_count=5,
        average_nonland_cmc=4.2,
        median_nonland_cmc=4.0,
        low_curve_nonland_count=8,
        high_curve_nonland_count=15,
        expected_compounded_mana_spent=55.0,
    )

    penalty = structural_score_penalty(diagnostics)

    assert penalty == 0.65
    assert structural_adjusted_score(1.0, diagnostics) == 0.35


def test_structural_score_penalty_leaves_plausible_shell_near_raw_score() -> None:
    diagnostics = DeckStructureDiagnostics(
        card_count=99,
        land_count=37,
        nonland_count=62,
        ramp_count=14,
        card_draw_count=12,
        removal_count=10,
        board_wipe_count=3,
        win_condition_count=8,
        average_nonland_cmc=3.1,
        median_nonland_cmc=3.0,
        low_curve_nonland_count=18,
        high_curve_nonland_count=7,
        expected_compounded_mana_spent=64.0,
    )

    assert structural_score_penalty(diagnostics) == 0.0
    assert structural_adjusted_score(0.82, diagnostics) == 0.82
