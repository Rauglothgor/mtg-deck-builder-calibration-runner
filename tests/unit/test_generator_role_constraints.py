from uuid import uuid4

from deckbuilder.generator.roles import (
    LANDS,
    RAMP,
    ROLE_QUOTAS,
    CardProfile,
    count_roles,
    detect_roles,
)
from deckbuilder.generator.search import (
    _can_add_without_role_overflow,
    _meets_role_constraints,
)


def _profile(name: str, *, type_line: str = "Creature", oracle_text: str = "") -> CardProfile:
    return CardProfile(
        oracle_id=uuid4(),
        name=name,
        color_identity=(),
        type_line=type_line,
        oracle_text=oracle_text,
        legality_commander="legal",
        is_commander_legal_as_commander=False,
    )


def test_can_add_without_role_overflow_rejects_extra_land() -> None:
    lands = [
        _profile(f"Land {index}", type_line="Basic Land")
        for index in range(ROLE_QUOTAS["lands"].maximum)
    ]
    extra_land = _profile("Extra Land", type_line="Basic Land")
    current_counts = count_roles(lands, "Atraxa, Praetors' Voice")

    assert not _can_add_without_role_overflow(
        extra_land,
        current_counts,
        "Atraxa, Praetors' Voice",
    )


def test_mana_land_does_not_count_as_ramp() -> None:
    land = _profile("Command Tower", type_line="Land", oracle_text="Add one mana of any color.")

    roles = detect_roles(land, "Atraxa, Praetors' Voice")

    assert LANDS in roles
    assert RAMP not in roles


def test_natural_language_mana_creature_counts_as_ramp() -> None:
    creature = _profile("Mana Dork", oracle_text="Add one mana of any color.")

    roles = detect_roles(creature, "Atraxa, Praetors' Voice")

    assert RAMP in roles


def test_meets_role_constraints_rejects_ramp_overflow() -> None:
    cards = [
        *[
            _profile(f"Land {index}", type_line="Basic Land")
            for index in range(ROLE_QUOTAS["lands"].minimum)
        ],
        *[
            _profile(f"Ramp {index}", oracle_text="Add {G}.")
            for index in range(ROLE_QUOTAS[RAMP].maximum + 1)
        ],
        *[
            _profile(f"Draw {index}", oracle_text="Draw two cards.")
            for index in range(ROLE_QUOTAS["card_draw"].minimum)
        ],
        *[
            _profile(f"Removal {index}", oracle_text="Destroy target creature.")
            for index in range(ROLE_QUOTAS["removal"].minimum)
        ],
        *[
            _profile(f"Wipe {index}", oracle_text="Each other creature gets -1/-1.")
            for index in range(ROLE_QUOTAS["board_wipe"].minimum)
        ],
        *[_profile(name) for name in ["Doubling Season", "Deepglow Skate", "Inexorable Tide"]],
    ]

    counts = count_roles(cards, "Atraxa, Praetors' Voice")

    assert counts[RAMP] == ROLE_QUOTAS[RAMP].maximum + 1
    assert not _meets_role_constraints(cards, "Atraxa, Praetors' Voice")
