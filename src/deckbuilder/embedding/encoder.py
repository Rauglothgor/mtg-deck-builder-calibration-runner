"""Card embedding generation and persistence."""

from __future__ import annotations

from collections.abc import Sequence

from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert

from deckbuilder.db.models import Card, CardEmbedding
from deckbuilder.db.session import get_engine, get_session

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def build_embedding_text(card: Card) -> str:
    """Build the text payload used for card embeddings."""
    oracle_text = card.oracle_text or ""
    return f"{card.name} {card.type_line} {oracle_text}".strip()


def load_model() -> SentenceTransformer:
    """Load the configured sentence-transformers encoder."""
    return SentenceTransformer(MODEL_NAME)


def fetch_cards_without_embeddings(model_name: str) -> list[Card]:
    """Return cards that do not yet have embeddings for the given model."""
    statement: Select[tuple[Card]] = (
        select(Card)
        .outerjoin(
            CardEmbedding,
            (CardEmbedding.oracle_id == Card.oracle_id) & (CardEmbedding.model_name == model_name),
        )
        .where(CardEmbedding.oracle_id.is_(None))
        .order_by(Card.name.asc())
    )
    with get_session() as session:
        result = session.execute(statement)
        return list(result.scalars().all())


def fetch_embedding_count() -> int:
    """Return the total number of stored embeddings."""
    engine = get_engine()
    statement = select(func.count()).select_from(CardEmbedding)
    with engine.connect() as connection:
        return connection.execute(statement).scalar_one()


def fetch_card_count() -> int:
    """Return the total number of cards."""
    engine = get_engine()
    statement = select(func.count()).select_from(Card)
    with engine.connect() as connection:
        return connection.execute(statement).scalar_one()


def chunked[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split a sequence into fixed-size batches."""
    return [items[index : index + size] for index in range(0, len(items), size)]


def encode_cards(cards: Sequence[Card], model: SentenceTransformer) -> list[list[float]]:
    """Encode a batch of cards into dense vectors."""
    texts = [build_embedding_text(card) for card in cards]
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return [vector.tolist() for vector in vectors]


def persist_embeddings(
    cards: Sequence[Card], embeddings: Sequence[list[float]], model_name: str
) -> int:
    """Insert embedding rows idempotently for the provided cards."""
    rows = [
        {
            "oracle_id": card.oracle_id,
            "model_name": model_name,
            "embedding": embedding,
        }
        for card, embedding in zip(cards, embeddings, strict=True)
    ]
    if not rows:
        return 0

    engine = get_engine()
    statement = insert(CardEmbedding).values(rows)
    do_nothing = statement.on_conflict_do_nothing(
        index_elements=[CardEmbedding.oracle_id, CardEmbedding.model_name]
    )
    with engine.begin() as connection:
        connection.execute(do_nothing)
    return len(rows)


def embed_all_cards() -> tuple[int, int]:
    """Generate embeddings for all cards missing them for the configured model."""
    model = load_model()
    cards = fetch_cards_without_embeddings(MODEL_NAME)
    inserted = 0
    for batch in chunked(cards, BATCH_SIZE):
        embeddings = encode_cards(batch, model)
        inserted += persist_embeddings(batch, embeddings, MODEL_NAME)
    return inserted, fetch_embedding_count()


def nearest_neighbors(card_name: str, limit: int = 10) -> list[tuple[str, float]]:
    """Return nearest-neighbor card names for a given card by L2 distance."""
    engine = get_engine()
    base_subquery = (
        select(CardEmbedding.embedding)
        .join(Card, Card.oracle_id == CardEmbedding.oracle_id)
        .where(Card.name == card_name, CardEmbedding.model_name == MODEL_NAME)
        .limit(1)
        .scalar_subquery()
    )
    distance = CardEmbedding.embedding.l2_distance(base_subquery.cast(Vector(384)))
    statement = (
        select(Card.name, distance.label("distance"))
        .join(CardEmbedding, Card.oracle_id == CardEmbedding.oracle_id)
        .where(CardEmbedding.model_name == MODEL_NAME, Card.name != card_name)
        .order_by(distance.asc())
        .limit(limit)
    )
    with engine.connect() as connection:
        result = connection.execute(statement)
        return [(name, float(distance_value)) for name, distance_value in result.all()]
