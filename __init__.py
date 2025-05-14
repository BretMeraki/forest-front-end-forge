from forest_app.persistence.database import engine, SessionLocal, get_db
from forest_app.persistence.models import Base, MemorySnapshotModel
from forest_app.persistence.repository import MemorySnapshotRepository
from forest_app.core.processors import ReflectionProcessor, CompletionProcessor
from forest_app.core.services import HTAService


def init_db():
    """
    Initializes the database by creating all tables.
    This should be called at application startup if the schema hasn't been created yet.
    """
    Base.metadata.create_all(bind=engine)


__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "Base",
    "MemorySnapshotModel",
    "MemorySnapshotRepository",
    "init_db",
    "ReflectionProcessor",
    "CompletionProcessor",
    "HTAService"
]
