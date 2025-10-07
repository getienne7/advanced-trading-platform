"""
Database configuration for Strategy Marketplace Service
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

# Database URL from environment variable
DATABASE_URL = os.getenv(
    "STRATEGY_MARKETPLACE_DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/strategy_marketplace"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database tables
    """
    from .models import Base
    Base.metadata.create_all(bind=engine)

def reset_db():
    """
    Reset database (drop and recreate all tables)
    WARNING: This will delete all data!
    """
    from .models import Base
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)