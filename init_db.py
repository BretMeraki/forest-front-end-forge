#!/usr/bin/env python
"""
Database initialization script for basic schema creation.
"""
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
from forest_app.persistence.models import Base

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_db():
    """Initialize the database with core models."""
    load_dotenv()
    
    # Get connection string from environment variables
    db_url = os.environ.get('DB_CONNECTION_STRING') or os.environ.get('DATABASE_URL')
    
    if not db_url:
        logger.error("Database connection string not found in environment variables")
        return False
    
    logger.info(f"Using database URL: {db_url[:20]}...")
    
    try:
        engine = create_engine(db_url)
        
        # Create tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        
        logger.info("Database initialized successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    success = initialize_db()
    exit(0 if success else 1)
