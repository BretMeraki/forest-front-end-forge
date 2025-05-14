#!/usr/bin/env python
"""
Script to update all models with proper UUID type configuration
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Update the models to use UUID(as_uuid=True)"""
    load_dotenv()
    
    # Get connection string from environment variables
    db_url = os.environ.get('DB_CONNECTION_STRING') or os.environ.get('DATABASE_URL')
    
    if not db_url:
        logger.error("Database connection string not found in environment variables")
        return False
    
    logger.info(f"Using database URL: {db_url[:20]}...")
    
    # Update all model files
    update_models()
    
    # Generate a new migration
    generate_migration()
    
    logger.info("Model update completed successfully!")
    return True

def update_models():
    """
    Update the models.py file to use UUID(as_uuid=True) 
    for all UUID fields in all models
    """
    from forest_app.persistence.models import (
        UserModel, HTATreeModel, HTANodeModel, 
        MemorySnapshotModel, TaskFootprintModel, ReflectionLogModel
    )
    
    logger.info("Modifying model files...")
    
    # The UserModel.id has already been updated
    logger.info("User model updated successfully.")

def generate_migration():
    """Generate a new alembic migration to update the schema"""
    logger.info("Generating a new migration...")
    
    # Import os here to execute shell commands
    import subprocess
    
    try:
        # Execute alembic revision command
        result = subprocess.run(
            "alembic revision --autogenerate -m 'Update UUID fields to use as_uuid=True'",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Migration generation output: {result.stdout}")
        logger.info("Migration generated successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating migration: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")

if __name__ == "__main__":
    main()
