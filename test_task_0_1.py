#!/usr/bin/env python
"""
Comprehensive test suite for Task 0.1 validation
Tests environment setup, configuration, database connectivity and Python version
"""
import os
import sys
import unittest
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("task_0_1_test_suite")

class Task01TestSuite(unittest.TestCase):
    """Test suite for Task 0.1 functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Setting up Task 0.1 test suite")
        load_dotenv()
        cls.project_dir = Path(__file__).parent
        
    def test_env_file_exists(self):
        """Test .env file exists"""
        env_path = self.project_dir / '.env'
        self.assertTrue(env_path.exists(), ".env file not found")
        logger.info("✅ .env file exists")
        
    def test_secret_key_configured(self):
        """Test SECRET_KEY is properly configured"""
        secret_key = os.getenv('SECRET_KEY')
        self.assertIsNotNone(secret_key, "SECRET_KEY environment variable not set")
        self.assertTrue(len(secret_key) >= 16, "SECRET_KEY should be at least 16 characters")
        logger.info("✅ SECRET_KEY is properly configured")
        
    def test_database_connection_string(self):
        """Test database connection string is configured"""
        db_connection = os.getenv('DB_CONNECTION_STRING') or os.getenv('DATABASE_URL')
        self.assertIsNotNone(db_connection, "Database connection string not found")
        logger.info(f"✅ Database connection string found: {db_connection[:20]}...")
        
    def test_feature_flags(self):
        """Test core feature flags are correctly set"""
        core_onboarding = os.getenv('FEATURE_ENABLE_CORE_ONBOARDING')
        core_hta = os.getenv('FEATURE_ENABLE_CORE_HTA')
        core_task_engine = os.getenv('FEATURE_ENABLE_CORE_TASK_ENGINE')
        
        self.assertEqual(core_onboarding, 'true', "FEATURE_ENABLE_CORE_ONBOARDING should be 'true'")
        self.assertEqual(core_hta, 'true', "FEATURE_ENABLE_CORE_HTA should be 'true'")
        self.assertEqual(core_task_engine, 'true', "FEATURE_ENABLE_CORE_TASK_ENGINE should be 'true'")
        
        logger.info(f"✅ Feature flags correctly configured: " 
                   f"CORE_ONBOARDING={core_onboarding}, "
                   f"CORE_HTA={core_hta}, "
                   f"CORE_TASK_ENGINE={core_task_engine}")
        
    def test_python_version(self):
        """Test Python version is at least 3.9"""
        python_version = sys.version_info
        self.assertTrue(python_version.major >= 3 and python_version.minor >= 9,
                      f"Python version should be at least 3.9, found {python_version.major}.{python_version.minor}")
        logger.info(f"✅ Python version check passed: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
    def test_runtime_txt(self):
        """Test runtime.txt contains the correct Python version"""
        runtime_path = self.project_dir / 'runtime.txt'
        self.assertTrue(runtime_path.exists(), "runtime.txt file not found")
        
        with open(runtime_path, 'r') as f:
            runtime_content = f.read().strip()
            
        self.assertTrue(runtime_content.startswith('python-3.11'), 
                       f"runtime.txt should specify Python 3.11.x, found: {runtime_content}")
        logger.info(f"✅ runtime.txt contains correct Python version: {runtime_content}")
        
    def test_documentation_files(self):
        """Test required documentation files exist"""
        doc_files = [
            "PERFORMANCE_STANDARDS.md",
            "DEVELOPER_QUICKSTART.md",
            "DATA_VALIDATION_CATALOG.md"
        ]
        
        for doc_file in doc_files:
            file_path = self.project_dir / doc_file
            self.assertTrue(file_path.exists(), f"Documentation file not found: {doc_file}")
            
            # Check file has content
            self.assertTrue(file_path.stat().st_size > 0, f"Documentation file is empty: {doc_file}")
            
        logger.info("✅ All documentation files exist and have content")
        
    def test_database_connection(self):
        """Test database connection actually works"""
        db_connection = os.getenv('DB_CONNECTION_STRING') or os.getenv('DATABASE_URL')
        self.assertIsNotNone(db_connection, "Database connection string not found")
        
        try:
            engine = create_engine(db_connection)
            with engine.connect() as connection:
                # Simple query to check connection
                result = connection.execute(text("SELECT 1"))
                self.assertEqual(result.scalar(), 1, "Database query failed")
                
                # Verify tables exist
                result = connection.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"
                ))
                self.assertTrue(result.scalar(), "users table not found in database")
                
                # Check other core tables
                core_tables = ['hta_nodes', 'hta_trees', 'memory_snapshots', 
                              'reflection_logs', 'task_footprints']
                
                for table in core_tables:
                    result = connection.execute(text(
                        f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')"
                    ))
                    self.assertTrue(result.scalar(), f"{table} table not found in database")
                    
            logger.info("✅ Database connection successful, all required tables exist")
        except SQLAlchemyError as e:
            self.fail(f"Database connection error: {str(e)}")
            
    def test_settings_file(self):
        """Test settings.py has proper secret key handling"""
        settings_path = self.project_dir / 'forest_app' / 'config' / 'settings.py'
        self.assertTrue(settings_path.exists(), "settings.py file not found")
        
        with open(settings_path, 'r') as f:
            settings_content = f.read()
            
        # Check for environment variable usage for SECRET_KEY
        self.assertTrue('os.getenv' in settings_content and 'SECRET_KEY' in settings_content, 
                       "settings.py should use environment variable for SECRET_KEY")
        
        logger.info("✅ settings.py properly handles SECRET_KEY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
