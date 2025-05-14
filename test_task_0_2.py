#!/usr/bin/env python
"""
Comprehensive test suite for Task 0.2 validation
Tests SQLAlchemy models and database schema with UUID and JSONB requirements
"""
import os
import sys
import unittest
import logging
import uuid
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("task_0_2_test_suite")

class Task02TestSuite(unittest.TestCase):
    """Test suite for Task 0.2 functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Setting up Task 0.2 test suite")
        load_dotenv()
        cls.project_dir = Path(__file__).parent
        
        # Get database connection
        db_url = os.environ.get('DB_CONNECTION_STRING') or os.environ.get('DATABASE_URL')
        if not db_url:
            raise ValueError("Database connection string not found")
        
        cls.engine = create_engine(db_url)
        cls.inspector = inspect(cls.engine)
    
    def test_users_table_uuid_primary_key(self):
        """Test users table has UUID primary key"""
        pk_columns = self.inspector.get_pk_constraint("users")
        self.assertEqual(len(pk_columns['constrained_columns']), 1, "users table should have exactly one primary key column")
        self.assertEqual(pk_columns['constrained_columns'][0], "id", "Primary key column should be named 'id'")
        
        columns = {c["name"]: c for c in self.inspector.get_columns("users")}
        self.assertIn("id", columns, "users table should have an 'id' column")
        self.assertEqual(columns["id"]["type"].__visit_name__, "UUID", "id column should be UUID type")
        logger.info("✅ users.id is UUID type")
        
    def test_hta_trees_table_uuid_and_foreign_keys(self):
        """Test hta_trees table has UUID primary key and correct foreign keys"""
        pk_columns = self.inspector.get_pk_constraint("hta_trees")
        self.assertEqual(pk_columns['constrained_columns'][0], "id", "Primary key column should be named 'id'")
        
        columns = {c["name"]: c for c in self.inspector.get_columns("hta_trees")}
        self.assertIn("id", columns, "hta_trees table should have an 'id' column")
        self.assertEqual(columns["id"]["type"].__visit_name__, "UUID", "id column should be UUID type")
        self.assertIn("user_id", columns, "hta_trees table should have a 'user_id' column")
        self.assertEqual(columns["user_id"]["type"].__visit_name__, "UUID", "user_id column should be UUID type")
        
        # Check foreign keys
        fk_constraints = self.inspector.get_foreign_keys("hta_trees")
        user_fk = next((fk for fk in fk_constraints if fk["referred_table"] == "users"), None)
        self.assertIsNotNone(user_fk, "hta_trees should have a foreign key to users")
        self.assertEqual(user_fk["referred_columns"], ["id"], "hta_trees.user_id should reference users.id")
        
        logger.info("✅ hta_trees table has correct UUID and foreign key to users")
    
    def test_manifest_jsonb_field(self):
        """Test hta_trees.manifest is JSONB type"""
        columns = {c["name"]: c for c in self.inspector.get_columns("hta_trees")}
        self.assertIn("manifest", columns, "hta_trees table should have a 'manifest' column")
        self.assertEqual(columns["manifest"]["type"].__visit_name__, "JSONB", "manifest column should be JSONB type")
        logger.info("✅ hta_trees.manifest is JSONB type")
    
    def test_manifest_gin_index(self):
        """Test GIN index exists on hta_trees.manifest"""
        indexes = self.inspector.get_indexes("hta_trees")
        gin_index = next((idx for idx in indexes if idx["name"] == "idx_hta_trees_manifest_gin"), None)
        self.assertIsNotNone(gin_index, "GIN index 'idx_hta_trees_manifest_gin' should exist on hta_trees.manifest")
        self.assertEqual(gin_index["dialect_options"]["postgresql_using"], "gin", "Index should use GIN")
        
        # Verify the index is on manifest field
        self.assertEqual(gin_index["column_names"], ["manifest"], "GIN index should be on manifest column")
        logger.info("✅ GIN index exists on hta_trees.manifest")
    
    def test_hta_nodes_uuid_fields(self):
        """Test hta_nodes table has UUID fields"""
        columns = {c["name"]: c for c in self.inspector.get_columns("hta_nodes")}
        uuid_field_names = ["id", "user_id", "parent_id", "tree_id", "roadmap_step_id"]
        
        for field_name in uuid_field_names:
            self.assertIn(field_name, columns, f"hta_nodes table should have a '{field_name}' column")
            if field_name in columns:
                self.assertEqual(columns[field_name]["type"].__visit_name__, "UUID", 
                               f"{field_name} column should be UUID type")
        
        logger.info("✅ hta_nodes has all UUID fields with correct types")
    
    def test_insert_with_uuid(self):
        """Test inserting a record with UUID values"""
        with self.engine.connect() as connection:
            # Generate test user
            test_id = uuid.uuid4()
            test_email = f"test_{test_id}@example.com"
            
            # Insert test user
            connection.execute(text(
                "INSERT INTO users (id, email, hashed_password, is_active) VALUES (:id, :email, 'test_password', true)"
            ), {"id": test_id, "email": test_email})
            
            # Verify user was inserted
            result = connection.execute(text(
                "SELECT id, email FROM users WHERE id = :id"
            ), {"id": test_id})
            user = result.fetchone()
            
            self.assertIsNotNone(user, "User should be inserted")
            self.assertEqual(str(user.id), str(test_id), "Retrieved UUID should match inserted UUID")
            self.assertEqual(user.email, test_email, "Retrieved email should match inserted email")
            
            # Clean up
            connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": test_id})
            connection.commit()
            
            logger.info("✅ Successfully inserted and retrieved UUID values")

if __name__ == "__main__":
    unittest.main(verbosity=2)
