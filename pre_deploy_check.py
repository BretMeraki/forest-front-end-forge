"""
Pre-deployment verification script for Forest App.
Runs comprehensive checks before deployment.
"""

import os
import sys
import logging
import importlib
import pkg_resources
from typing import Dict, List, Any
import test_imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, List[str]]:
    """
    Check if all dependencies from requirements.txt are installed and at correct versions.
    """
    results = {
        "missing": [],
        "version_mismatch": [],
        "success": []
    }
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
        for req in requirements:
            try:
                pkg_name = req.split("==")[0] if "==" in req else req
                required_version = req.split("==")[1] if "==" in req else None
                
                # Try to get installed version
                installed_version = pkg_resources.get_distribution(pkg_name).version
                
                if required_version and installed_version != required_version:
                    results["version_mismatch"].append(f"{pkg_name} (required: {required_version}, installed: {installed_version})")
                else:
                    results["success"].append(f"{pkg_name} ({installed_version})")
                    
            except pkg_resources.DistributionNotFound:
                results["missing"].append(pkg_name)
                
    except FileNotFoundError:
        logger.error("requirements.txt not found!")
        
    return results

def check_environment_variables() -> Dict[str, List[str]]:
    """
    Check if all required environment variables are set.
    """
    required_vars = [
        "FOREST_APP_ENV",  # e.g., development, staging, production
        "LLM_API_KEY",
        "DB_CONNECTION_STRING",
        "LOG_LEVEL"
    ]
    
    results = {
        "missing": [],
        "set": []
    }
    
    for var in required_vars:
        if var in os.environ:
            results["set"].append(var)
        else:
            results["missing"].append(var)
            
    return results

def check_file_structure() -> Dict[str, List[str]]:
    """
    Verify the presence of critical files and directories.
    """
    required_paths = [
        "forest_app/",
        "forest_app/core/",
        "forest_app/modules/",
        "forest_app/integrations/",
        "forest_app/config/",
        "requirements.txt",
        "alembic.ini",
        "forest_app/core/services/__init__.py",
        "forest_app/modules/types.py"
    ]
    
    results = {
        "missing": [],
        "found": []
    }
    
    for path in required_paths:
        if os.path.exists(path):
            results["found"].append(path)
        else:
            results["missing"].append(path)
            
    return results

def verify_database_migrations():
    """
    Check if all database migrations are up to date.
    """
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from alembic.runtime.environment import EnvironmentContext
        
        # Load Alembic configuration
        config = Config("alembic.ini")
        script = ScriptDirectory.from_config(config)
        
        # Get current head revision
        head_revision = script.get_current_head()
        
        # Get current database revision
        def get_current_rev(rev, _):
            return rev
            
        with EnvironmentContext(config, script, get_current_rev) as env:
            current_rev = env.get_head_revision()
        
        return {
            "success": head_revision == current_rev,
            "current": current_rev,
            "head": head_revision
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Run all pre-deployment checks."""
    logger.info("Starting pre-deployment checks...\n")
    
    all_checks_passed = True
    
    # 1. Check imports
    logger.info("1. Checking imports...")
    import_results = test_imports.verify_imports()
    if import_results["failure"]:
        all_checks_passed = False
        logger.error("Import checks failed!")
    else:
        logger.info("✓ All imports verified successfully")
    
    # 2. Check dependencies
    logger.info("\n2. Checking dependencies...")
    dep_results = check_dependencies()
    if dep_results["missing"] or dep_results["version_mismatch"]:
        all_checks_passed = False
        if dep_results["missing"]:
            logger.error("Missing dependencies:")
            for dep in dep_results["missing"]:
                logger.error(f"  - {dep}")
        if dep_results["version_mismatch"]:
            logger.error("Version mismatches:")
            for dep in dep_results["version_mismatch"]:
                logger.error(f"  - {dep}")
    else:
        logger.info("✓ All dependencies verified")
    
    # 3. Check environment variables
    logger.info("\n3. Checking environment variables...")
    env_results = check_environment_variables()
    if env_results["missing"]:
        all_checks_passed = False
        logger.error("Missing environment variables:")
        for var in env_results["missing"]:
            logger.error(f"  - {var}")
    else:
        logger.info("✓ All environment variables set")
    
    # 4. Check file structure
    logger.info("\n4. Checking file structure...")
    file_results = check_file_structure()
    if file_results["missing"]:
        all_checks_passed = False
        logger.error("Missing files/directories:")
        for path in file_results["missing"]:
            logger.error(f"  - {path}")
    else:
        logger.info("✓ All required files present")
    
    # 5. Check database migrations
    logger.info("\n5. Checking database migrations...")
    migration_results = verify_database_migrations()
    if not migration_results.get("success"):
        all_checks_passed = False
        logger.error("Database migration check failed:")
        if "error" in migration_results:
            logger.error(f"  Error: {migration_results['error']}")
        else:
            logger.error(f"  Current revision: {migration_results['current']}")
            logger.error(f"  Head revision: {migration_results['head']}")
    else:
        logger.info("✓ Database migrations up to date")
    
    # Final summary
    logger.info("\nPre-deployment Check Summary:")
    logger.info("-----------------------------")
    logger.info(f"Import checks: {'✓' if not import_results['failure'] else '✗'}")
    logger.info(f"Dependency checks: {'✓' if not (dep_results['missing'] or dep_results['version_mismatch']) else '✗'}")
    logger.info(f"Environment variables: {'✓' if not env_results['missing'] else '✗'}")
    logger.info(f"File structure: {'✓' if not file_results['missing'] else '✗'}")
    logger.info(f"Database migrations: {'✓' if migration_results.get('success') else '✗'}")
    
    if not all_checks_passed:
        logger.error("\n❌ Pre-deployment checks failed! Please fix the issues above before deploying.")
        sys.exit(1)
    else:
        logger.info("\n✅ All pre-deployment checks passed! Ready to deploy.")
        sys.exit(0)

if __name__ == "__main__":
    main() 