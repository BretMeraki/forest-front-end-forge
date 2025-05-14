"""
Debug script for test_enhanced_hta_service
This will help identify the source of the error.
"""

import asyncio
import uuid
import sys
from datetime import datetime, timezone
import os
import logging
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("debug_suite")

results = []

def record_result(name, success, suggestion=None, error=None):
    results.append({
        "name": name,
        "success": success,
        "suggestion": suggestion,
        "error": error
    })
    if success:
        logger.info(f"[PASS] {name}")
    else:
        logger.error(f"[FAIL] {name}: {error}")
        if suggestion:
            logger.error(f"  Suggestion: {suggestion}")

def check_import(module, name, suggestion=None):
    try:
        __import__(module)
        record_result(f"Import: {name}", True)
        return True
    except Exception as e:
        record_result(f"Import: {name}", False, suggestion, str(e))
        return False

def check_env_var(var, suggestion=None):
    value = os.getenv(var)
    if value:
        record_result(f"Env: {var}", True)
        return True
    else:
        record_result(f"Env: {var}", False, suggestion or f"Set {var} in your .env file.")
        return False

def check_file_exists(path, suggestion=None):
    if os.path.exists(path):
        record_result(f"File: {path}", True)
        return True
    else:
        record_result(f"File: {path}", False, suggestion or f"Create or restore {path}.")
        return False

def check_db_connectivity():
    try:
        from forest_app.persistence.database import engine
        conn = engine.connect()
        conn.close()
        record_result("Database connectivity", True)
        return True
    except Exception as e:
        record_result("Database connectivity", False, "Check DB_CONNECTION_STRING and DB server.", str(e))
        return False

def check_db_migrations():
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "alembic", "current"], capture_output=True, text=True)
        if result.returncode == 0:
            record_result("Alembic migrations", True)
            return True
        else:
            record_result("Alembic migrations", False, "Run 'alembic upgrade head' to apply migrations.", result.stderr)
            return False
    except Exception as e:
        record_result("Alembic migrations", False, "Check alembic installation and config.", str(e))
        return False

def check_feature_flags():
    try:
        from forest_app.core.feature_flags import Feature, is_enabled
        enabled_flags = []
        for feature in Feature:
            if is_enabled(feature):
                enabled_flags.append(feature.name)
        record_result("Feature flags loaded", True)
        logger.info(f"Enabled features: {enabled_flags}")
        return True
    except Exception as e:
        record_result("Feature flags loaded", False, "Check feature flag config and imports.", str(e))
        return False

def check_sentry():
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        record_result("Sentry DSN", False, "Set SENTRY_DSN in .env if you want Sentry error reporting.")
        return False
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn)
        record_result("Sentry SDK init", True)
        return True
    except Exception as e:
        record_result("Sentry SDK init", False, "Check SENTRY_DSN and sentry_sdk install.", str(e))
        return False

def check_llm():
    try:
        from forest_app.core.integrations.llm import LLMClient
        _ = LLMClient()
        record_result("LLMClient import/init", True)
        return True
    except Exception as e:
        record_result("LLMClient import/init", False, "Check LLMClient config and dependencies.", str(e))
        return False

def check_fastapi_health():
    import requests
    url = os.getenv("HEALTH_URL", "http://localhost:8000/health")
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            record_result("FastAPI /health endpoint", True)
            return True
        else:
            record_result("FastAPI /health endpoint", False, f"Check FastAPI server at {url}.", f"Status: {resp.status_code}")
            return False
    except Exception as e:
        record_result("FastAPI /health endpoint", False, f"Check FastAPI server at {url}.", str(e))
        return False

def main():
    logger.info("\n===== Forest OS Comprehensive Debug Suite =====\n")
    load_dotenv()

    # Core imports
    check_import("forest_app.main", "forest_app.main", "Check PYTHONPATH and forest_app install.")
    check_import("forest_app.config.settings", "forest_app.config.settings", "Check config/settings.py.")
    check_import("forest_app.core.feature_flags", "forest_app.core.feature_flags", "Check core/feature_flags.py.")
    check_import("forest_app.persistence.database", "forest_app.persistence.database", "Check DB config and install.")
    check_import("forest_app.core.integrations.llm", "forest_app.core.integrations.llm", "Check LLM integration.")

    # Test helpers
    check_file_exists("forest_app/core/services/test_helpers", "Restore test_helpers directory.")
    check_import("forest_app.core.services.test_helpers.mock_enhanced_hta_service", "mock_enhanced_hta_service", "Check test_helpers/mock_enhanced_hta_service.py.")

    # Env vars
    check_env_var("SECRET_KEY")
    check_env_var("DB_CONNECTION_STRING")
    check_env_var("GOOGLE_API_KEY")

    # DB
    check_db_connectivity()
    check_db_migrations()

    # Feature flags
    check_feature_flags()

    # Sentry
    check_sentry()

    # LLM
    check_llm()

    # FastAPI health (optional)
    try:
        import requests
        check_fastapi_health()
    except ImportError:
        logger.warning("requests not installed, skipping FastAPI /health check.")

    # Summary
    logger.info("\n===== Debug Suite Summary =====\n")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        logger.info(f"{status:4} | {r['name']}")
        if not r["success"] and r["suggestion"]:
            logger.info(f"      Suggestion: {r['suggestion']}")
        if not r["success"] and r["error"]:
            logger.info(f"      Error: {r['error']}")
    logger.info("\n===== End of Debug Suite =====\n")

if __name__ == "__main__":
    main()
