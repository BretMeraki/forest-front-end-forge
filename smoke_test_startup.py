import os
import sys
import logging
import importlib
from contextlib import suppress

REQUIRED_ENV_VARS = [
    "GOOGLE_API_KEY",
    "DB_CONNECTION_STRING",
    "SECRET_KEY"
]

CRITICAL_MODULES = [
    "forest_app.main",
    "forest_app.core.orchestrator",
    "forest_app.core.services.enhanced_hta_service",
    "forest_app.core.feature_flags",
    "forest_app.persistence.database",
    "forest_app.core.integrations.discovery_integration",
    "forest_app.routers.hta",
    "forest_app.routers.users",
    "forest_app.routers.goals",
    "forest_app.routers.snapshots",
    "forest_app.routers.core",
    "forest_app.routers.auth",
    "forest_app.routers.onboarding",
    "forest_app.api.routers.discovery_journey",
]

def check_env_vars():
    print("[ENV] Checking required environment variables...")
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        print(f"✗ Missing environment variables: {missing}")
        return False
    print("✓ All required environment variables are set.")
    return True

def check_imports():
    print("[IMPORTS] Checking critical module imports...")
    ok = True
    for mod in CRITICAL_MODULES:
        try:
            importlib.import_module(mod)
            print(f"✓ Imported {mod}")
        except Exception as e:
            print(f"✗ Failed to import {mod}: {e}")
            ok = False
    return ok

def check_db():
    print("[DB] Checking database connectivity...")
    try:
        from forest_app.persistence.database import get_db
        from sqlalchemy import text
        db = next(get_db())
        db.execute(text("SELECT 1"))
        print("✓ Database connection OK")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def check_di_container():
    print("[DI] Checking DI container initialization...")
    try:
        from forest_app.containers import init_container
        container = init_container()
        assert container is not None
        # Check a few key providers
        for provider in ["llm_client", "hta_service", "enhanced_hta_service", "event_bus"]:
            if not hasattr(container, provider):
                print(f"✗ DI container missing provider: {provider}")
                return False
        print("✓ DI container initialized and key providers present.")
        return True
    except Exception as e:
        print(f"✗ DI container failed: {e}")
        return False

def check_feature_flags():
    print("[FEATURE FLAGS] Checking feature flag status...")
    try:
        from forest_app.core.feature_flags import Feature, is_enabled
        flags = [attr for attr in dir(Feature) if attr.isupper()]
        for flag in flags:
            enabled = is_enabled(getattr(Feature, flag))
            print(f"  - {flag}: {'ENABLED' if enabled else 'disabled'}")
        print("✓ Feature flags checked.")
        return True
    except Exception as e:
        print(f"✗ Feature flag check failed: {e}")
        return False

def check_integrations():
    print("[INTEGRATIONS] Checking LLM, cache, and event bus initialization...")
    ok = True
    try:
        from forest_app.containers import init_container
        container = init_container()
        llm = container.llm_client()
        if not llm:
            print("✗ LLM client not initialized.")
            ok = False
        else:
            print("✓ LLM client initialized.")
    except Exception as e:
        print(f"✗ LLM client check failed: {e}")
        ok = False
    try:
        cache = container.cache_service()
        if not cache:
            print("✗ Cache service not initialized.")
            ok = False
        else:
            print("✓ Cache service initialized.")
    except Exception as e:
        print(f"✗ Cache service check failed: {e}")
        ok = False
    try:
        event_bus = container.event_bus()
        if not event_bus:
            print("✗ Event bus not initialized.")
            ok = False
        else:
            print("✓ Event bus initialized.")
    except Exception as e:
        print(f"✗ Event bus check failed: {e}")
        ok = False
    return ok

def check_fastapi_routes():
    print("[FASTAPI] Checking FastAPI app and routes...")
    try:
        from forest_app.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        # Check docs, openapi, and a few key endpoints
        endpoints = ["/docs", "/openapi.json", "/discovery/progress-summary", "/discovery/patterns"]
        ok = True
        for ep in endpoints:
            resp = client.get(ep)
            if resp.status_code >= 500:
                print(f"✗ Endpoint {ep} returned {resp.status_code}")
                ok = False
            else:
                print(f"✓ Endpoint {ep} returned {resp.status_code}")
        return ok
    except Exception as e:
        print(f"✗ FastAPI route check failed: {e}")
        return False

def main():
    print("=== Forest Startup Smoke Test ===")
    ok = True
    ok &= check_env_vars()
    ok &= check_imports()
    ok &= check_db()
    ok &= check_di_container()
    ok &= check_feature_flags()
    ok &= check_integrations()
    ok &= check_fastapi_routes()
    if ok:
        print("\nAll startup checks passed! Forest backend is ready for deployment.")
        sys.exit(0)
    else:
        print("\nStartup checks failed. See above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 