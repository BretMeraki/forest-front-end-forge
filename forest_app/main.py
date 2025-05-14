# forest_app/main.py (MODIFIED: DI Wiring moved before router inclusion)

import logging
import sys
import os
from typing import Callable, Any # Added Any

# --- Explicitly add /app to sys.path ---
# This helps resolve module imports in some deployment environments
APP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT_DIR not in sys.path:
    sys.path.insert(0, APP_ROOT_DIR)
    sys.path.insert(0, os.path.join(APP_ROOT_DIR, 'forest_app'))
# --- End sys.path modification ---


# --- Sentry Integration Imports ---
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forest_app.middleware.logging import LoggingMiddleware

# --- Core, Persistence & Feature Flag Imports ---
# Keep necessary imports for init_db, security, models etc.
# from forest_app.modules.trigger_phrase import TriggerPhraseHandler # Likely not needed globally now
from forest_app.core.security import initialize_security_dependencies
from forest_app.persistence.database import init_db
from forest_app.persistence.repository import get_user_by_email
from forest_app.persistence.models import UserModel # Keep model import

# --- Import Feature Flags and Checker ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    feature_flags_available = True
except ImportError as ff_import_err:
    # Log this potential error early after basic logging is configured
    logging.getLogger("main_init").error("Failed to import Feature Flags components: %s", ff_import_err)
    feature_flags_available = False
    # Define dummies if needed for code structure, though checks later should handle it
    class Feature: pass
    def is_enabled(feature: Any) -> bool: return False


# --- Use Absolute Import for Container CLASS and INSTANCE --- ### MODIFIED ###
from forest_app.containers import Container, init_container

# --- Enhanced Architecture Components ---
from forest_app.core.initialize_architecture import inject_enhanced_architecture
from forest_app.core.integrations.discovery_integration import setup_discovery_journey

# Initialize container
container = init_container()

# --- Router Imports ---
# Keep these here as they might rely on container/models
from forest_app.routers import auth, users, onboarding, hta, snapshots, core, goals, trees
# Import the new discovery journey router from api/routers
from forest_app.api.routers import discovery_journey


# --------------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------------
log_handler = logging.StreamHandler(sys.stdout)
log_format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
# Use datefmt for ISO8601 format in logs
date_format = '%Y-%m-%dT%H:%M:%S%z' # Example ISO 8601 format

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG, # Keep DEBUG to capture all levels for now
    format=log_format_string,
    datefmt=date_format,
    handlers=[log_handler],
    force=True # Force override if already configured elsewhere
)
logger = logging.getLogger(__name__) # Get logger for this module

# Log path modification confirmation now that logging is configured
if APP_ROOT_DIR not in sys.path: # Check condition again (should be false now)
     pass # Already added
else:
     logger.warning(f"--- Path {APP_ROOT_DIR} was already in sys.path or added. ---")

logger.info("----- Forest OS API Starting Up (DI Enabled) -----")

# --- DEBUG LOGGING LINES ---
try:
    logger.debug(f"--- DEBUG: Current Working Directory: {os.getcwd()}")
    logger.debug(f"--- DEBUG: Python Path (post-modification): {sys.path}")
except Exception as debug_e:
    logger.error(f"--- DEBUG: Failed to get CWD or sys.path: {debug_e}")
# --- END DEBUG LOGGING LINES ---


# --------------------------------------------------------------------------
# Sentry Integration
# --------------------------------------------------------------------------
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,      # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events (or WARNING)
        )
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")), # Sample fewer traces usually
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            integrations=[SqlalchemyIntegration(), sentry_logging], # Use configured logging integration
            environment=os.getenv("APP_ENV", "development"),
            release=os.getenv("APP_RELEASE_VERSION", "unknown"), # Provide a default release
        )
        logger.info("Sentry SDK initialized successfully.")
    except Exception as sentry_init_e: logger.exception("Failed to initialize Sentry SDK: %s", sentry_init_e)
else: logger.warning("SENTRY_DSN environment variable not found. Sentry integration skipped.")

# --------------------------------------------------------------------------
# Database Initialization
# --------------------------------------------------------------------------
try:
    logger.info("Attempting to initialize database tables via init_db()...")
    init_db()
    logger.info("Database initialization check complete.")
except Exception as db_init_e:
    logger.exception("CRITICAL Error during database initialization: %s", db_init_e)
    sys.exit(f"CRITICAL: Database initialization failed: {db_init_e}")

# --------------------------------------------------------------------------
# Security Dependency Initialization
# --------------------------------------------------------------------------
logger.info("Initializing security dependencies...")
try:
    # Basic check if UserModel seems valid (has an email annotation)
    if not hasattr(UserModel, '__annotations__') or 'email' not in UserModel.__annotations__:
         logger.critical("UserModel may be incomplete or a dummy class. Security init might fail.")
    initialize_security_dependencies(get_user_by_email, UserModel)
    logger.info("Security dependencies initialized successfully.")
except TypeError as sec_init_err:
     logger.exception(f"CRITICAL: Failed security init - check function signature in core.security: {sec_init_err}")
     sys.exit(f"CRITICAL: Security dependency initialization failed (TypeError): {sec_init_err}")
except Exception as sec_init_gen_err:
     logger.exception(f"CRITICAL: Unexpected error during security init: {sec_init_gen_err}")
     sys.exit(f"CRITICAL: Security dependency initialization failed unexpectedly: {sec_init_gen_err}")


# --------------------------------------------------------------------------
# --- DI Container Setup (Instance created on import) --- ### COMMENT UPDATED ###
# --------------------------------------------------------------------------
# Container instance is created when 'from forest_app.containers import container' runs
if not container: # Basic check
     logger.critical("CRITICAL: DI Container instance is None after import.")
     sys.exit("CRITICAL: Failed to get DI Container instance.")
else:
     logger.info("DI Container instance imported successfully.")


# --------------------------------------------------------------------------
# FastAPI Application Instance Creation
# --------------------------------------------------------------------------
logger.info("Creating FastAPI application instance...")
app = FastAPI(
    title="Forest OS API",
    version="1.23",
    description="API for interacting with the Forest OS personal growth assistant.",
)

# --- Store container on app.state ---
app.state.container = container
logger.info("DI Container instance stored in app.state.")

# --- Initialize Enhanced Architecture ---
logger.info("Initializing enhanced scalable architecture components...")
inject_enhanced_architecture(app)

# --- Initialize Discovery Journey Module ---
logger.info("Initializing Journey of Discovery module...")
setup_discovery_journey(app)

# --------------------------------------------------------------------------
# **** DI Container Wiring (MOVED HERE - BEFORE ROUTERS) ****
# --------------------------------------------------------------------------
try:
    logger.info("Wiring Dependency Injection container...")
    # Use the imported container instance directly
    container.wire(modules=[
        __name__, # Wire this main module if using @inject here
        "forest_app.routers.auth",
        "forest_app.routers.users",
        "forest_app.routers.onboarding",
        "forest_app.routers.hta",
        "forest_app.routers.snapshots",
        "forest_app.routers.core",
        "forest_app.routers.goals",
        "forest_app.core.orchestrator", # Add other modules using DI if needed
        "forest_app.helpers",
        # Enhanced architecture components
        "forest_app.core.services.enhanced_hta_service",
        "forest_app.core.initialize_architecture",
        "forest_app.core.task_queue",
        "forest_app.core.event_bus",
        "forest_app.core.cache_service",
        "forest_app.core.circuit_breaker"
        # Add other modules/packages containing @inject decorators or Provide markers
    ])
    logger.info("Dependency Injection container wired successfully.")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to wire DI Container: {e}", exc_info=True)
    raise RuntimeError(f"Failed to wire DI Container: {e}") from e


# --------------------------------------------------------------------------
# Include Routers
# --------------------------------------------------------------------------
logger.info("Including API routers...")
try:
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(users.router, prefix="/users", tags=["Users"])
    app.include_router(onboarding.router, prefix="/onboarding", tags=["Onboarding"])
    app.include_router(hta.router, prefix="/hta", tags=["HTA"])
    app.include_router(snapshots.router, prefix="/snapshots", tags=["Snapshots"])
    app.include_router(core.router, prefix="/core", tags=["Core"])
    app.include_router(goals.router, prefix="/goals", tags=["Goals"]) # Changed prefix /goal to /goals
    app.include_router(trees.router, prefix="/trees", tags=["Trees"]) # Added trees router for HTA tree management
    app.include_router(discovery_journey.router, tags=["Discovery Journey"])
    logger.info("API routers included successfully.")
except Exception as router_err:
     logger.critical(f"CRITICAL: Failed to include routers: {router_err}")
     sys.exit(f"CRITICAL: Router inclusion failed: {router_err}")


# --------------------------------------------------------------------------
# Middleware Configuration
# --------------------------------------------------------------------------
logger.info("Configuring middleware (CORS)...")
origins = [
    "http://localhost", "http://localhost:8000", "http://localhost:8501",
    # Ensure FRONTEND_URL env var is set in your deployment environment
    os.getenv("FRONTEND_URL", "*") # Use wildcard only if necessary and safe
]
# Remove potential duplicates and sort
origins = sorted(list(set(o for o in origins if o))) # Filter out empty strings

app.add_middleware(
    LoggingMiddleware
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the cleaned list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS middleware configured for origins: {origins}")


# --------------------------------------------------------------------------
# Startup Event (Feature Flag Logging Only) ### UPDATED COMMENT ###
# --------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event executing...")

    # --- Feature Flag Logging (Kept in startup event) ---
    logger.info("--- Verifying Feature Flag Status (from settings) ---")
    if feature_flags_available and hasattr(Feature, '__members__'):
        for feature in Feature:
            try:
                # Use the imported is_enabled function
                status = is_enabled(feature)
                logger.info(f"Feature: {feature.name:<35} Status: {'ENABLED' if status else 'DISABLED'}")
            except Exception as e:
                logger.error(f"Error checking status for feature {feature.name}: {e}")
    elif not feature_flags_available:
        logger.error("Feature flags module failed import, cannot check status.")
    else:
         logger.warning("Feature enum has no members defined?")
    logger.info("-----------------------------------------------------")
    # --- END Feature Flag Logging ---
    
    # --- Log architecture status ---
    if hasattr(app.state, 'architecture'):
        logger.info("Enhanced architecture is active and initialized")
        try:
            # Get and log metrics from components
            if hasattr(app.state.architecture, 'task_queue'):
                task_queue = app.state.architecture.task_queue()
                queue_status = await task_queue.get_queue_status()
                logger.info(f"Task Queue Status: {queue_status}")
                
            if hasattr(app.state.architecture, 'cache_service'):
                cache_service = app.state.architecture.cache_service()
                logger.info(f"Cache Service Active: {cache_service.config.backend.value}")
                
            if hasattr(app.state.architecture, 'event_bus'):
                event_bus = app.state.architecture.event_bus()
                bus_metrics = event_bus.get_metrics()
                logger.info(f"Event Bus Metrics: {bus_metrics}")
        except Exception as arch_err:
            logger.error(f"Error getting architecture component status: {arch_err}")
    else:
        logger.warning("Enhanced architecture not found in app.state")

    logger.info("Startup event complete.")


# --------------------------------------------------------------------------
# Shutdown Event
# --------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown event executing...")
    
    # --- Graceful shutdown of enhanced architecture components ---
    if hasattr(app.state, 'architecture'):
        logger.info("Shutting down enhanced architecture components...")
        try:
            # Stop task queue
            if hasattr(app.state.architecture, 'task_queue'):
                task_queue = app.state.architecture.task_queue()
                await task_queue.stop()
                logger.info("Task Queue stopped successfully")
        except Exception as shutdown_err:
            logger.error(f"Error during architecture component shutdown: {shutdown_err}")
    
    logger.info("Shutdown event complete.")


# --------------------------------------------------------------------------
# Root Endpoint
# --------------------------------------------------------------------------
@app.get("/", tags=["Status"], include_in_schema=False)
async def read_root():
    """ Basic status endpoint """
    return {"message": f"Welcome to the Forest OS API (Version {app.version})"}

# --------------------------------------------------------------------------
# Local Development Run Hook
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn development server directly via __main__...")
    reload_flag = os.getenv("APP_ENV", "development") == "development" and \
                 os.getenv("UVICORN_RELOAD", "True").lower() in ("true", "1")

    # Make sure to pass the app object correctly
    # Use "forest_app.main:app" if running from outside the directory
    # Use "main:app" if running from within the forest_app directory
    uvicorn.run(
        "main:app", # Changed for direct run
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload_flag,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info").lower(),
    )
