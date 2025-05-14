# Import key persistence components to make them available at package level
# This helps avoid relative import issues when running tests
from forest_app.persistence.database import engine, SessionLocal, get_db
from forest_app.persistence.models import *
