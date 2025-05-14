"""Update UUID fields to use as_uuid=True

Revision ID: update_uuid_fields
Revises: f5b76ed1b9bd
Create Date: 2025-05-11 09:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'update_uuid_fields'
down_revision: Union[str, None] = 'f5b76ed1b9bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create a GIN index on manifest field in hta_trees
    op.create_index('idx_hta_trees_manifest_gin', 'hta_trees', ['manifest'], postgresql_using='gin')


def downgrade() -> None:
    # Drop the GIN index on manifest field in hta_trees
    op.drop_index('idx_hta_trees_manifest_gin', table_name='hta_trees')
