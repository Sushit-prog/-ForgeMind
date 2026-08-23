"""task enqueue_attempts for the stale-CREATED sweep

Revision ID: c4a91f7b2d30
Revises: a8c5d1f42b67
Create Date: 2026-08-23 22:30:11.402917

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4a91f7b2d30"
down_revision: Union[str, None] = "a8c5d1f42b67"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("enqueue_attempts", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("tasks", "enqueue_attempts")
