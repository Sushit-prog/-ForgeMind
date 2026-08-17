"""implementation_summaries table

Revision ID: b7f3a91e2c45
Revises: d64e9ba2c994
Create Date: 2026-08-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7f3a91e2c45'
down_revision: Union[str, None] = 'd64e9ba2c994'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('implementation_summaries',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('task_id', sa.Uuid(), nullable=False),
    sa.Column('step_id', sa.Uuid(), nullable=True),
    sa.Column('worktree_id', sa.Uuid(), nullable=True),
    sa.Column('commit_sha', sa.String(length=64), nullable=True),
    sa.Column('files_changed', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('tests_added', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
    sa.Column('deviations_from_research', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['step_id'], ['plan_steps.id'], name=op.f('fk_implementation_summaries_step_id_plan_steps'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], name=op.f('fk_implementation_summaries_task_id_tasks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['worktree_id'], ['worktrees.id'], name=op.f('fk_implementation_summaries_worktree_id_worktrees'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_implementation_summaries'))
    )
    op.create_index(op.f('ix_implementation_summaries_task_id'), 'implementation_summaries', ['task_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_implementation_summaries_task_id'), table_name='implementation_summaries')
    op.drop_table('implementation_summaries')
