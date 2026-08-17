"""test_runs, failures, and failure_classifications tables

Revision ID: 9e3f2c81d4a7
Revises: b7f3a91e2c45
Create Date: 2026-08-17 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '9e3f2c81d4a7'
down_revision: Union[str, None] = 'b7f3a91e2c45'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('test_runs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('task_id', sa.Uuid(), nullable=False),
    sa.Column('worktree_id', sa.Uuid(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('passed', sa.Integer(), nullable=False),
    sa.Column('failed', sa.Integer(), nullable=False),
    sa.Column('duration_ms', sa.Integer(), nullable=False),
    sa.Column('exit_code', sa.Integer(), nullable=True),
    sa.Column('timed_out', sa.Boolean(), nullable=False),
    sa.Column('output', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], name=op.f('fk_test_runs_task_id_tasks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['worktree_id'], ['worktrees.id'], name=op.f('fk_test_runs_worktree_id_worktrees'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_test_runs'))
    )
    op.create_index(op.f('ix_test_runs_task_id'), 'test_runs', ['task_id'], unique=False)

    op.create_table('failures',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('test_run_id', sa.Uuid(), nullable=False),
    sa.Column('test', sa.String(length=512), nullable=False),
    sa.Column('output', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['test_run_id'], ['test_runs.id'], name=op.f('fk_failures_test_run_id_test_runs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_failures'))
    )

    op.create_table('failure_classifications',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('task_id', sa.Uuid(), nullable=False),
    sa.Column('test_run_id', sa.Uuid(), nullable=True),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('root_cause', sa.Text(), nullable=False),
    sa.Column('fix_instruction', sa.Text(), nullable=True),
    sa.Column('fixable', sa.Boolean(), nullable=False),
    sa.Column('is_flaky', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], name=op.f('fk_failure_classifications_task_id_tasks'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['test_run_id'], ['test_runs.id'], name=op.f('fk_failure_classifications_test_run_id_test_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_failure_classifications'))
    )
    op.create_index(op.f('ix_failure_classifications_task_id'), 'failure_classifications', ['task_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_failure_classifications_task_id'), table_name='failure_classifications')
    op.drop_table('failure_classifications')
    op.drop_table('failures')
    op.drop_index(op.f('ix_test_runs_task_id'), table_name='test_runs')
    op.drop_table('test_runs')
