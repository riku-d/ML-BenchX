"""Stage 4 experiment history

Adds ``experiment_records`` — the corpus Stage 4's meta-learner ranks against. One row
per past experiment: what the dataset looked like as a meta-feature vector, and which
algorithm family actually won on it.

**Purely additive.** One new table and its indexes. No existing table, column, index or
row is altered or removed, so an existing database upgrades with every dataset,
workspace, plan and run intact.

Two things in here are deliberate rather than generated.

**The timestamp default is ``(CURRENT_TIMESTAMP)``, not ``now()``.** Autogenerate
rendered ``now()`` because it was run against PostgreSQL, and SQLite has no such
function — the migration would have applied on one engine and failed on the other.
``CURRENT_TIMESTAMP`` is standard SQL and is what the three earlier migrations use, so
this one stays portable and consistent with them.

**``holdout`` is indexed.** Every similarity search filters on it, because NFR-018's
accuracy KPI cannot be measured on the records that produced the ranking.

Revision ID: ca103e7ef01b
Revises: dffe8ca073fe
Create Date: 2026-11-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca103e7ef01b'
down_revision: Union[str, Sequence[str], None] = 'dffe8ca073fe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'experiment_records',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('source', sa.String(length=16), nullable=False),
        sa.Column('source_ref', sa.String(length=64), nullable=False),
        sa.Column('dataset_name', sa.String(length=255), nullable=False),
        sa.Column('target', sa.String(length=255), nullable=False),
        sa.Column('task', sa.String(length=32), nullable=False),
        sa.Column('metafeature_version', sa.Integer(), nullable=False),
        sa.Column('n_rows', sa.Integer(), nullable=False),
        sa.Column('n_cols', sa.Integer(), nullable=False),
        sa.Column('n_numerical', sa.Integer(), nullable=False),
        sa.Column('n_categorical', sa.Integer(), nullable=False),
        sa.Column('sparsity', sa.Float(), nullable=False),
        sa.Column('imbalance_ratio', sa.Float(), nullable=False),
        sa.Column('mean_abs_correlation', sa.Float(), nullable=False),
        sa.Column('n_classes', sa.Integer(), nullable=False),
        sa.Column('meta_features', sa.JSON(), nullable=False),
        sa.Column('best_family', sa.String(length=64), nullable=False),
        sa.Column('family_scores', sa.JSON(), nullable=False),
        sa.Column('metric', sa.String(length=32), nullable=False),
        sa.Column('folds', sa.Integer(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False),
        sa.Column('training_ms', sa.Integer(), nullable=False),
        sa.Column('holdout', sa.Boolean(), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source', 'source_ref', name='uq_experiment_source_ref'),
    )
    with op.batch_alter_table('experiment_records', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_experiment_records_best_family'), ['best_family'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_experiment_records_holdout'), ['holdout'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_experiment_records_metafeature_version'),
            ['metafeature_version'],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f('ix_experiment_records_project_id'), ['project_id'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_experiment_records_source_ref'), ['source_ref'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_experiment_records_task'), ['task'], unique=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('experiment_records', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_experiment_records_task'))
        batch_op.drop_index(batch_op.f('ix_experiment_records_source_ref'))
        batch_op.drop_index(batch_op.f('ix_experiment_records_project_id'))
        batch_op.drop_index(batch_op.f('ix_experiment_records_metafeature_version'))
        batch_op.drop_index(batch_op.f('ix_experiment_records_holdout'))
        batch_op.drop_index(batch_op.f('ix_experiment_records_best_family'))

    op.drop_table('experiment_records')
