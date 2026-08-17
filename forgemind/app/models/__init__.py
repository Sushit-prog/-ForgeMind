"""SQLAlchemy models — the ForgeMind schema (architecture doc section G).

Implements the milestone-scoped tables:
  tasks, plans, plan_steps, task_steps, capabilities, policies, audit_logs,
  repositories, worktrees

Remaining Section-G tables (agent_runs, tool_calls, approvals, memories,
evaluations, research_artifacts, test_runs, failures, review_results,
security_results, pull_requests, ...) arrive with the phases that use them
(agents/tools/eval).
"""

from app.models.base import Base, JsonType
from app.models.audit import AuditLog
from app.models.capability import Capability
from app.models.execution_event import ExecutionEvent
from app.models.failure import FailureClassification
from app.models.implementation_summary import ImplementationSummary
from app.models.plan import Plan, PlanStep
from app.models.policy import Policy
from app.models.repository import Repository, Worktree
from app.models.research_artifact import ResearchArtifact
from app.models.review_result import ReviewResult
from app.models.security_result import SecurityResult
from app.models.task import Task, TaskStatus, TaskStep
from app.models.test_run import Failure, TestRun
from app.models.tool_call import ToolCall, ToolCallStatus

__all__ = [
    "AuditLog",
    "Base",
    "Capability",
    "ExecutionEvent",
    "Failure",
    "FailureClassification",
    "ImplementationSummary",
    "JsonType",
    "Plan",
    "PlanStep",
    "Policy",
    "Repository",
    "ResearchArtifact",
    "ReviewResult",
    "SecurityResult",
    "Task",
    "TaskStatus",
    "TaskStep",
    "TestRun",
    "ToolCall",
    "ToolCallStatus",
    "Worktree",
]
