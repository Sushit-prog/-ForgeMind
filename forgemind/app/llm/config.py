"""Model selection per role (architecture doc section 34).

Model names live in the ENVIRONMENT (``LLM_MODEL_PLANNER`` etc.), never in
code — swapping models is a config change, not a code change. Unknown
roles return None (the provider then uses its own default, which for
OpenRouter means the request MUST name a model — callers fail loudly if
one is missing, they never silently default to a hardcoded name).
"""

from __future__ import annotations

from app.config import get_settings

# role -> settings field that carries the model (env: LLM_MODEL_<ROLE>).
_ROLE_MODEL_FIELDS = {
    "planner": "llm_model_planner",
    "debugger": "llm_model_debugger",
    "reviewer": "llm_model_reviewer",
    "security": "llm_model_security",
}


def get_model_for_role(role: str) -> str | None:
    """The configured model for ``role``, or None if unset."""
    field = _ROLE_MODEL_FIELDS.get(role)
    if field is None:
        return None
    return getattr(get_settings(), field, None)


def known_roles() -> list[str]:
    return list(_ROLE_MODEL_FIELDS)
