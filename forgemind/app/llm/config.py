"""Model selection per role (architecture doc section 34).

Model names live in the ENVIRONMENT (``LLM_MODEL_PLANNER`` etc.), never in
code — swapping models is a config change, not a code change. Unknown
roles return None (the provider then uses its own default, which for
OpenRouter means the request MUST name a model — callers fail loudly if
one is missing, they never silently default to a hardcoded name).

Each role can also carry an optional FALLBACK CHAIN
(``LLM_MODEL_<ROLE>_FALLBACKS``, comma-separated, ordered): models tried
in order when the primary keeps returning transient errors (free-tier 429
rate limits above all). Empty/unset means no fallback — single-model
behavior unchanged.
"""

from __future__ import annotations

from app.config import get_settings

# role -> settings field that carries the model (env: LLM_MODEL_<ROLE>).
# Keys MUST match the ``role`` strings callers pass to build_provider
# ("research", not "researcher" — see build_researcher).
_ROLE_MODEL_FIELDS = {
    "planner": "llm_model_planner",
    "research": "llm_model_research",
    "developer": "llm_model_developer",
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


def get_fallback_models_for_role(role: str) -> list[str]:
    """The configured FALLBACK chain for ``role`` (ordered), or [] if none.

    Read from env as a comma-separated list (``LLM_MODEL_<ROLE>_FALLBACKS``):
    whitespace is stripped and empty entries dropped, so
    ``"a/b:free, c/d:free ,, "`` is a clean two-model chain.
    """
    field = _ROLE_MODEL_FIELDS.get(role)
    if field is None:
        return []
    raw = getattr(get_settings(), f"{field}_fallbacks", None)
    if not raw:
        return []
    return [model.strip() for model in raw.split(",") if model.strip()]


def known_roles() -> list[str]:
    return list(_ROLE_MODEL_FIELDS)
