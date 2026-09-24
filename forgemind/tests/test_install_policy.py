"""Install-command policy tests (Phase 13, security posture of command_policy).

The core guarantee: ``repositories.install_command`` is a SERVER-SIDE value —
detected + validated at discovery, re-validated at invocation — and
``shell.install_deps`` takes no command input at all. Verify the allowlist,
the metachar/path checks (with the deliberate bracket allowance for pip's
extras syntax), and the token list the provisioner is allowed to run.
"""

from __future__ import annotations

import pytest

from app.shell.install_policy import (
    InstallCommandError,
    validate_install_command,
)


def test_accepts_pip_extras_command() -> None:
    """The Phase-13 default detected command is legal and tokenizes cleanly
    (brackets in ``.[dev]`` are pip extras syntax, inert in arg-list execution)."""
    assert validate_install_command('pip install -e ".[dev]"') == [
        "pip",
        "install",
        "-e",
        ".[dev]",
    ]
    assert validate_install_command("pip install .") == ["pip", "install", "."]


def test_rejects_non_pip_binary() -> None:
    with pytest.raises(InstallCommandError):
        validate_install_command("npm install")
    with pytest.raises(InstallCommandError):
        validate_install_command("poetry install")


def test_rejects_command_composition_metachars() -> None:
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install a; rm -rf /")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install a && craft")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install a | bash")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install $(id)")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install a > /tmp/pwn")


def test_rejects_escaping_or_absolute_paths() -> None:
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install ../evil")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install /etc/passwd")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install C:\\Users\\pwn")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install ~/.ssh/id_rsa")


def test_rejects_empty_and_unparseable() -> None:
    with pytest.raises(InstallCommandError):
        validate_install_command("")
    with pytest.raises(InstallCommandError):
        validate_install_command("   ")
    with pytest.raises(InstallCommandError):
        validate_install_command("pip install 'unterminated")


def test_error_type_is_distinct() -> None:
    """InstallCommandError is its own value error, separate from the test
    command policy's type — callers can treat an adversarial install command
    as its own failure class, never a mis-typed test-command error."""
    assert issubclass(InstallCommandError, ValueError)
    from app.shell.command_policy import TestCommandError

    assert InstallCommandError is not TestCommandError