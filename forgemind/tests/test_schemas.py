"""Unit tests for request-schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas import TaskCreate

VALID_URLS = [
    "https://github.com/org/repo.git",
    "https://github.com/org/repo",
    "http://gitlab.example.com/group/project.git",
    "git@github.com:org/repo.git",
    "ssh://git@github.com/org/repo.git",
    "git://github.com/org/repo.git",
]

INVALID_URLS = [
    "not-a-url",
    "https://",  # no host or path
    "https://github.com",  # no repo path
    "ftp://github.com/org/repo.git",  # unsupported scheme
    "",  # empty
    "   ",  # blank after strip
]


@pytest.mark.parametrize("url", VALID_URLS)
def test_valid_repository_urls_accepted(url: str) -> None:
    task = TaskCreate(objective="Fix a bug", repository_url=url)
    assert task.repository_url == url.strip()


@pytest.mark.parametrize("url", INVALID_URLS)
def test_invalid_repository_urls_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        TaskCreate(objective="Fix a bug", repository_url=url)


def test_empty_objective_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskCreate(objective="", repository_url="https://github.com/org/repo.git")


def test_missing_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskCreate(objective="Fix a bug")  # missing repository_url
    with pytest.raises(ValidationError):
        TaskCreate(repository_url="https://github.com/org/repo.git")  # missing objective
