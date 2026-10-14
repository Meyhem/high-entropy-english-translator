import pytest

from heet.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config()
