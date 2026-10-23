"""Request-local provider credentials; never persisted or shared across users."""
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import SecretStr

_request_key: ContextVar[SecretStr | None] = ContextVar('request_model_api_key',default=None)


def current_model_api_key():
    return _request_key.get()


@contextmanager
def use_model_api_key(key: SecretStr | None):
    token=_request_key.set(key)
    try:
        yield
    finally:
        _request_key.reset(token)
