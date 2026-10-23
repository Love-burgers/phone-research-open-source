"""A turn deadline shared by provider and source adapters without global state."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_deadline = ContextVar('chat_turn_deadline', default=None)


class DeadlineExceeded(TimeoutError):
    pass


def remaining_seconds():
    end = _deadline.get()
    if end is None:
        return None
    remaining = end-monotonic()
    if remaining <= 0:
        raise DeadlineExceeded('turn_deadline')
    return remaining


def request_timeout(configured):
    remaining = remaining_seconds()
    return min(configured, remaining) if remaining is not None else configured


@contextmanager
def turn_deadline(seconds):
    end = monotonic()+seconds
    previous = _deadline.get()
    token = _deadline.set(min(end, previous) if previous is not None else end)
    try:
        yield
    finally:
        _deadline.reset(token)
