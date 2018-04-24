from contextlib import contextmanager


_CURRENT_CONTEXT = {}


def current_context():
    global _CURRENT_CONTEXT
    return _CURRENT_CONTEXT


@contextmanager
def context(*args, **kwargs):
    global _CURRENT_CONTEXT
    old = current_context()
    new = old.copy()
    new.update(*args, **kwargs)
    _CURRENT_CONTEXT = new
    yield new
    _CURRENT_CONTEXT = old
