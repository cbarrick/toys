from contextlib import contextmanager


class Context:
    def __init__(self, stack=None):
        if stack is None:
            stack = current_context().stack
        self.stack = stack

    def __copy__(self):
        return self.copy()

    def copy(self):
        stack = self.stack.copy()
        return Context(stack)

    def push(self, *args, **kwargs):
        frame = self.stack[-1].copy()
        frame = frame.update(*args, **kwargs)
        self.stack.append(frame)
        return self

    def pop(self):
        return self.stack.pop()

    def get(self, key, default=None):
        frame = self.stack[-1]
        return frame.get(key, default)


_INITIAL_CONTEXT = Context([])
_CURRENT_CONTEXT = _INITIAL_CONTEXT


def current_context():
    global _CURRENT_CONTEXT
    return _CURRENT_CONTEXT


@contextmanager
def context(*args, **kwargs):
    global _CURRENT_CONTEXT
    old = current_context()
    new = old.copy()
    new.push(*args, **kwargs)
    _CURRENT_CONTEXT = new
    yield new
    _CURRENT_CONTEXT = old
