from collections.abc import MutableMapping


class context(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._items = dict(*args, **kwargs)
        self._stack = []

    def __enter__(self):
        ctx = current_context()
        self._stack.append(ctx)
        set_current_context(self)
        return self

    def __exit__(self, *args):
        assert current_context() is self
        ctx = self._stack.pop()
        set_current_context(ctx)
        return False

    def __getitem__(self, key):
        try:
            return self._items.__getitem__(key)
        except KeyError:
            prev = self._stack[-1]
            return prev[key]

    def __setitem__(self, key, val):
        self._items[key] = val

    def __delitem__(self, key):
        del self._items[key]

    def __iter__(self, memo=None):
        # We must handle an infinite loop in the reentrant case.
        key = id(self)
        if memo is None: memo = {}
        if key not in memo: memo[key] = 0
        memo[key] += 1
        i = memo[key]

        yield from self._items
        if i <= len(self._stack):
            prev = self._stack[-i].__iter__(memo)
            yield from (x for x in prev if x not in self._items)

    def __len__(self):
        return len(tuple(iter(self)))

    def __repr__(self):
        return 'context(' + str(dict(self)) + ')'

    def __copy__(self):
        from copy import copy
        cpy = context()
        cpy._items = copy(self._items)
        cpy._stack = copy(self._stack)
        return cpy

    def __deepcopy__(self, memo):
        from copy import deepcopy
        cpy = context()
        cpy._items = deepcopy(self._items, memo)
        cpy._stack = deepcopy(self._stack, memo)
        return cpy

    def copy(self):
        return self.__copy__()


_CURRENT_CONTEXT = context()


def current_context():
    global _CURRENT_CONTEXT
    return _CURRENT_CONTEXT


def set_current_context(ctx):
    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = ctx
