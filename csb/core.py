import functools
import tensorflow as tf
import types


def graph_property(fn):
    return GraphProperty(fn)


class GraphProperty():
    '''A decorator for methods that define part of a TensorFlow graph.

    Graph properties are only executed once, and the result is cached so that
    future calls always returns the same value. And graph properties are
    property descriptors, meaning they are accessed as attributes, not methods.
    '''

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        try:
            return obj._graph_vals[self]
        except KeyError:
            obj._graph_vals[self] = self.fn(obj)
            return self.__get__(obj, type)
        except AttributeError:
            obj._graph_vals = {}
            return self.__get__(obj, type)


class Model:
    '''A base class for TensorFlow models.'''

    def build(self, graph):
        with graph.as_default():
            # Destroy any previously built graph_properties
            self._graph_vals = {}

            # Simply touching a GraphProperty is enough to build it.
            # So by collecting them, we are rebuilding them.
            ops = {}
            for name in dir(type(self)):
                attr = getattr(type(self), name)
                if isinstance(attr, GraphProperty):
                    ops[name] = getattr(self, name)
        return ops
