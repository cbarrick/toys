import functools
import tensorflow as tf
import types


def graph_property(fn):
    '''A decorator for properties that define part of a TensorFlow graph.

    Graph properties are executed once when they are first accessed, then the
    instance attribute for the property is replaced by the returned value. This
    allows you to define TensorFlow computations as class properties and
    ensures that the computations are only added to the graph once for each
    instance.
    '''
    return GraphProperty(fn)


class GraphProperty:
    def __init__(self, body):
        self.body = body

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        val = self.body(obj)
        setattr(obj, self.body.__name__, val)
        return val


class MetaModel(type):
    def __init__(cls, name, bases, namespace):
        old_init = getattr(cls, '__init__', lambda self: self)

        def new_init(self, *args, **kwargs):
            # Create a graph first, so __init__ has access to it.
            # It can be overridden in __init__ if it must.
            self.graph = tf.Graph()

            # Sessions are added here when created with `Model.new_session()`.
            self._sessions = set()

            # Call the original __init__ first so that
            # graph properties have access to anything defined there.
            ret = old_init(self, *args, **kwargs)

            # Ensure all graph properties are built.
            with self.graph.as_default():
                for name, attr in namespace.items():
                    if isinstance(attr, GraphProperty):
                        getattr(self, name)

            # Initizlize any sessions created in __init__.
            with self.graph.as_default():
                self._global_variables_initializer = tf.global_variables_initializer()
            for sess in self._sessions:
                sess.run(self._global_variables_initializer)

            # Close the graph for added safety.
            # Maybe TF can optimize this case?
            self.graph.finalize()

            # Returning anything other than None from __init__ is a TypeError,
            # BUT we should be consistent with whatever the user returned.
            return ret

        setattr(cls, '__init__', new_init)


class Model(metaclass=MetaModel):
    '''A base class for TensorFlow models.'''

    def new_session(self, *args, **kwargs):
        '''Returns a new session tied to this model's graph.'''
        sess = tf.Session(graph=self.graph)
        self._initialize_session(sess)
        return sess

    def _initialize_session(self, sess):
        # Only initialize sessions after the graph is finalized.
        # Sessions created before then are initialized after __init__.
        self._sessions.add(sess)
        if self.graph.finalized:
            sess.run(self._global_variables_initializer)
