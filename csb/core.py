import functools
import tensorflow as tf
import types


def graph_property(_fn=None, **kwargs):
    '''A decorator for properties that define part of a TensorFlow graph.

    Can be used in the following forms:
    - `@graph_property`
    - `@graph_property()`
    - `@graph_property(**kwargs)`

    Graph properties are executed once when they are first accessed, then the
    instance attribute for the property is replaced by the returned value. This
    allows you to define TensorFlow computations as class properties and
    ensures that the computations are only added to the graph once for each
    instance.

    Graph properties are called with a single argument, the `tf.VariableScope`
    in which the property is being constructed. By default, the name of the
    scope is the name of the property. When the decorator is called with
    keyword arguments, the arguments are passed to the variable scope.
    '''
    def decorator(fn):
        return GraphProperty(fn, **kwargs)
    if _fn:
        # This was an actual decorator call, ex: @graph_property
        return decorator(_fn)
    else:
        # This is a factory call, ex: @graph_property(...)
        return decorator


class GraphProperty:
    '''A property descriptor for methods that define part of a TensorFlow graph.

    See `graph_property` for more documentation.
    '''

    def __init__(self, body, **kwargs):
        self.body = body
        self.scope_args = kwargs

    def __get__(self, obj, type=None):
        if obj is None:
            return self

        scope_name = self.scope_args.pop('name', None)
        property_name = self.body.__name__
        with tf.variable_scope(scope_name, default_name=property_name, **self.scope_args) as scope:
            val = self.body(obj, scope)

        setattr(obj, property_name, val)
        return val


class _MetaModel(type):
    '''The metaclass for the `Model` base class.'''

    def __init__(cls, name, bases, namespace):
        old_init = getattr(cls, '__init__', lambda self: self)

        def new_init(self, *args, **kwargs):
            # Sessions are added here when created with `Model.new_session()`.
            self._sessions = set()

            with tf.Graph().as_default() as graph, tf.variable_scope(name) as scope:
                self.graph = graph
                self.scope = scope

                # Call the original __init__ first so that
                # graph properties have access to anything defined there.
                ret = old_init(self, *args, **kwargs)

                # Ensure all graph properties are built.
                for attr_name, attr in namespace.items():
                    if isinstance(attr, GraphProperty):
                        getattr(self, attr_name)

                # Initizlize any sessions created in __init__.
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


class Model(metaclass=_MetaModel):
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
