{{ fullname }}
==================================================

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


Member Index
--------------------------------------------------
.. rubric:: Methods
.. autosummary::

	{% for meth in methods %}
	~{{ objname }}.{{ meth }}
	{% endfor %}


.. rubric:: Attributes
.. autosummary::

	{% for attr in attributes %}
	~{{ objname }}.{{ attr }}
	{% endfor %}
