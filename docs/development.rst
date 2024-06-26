Development
===========

The ``metalearners`` repository can be cloned as follows

.. code-block:: console

  git clone https://github.com/Quantco/metalearners.git

The dependencies are managed via
`pixi <https://pixi.sh/latest/>`_. Please make sure to install ``pixi`` on
your system. Once pixi is installed, you can install and run the
pre-commit hooks as follows:


.. code-block:: console

  pixi run pre-commit-install
  pixi run pre-commit-run


You can run the tests as follows:

.. code-block:: console

  pixi run postinstall
  pixi run pytest tests

You can build the documentation locally by running

.. code-block:: console

  pixi run -e docs postinstall
  pixi run -e docs docs

You can then inspect the locally built docs by opening ``docs/_build/index.html``.

You can find all pixi tasks in ``pixi.toml``.
