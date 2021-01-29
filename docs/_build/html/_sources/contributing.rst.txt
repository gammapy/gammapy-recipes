.. _contributing:

Contributing
------------

Please follow these rules below before making a pull request in
the Gammapy recipes `Github repository <https://github.com/Bultako/tutorials>`_.

- copy any of the existing folders in ``recipes`` and rename it as your own recipe
- modify the ``env.yml`` file and specify your needed dependencies
- remove the rest of the files other than ``env.yml``
- add your notebook, used ``.png`` images and any needed data not present in ``GAMMAPY_DATA``
- paste ``"tags": ["nbsphinx-thumbnail"]`` in the **code cell** metadata creating the **thumbnail**
- make a pull request with your additions
- wait for content integration pass

--------

**Maintainers**

This repository is designed to be maintained in a minimal effort basis. Most changes should come
from additions of recipes made as pull requests contributions, and passing CI tests deployed as
Github actions. Maintainers should just accept and merge pull requests.

**Specific commands**

- Rebuilding of the whole collection of recipes: ``python scripts/process.py``
- Rebuilding of a specific recipe: ``python scripts/process.py --src recipes/folder``
- Building of Sphinx documentation: ``cd docs && python -m sphinx . _build/html -b html && cd ..``

**Renaming or removing existing recipes**

In special cases when renaming or removing of a recipe is needed,
please remove the old recipe folders and their content in the following places below.

- ``docs/_static/notebooks``
- ``docs/notebooks``
- ``recipes``