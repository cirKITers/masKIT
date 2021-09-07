=====================
Versions and Releases
=====================

The version numbers of masKIT follow the `Semantic Versioning Specification`_
in the MAJOR.MINOR.PATCH format.
New versions are published via `PyPI`_ for installation via ``pip``.

Release Workflow
================

.. note::

    This section is only relevant for maintainers of ``masKIT``.

Releases are performed manually and should happen at least when
an important fix or major feature is added.
Most releases will bump the *minor* version number;
the *patch* version number is mostly bumped for fixes and the *major* version
number to indicate breaking changes.

1. Review all changes added by the new release
    * Naming of functions/classes/parameters
    * Docs are up to date and consistent
    * Unittests cover all obvious cases

2. Bump the version number
    * Adjust and commit ``maskit.__init__.__version__``
    * Create a git tag such as ``git tag -a "v0.1.0" -m "description"``
    * Push the commit and tag to github

3. Publish to PyPI
    * **You need maintainer access** on the `PyPI maskit project`_
    * Check out the tagged version commit
    * Run ``flit publish``

.. _`Semantic Versioning Specification`: https://semver.org
.. _PyPI: https://pypi.org
.. _`PyPI maskit project`: https://pypi.org/project/maskit/
