Contributing Guidelines
=======================

We welcome contributions to SCOPE-XR! This page provides a quick overview of how to contribute. For detailed guidelines, see the full `CONTRIBUTING.md <https://github.com/jacopoaltieri/scope-xr/blob/main/CONTRIBUTING.md>`_ file.

Ways to Contribute
------------------

**Code Contributions:**

* Bug fixes
* New features
* Performance improvements
* Test coverage improvements

**Documentation:**

* Fixing typos or clarifying text
* Adding examples
* Writing tutorials
* Translating documentation

**Community:**

* Reporting bugs
* Suggesting features
* Answering questions
* Reviewing pull requests

Quick Start for Contributors
-----------------------------

1. **Fork the repository** on GitHub

2. **Set up development environment:**

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/scope-xr.git
       cd scope-xr
       python -m venv venv
       source venv/bin/activate  # On Windows: .\venv\Scripts\activate
       pip install -e .[dev,test,docs]

3. **Create a feature branch:**

   .. code-block:: bash

       git checkout -b feature/your-feature-name

4. **Make your changes** and commit:

   .. code-block:: bash

       git add .
       git commit -m "Add: brief description of changes"

5. **Run tests and linting:**

   .. code-block:: bash

       ruff check
       pytest

6. **Push and create Pull Request:**

   .. code-block:: bash

       git push origin feature/your-feature-name

Code Style
----------

SCOPE-XR uses **ruff** for code linting and formatting:

.. code-block:: bash

    # Check for issues
    ruff check
    
    # Auto-fix where possible
    ruff check --fix

**Style Guidelines:**

* Follow PEP 8
* Use NumPy-style docstrings
* Write descriptive variable names
* Keep functions focused and concise

Testing
-------

All contributions should include tests:

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run with coverage
    pytest --cov=scopexr --cov-report=html
    
    # Run specific test
    pytest tests/test_circle_detection.py

Documentation
-------------

Update documentation for any changes:

.. code-block:: bash

    cd docs
    make html
    # Open docs/build/html/index.html

Pull Request Guidelines
-----------------------

**Before submitting:**

* [ ] Tests pass
* [ ] Code follows style guide (ruff passes)
* [ ] Documentation updated
* [ ] Commit messages are clear

**PR Title Format:**

.. code-block:: text

    [Type] Brief description
    
    Examples:
    [Feature] Add sub-pixel oversampling
    [Bugfix] Fix circle detection for low contrast
    [Docs] Add troubleshooting guide

Reporting Issues
----------------

When reporting bugs, please include:

* SCOPE-XR version
* Python version
* Operating system
* Steps to reproduce
* Expected vs. actual behavior
* Error messages/traceback

Feature Requests
----------------

For feature suggestions, describe:

* The proposed feature
* Use case / motivation
* Expected behavior
* Relevant scientific references (if applicable)

Questions?
----------

* Check the documentation first
* Search existing issues
* Open a new issue for questions
* Join discussions on GitHub

Code of Conduct
---------------

Be respectful, welcoming, and constructive. We aim to foster an inclusive community.

License
-------

By contributing, you agree that your contributions will be licensed under GPL-3.0.

---

For complete guidelines, see `CONTRIBUTING.md <https://github.com/jacopoaltieri/scope-xr/blob/main/CONTRIBUTING.md>`_.
