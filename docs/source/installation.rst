Installation
=============

SCOPE-XR is installed as a standard Python package. It is recommended to
use a virtual environment (venv or conda) to keep your system clean.

Prerequisites
------------

-  Python 3.9 or higher
-  Git

1. Quick Install (For regular users)
------------

If you just want to use the software without modifying the code, install
directly from the source via pip:

.. code:: bash

   pip install git+https://github.com/jacopoaltieri/scope-xr.git

⚠️ **Important**: Configuration files (``.yaml``) are required for
execution. Please download them from the ``examples`` folder and place
them in your working directory.

2. Manual Install (From source)
------------

Recommended if you wish to modify the code or contribute:

1. **Clone the repository:**

   .. code:: bash

      git clone https://github.com/jacopoaltieri/scope-xr
      cd scope-xr

2. **Create and activate a virtual environment (Optional but
   Recommended):**

   **Windows:**

   .. code:: bash

      python -m venv venv
      .\venv\Scripts\activate

   **Linux/macOS:**

   .. code:: bash

      python3 -m venv venv
      source venv/bin/activate

3. **Install the package:**

   Install in editable mode (recommended for development) to ensure all
   dependencies are handled automatically:

   .. code:: bash

      pip install -e .