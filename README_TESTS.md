Running tests
=============

Quick steps to run the test suite locally:

1. Create a virtual environment and activate it (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run pytest from the repository root:

```bash
pytest -q
```

Notes:
- The unit tests import `NNNC` from the repository root; run pytest from the repo root so imports resolve.
Running tests
=============

Quick steps to run the test suite locally:

1. Create a virtual environment and activate it (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run pytest from the repository root:

```bash
pytest -q
```

Notes:
- The unit tests reference the `reference_impl` directory; pytest is configured to run from the repo root so the tests insert `reference_impl` onto sys.path.
