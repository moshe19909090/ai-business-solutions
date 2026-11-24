# AI Business Solutions

A portfolio of practical, independently deployable AI solutions for US small businesses
and the firms that serve them.

## Solutions

| Solution | Status | Description |
|---|---|---|
| [Bookkeeping Exception Assistant](solutions/bookkeeping-assistant/) | In development (MVP) | Flags bank transactions that need a bookkeeper's attention: missing receipts, amount mismatches, possible duplicates, ambiguous matches, incomplete documents. |

## Conventions

- **Self-contained solutions.** Each solution lives in `solutions/<name>/` with its own
  `pyproject.toml`, dependencies, tests, and README. There is no shared library yet. Code is
  copied first, and extracted into a shared package only after repeated, proven reuse.
- **LLM extracts, code decides.** LLMs are used only where they add value (for example, reading
  unstructured documents). Financial calculations and business rules are deterministic,
  typed, and tested. Money uses `Decimal`, never `float`.
- **Human in the loop.** Solutions produce findings for human review. They never modify
  accounting records or act as an accounting, audit, tax, or legal authority.
- **Synthetic data only.** No real client data is ever committed to this repository.
- **Tooling (Python):** `uv`, Python 3.12+, Ruff, `mypy --strict`, pytest.
