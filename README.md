# pybingraph
pybingraph

Binary inspection API + UI using **angr**. Provides:

- `GET /symtab?filepath=...` to render a symbol table
- `GET /cfg?filepath=...&function=0x401000` to render a function CFG as SVG
- A Streamlit UI (`streamlit_app.py`) that uses the same core library

## Requirements

- Python **3.12** (managed by `uv`)
- Graphviz system binary (`dot`) for SVG rendering:

```bash
brew install graphviz
```

## Setup

```bash
uv sync
```

### Fetch playground binaries

```bash
./scripts/fetch_playground.sh
```

## Run the API

```bash
uv run bingraph-api --root ./playground/angr-binaries
```

Then open:

- `http://127.0.0.1:8000/symtab?filepath=<relative/path/to/binary>`

Example (once playground exists):

```
http://127.0.0.1:8000/symtab?filepath=tests/x86_64/fauxware
```

## Tests

```bash
uv run pytest
```

Slow integration tests are marked with `@pytest.mark.slow` and will be skipped if playground binaries are not present.

## Troubleshooting

If you see:

```
angr.state_plugins.unicorn_engine | failed loading "unicornlib.dylib", unicorn support disabled
```

This is a common optional-dependency warning on macOS. CFGFast and the analysis used here still work without Unicorn. If you want Unicorn-enabled execution, install the `unicorn` package (and ensure it has compatible wheels for your Python version) or use `angr[full]` and rebuild the environment.
