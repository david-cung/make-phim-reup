# Local Movie Translator — Python worker

Long-running child process spawned by the Rust host. Communicates with the
host over `stdin`/`stdout` using line-delimited JSON-RPC 2.0. Structured
logs are written to `stderr` — the Rust supervisor captures them and
merges them into the app's log file.

## Phase 1 status

Only the transport, dispatcher, logging, error contract and four control
methods (`initialize`, `ping`, `env_info`, `shutdown`) are implemented.

AI providers (`faster-whisper`, `llama.cpp`, Piper) will be added in
Phases 3–6 under `movie_translator_worker.providers.*` and behind the
provider interfaces already sketched in `ARCHITECTURE.md`.

## Layout

```
src/movie_translator_worker/
  __init__.py
  __main__.py         # entry-point: `python -m movie_translator_worker`
  rpc.py              # framing + dispatch + error contract
  handlers.py         # Phase 1 methods
  logging.py          # JSON logger to stderr
  errors.py           # RpcError code constants
tests/
  test_rpc.py
```

## Running standalone

```bash
python -m movie_translator_worker
# then type JSON-RPC requests on stdin, e.g.
# {"jsonrpc":"2.0","id":"1","method":"ping","params":{}}
```

## Tests

```bash
pip install -e .[dev]
pytest -q
```
