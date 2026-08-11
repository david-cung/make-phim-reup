"""Worker entrypoint: `python -m movie_translator_worker`."""

from __future__ import annotations

import os
import signal
import sys

from . import handlers, logging as log
from .rpc import Dispatcher, serve


def _install_signal_handlers() -> None:
    def _handle(signum: int, _frame) -> None:  # noqa: ANN001 — signal signature
        log.info("received signal, exiting", signal=signum)
        # A clean exit closes stdout; the Rust host treats this as the
        # worker having shut down and either restarts it (crash) or drops
        # the supervisor loop (if shutdown was requested).
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            # Not available on Windows for some signals; ignore.
            pass


def main() -> int:
    _install_signal_handlers()
    log.info(
        "worker starting",
        pid=os.getpid(),
        python=sys.version.split()[0],
        data_root=os.environ.get("LMT_DATA_DIR"),
        app_version=os.environ.get("LMT_APP_VERSION"),
    )
    dispatcher = Dispatcher()
    handlers.install(dispatcher)
    try:
        serve(dispatcher)
    except KeyboardInterrupt:
        return 0
    except Exception as e:  # pragma: no cover - safety net
        log.error("fatal error in serve loop", error=str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
