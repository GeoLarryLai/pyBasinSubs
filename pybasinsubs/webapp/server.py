# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Local launcher (127.0.0.1, free port, opens the browser, clean shutdown). Same shape as pyPetrograph's."""
from __future__ import annotations

import socket
import threading
import webbrowser

HOST = "127.0.0.1"
DEFAULT_PORT = 8767


def free_port(preferred: int = DEFAULT_PORT) -> int:
    """``preferred`` if it is free, else any free port."""
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind((HOST, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise OSError("no free port")


def serve(app, name: str, port: int | None = None, open_browser: bool = True, page: str = "") -> None:
    """Run ``app`` on http://127.0.0.1:<free port> until Ctrl+C or the Quit button (POST /api/quit)."""
    import uvicorn
    port = free_port(port or DEFAULT_PORT)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning"))
    app.state.server = server
    url = f"http://{HOST}:{port}/" + (f"#{page}" if page else "")
    print(f"{name}: {url}  (local only; Ctrl+C or the Quit button to stop)", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    print(f"{name}: stopped", flush=True)
