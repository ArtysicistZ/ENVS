#!/usr/bin/env python3
"""
OSWorld VM Screen Viewer — serves live screenshots via HTTP for VSCode Ports.

Usage:
  python scripts/vm_screen_viewer.py [VM_IP] [PORT]

  VM_IP  : private IP of the OSWorld VM (default: 172.31.17.157)
  PORT   : local port to serve on (default: 5910)

Then in VSCode: Ports panel → Add Port → 5910 → open in browser.

The page auto-refreshes the screenshot every 1.5s.
You can also use the remote env server as the screenshot source:
  VM_URL=http://172.31.17.157:5000  python scripts/vm_screen_viewer.py
"""

import sys
import os
import time
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen
from urllib.error import URLError

VM_IP   = os.environ.get("VM_IP",  sys.argv[1] if len(sys.argv) > 1 else "172.31.17.157")
PORT    = int(os.environ.get("PORT", sys.argv[2] if len(sys.argv) > 2 else "5910"))
VM_URL  = os.environ.get("VM_URL", f"http://{VM_IP}:5000")
REFRESH = float(os.environ.get("REFRESH_S", "1.5"))

_last_png: bytes = b""
_lock = threading.Lock()
_fetch_errors = 0

HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OSWorld VM Screen — {vm_url}</title>
  <style>
    body {{ margin:0; background:#111; display:flex; flex-direction:column;
            align-items:center; font-family:monospace; color:#ccc; }}
    h2   {{ margin:8px 0 4px; font-size:14px; color:#8cf; }}
    #ts  {{ font-size:11px; color:#888; margin-bottom:4px; }}
    img  {{ max-width:100vw; border:1px solid #444; cursor:crosshair; }}
    #status {{ font-size:11px; color:#f88; margin-top:4px; }}
  </style>
</head>
<body>
  <h2>OSWorld VM &mdash; {vm_url}</h2>
  <div id="ts">loading…</div>
  <img id="sc" src="/screenshot.png" alt="VM screen">
  <div id="status"></div>
  <script>
    const REFRESH = {refresh};
    const img = document.getElementById('sc');
    const ts  = document.getElementById('ts');
    const st  = document.getElementById('status');
    let errors = 0;
    function refresh() {{
      const nonce = Date.now();
      const next  = new Image();
      next.onload  = () => {{ img.src = next.src; ts.textContent = new Date().toLocaleTimeString(); errors = 0; st.textContent=''; }};
      next.onerror = () => {{ errors++; st.textContent = 'fetch error #' + errors; }};
      next.src = '/screenshot.png?t=' + nonce;
    }}
    setInterval(refresh, REFRESH * 1000);
    refresh();

    // Click to coordinate logger
    img.addEventListener('click', e => {{
      const r = img.getBoundingClientRect();
      const sx = img.naturalWidth, sy = img.naturalHeight;
      const x = Math.round((e.clientX - r.left) * sx / r.width);
      const y = Math.round((e.clientY - r.top)  * sy / r.height);
      console.log('click at VM pixel (' + x + ', ' + y + ')');
      st.textContent = 'clicked VM pixel (' + x + ', ' + y + ')';
    }});
  </script>
</body>
</html>
""".format(vm_url=VM_URL, refresh=REFRESH)


def _fetch_screenshot() -> bytes:
    url = f"{VM_URL}/screenshot"
    with urlopen(url, timeout=5) as r:
        return r.read()


def _background_fetcher():
    global _last_png, _fetch_errors
    while True:
        try:
            data = _fetch_screenshot()
            with _lock:
                _last_png = data
            _fetch_errors = 0
        except Exception as exc:
            _fetch_errors += 1
            if _fetch_errors <= 3 or _fetch_errors % 10 == 0:
                print(f"[fetcher] error #{_fetch_errors}: {exc}", flush=True)
        time.sleep(REFRESH)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global _last_png
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif path == "/screenshot.png":
            with _lock:
                data = _last_png
            if data:
                self._send(200, "image/png", data)
            else:
                # Fetch synchronously if cache is empty
                try:
                    data = _fetch_screenshot()
                    with _lock:
                        _last_png = data
                    self._send(200, "image/png", data)
                except Exception as exc:
                    self._send(503, "text/plain", f"VM not reachable: {exc}".encode())
        else:
            self._send(404, "text/plain", b"not found")


def main():
    t = threading.Thread(target=_background_fetcher, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"VM Screen Viewer — fetching from {VM_URL}/screenshot", flush=True)
    print(f"Serving on http://0.0.0.0:{PORT}", flush=True)
    print(f"VSCode: Ports panel → Add Port → {PORT} → open in browser", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
