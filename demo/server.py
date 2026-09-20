"""Phase 10 -- demo UI.

Deliberately stdlib-only (http.server, no Flask/FastAPI): the library's own
selling point is "no forced dependency," and a demo that needs `pip install`
before you can even see it working undercuts that. Serves one static page
(index.html) and one JSON endpoint that runs the real Firewall.

Run: python demo/server.py
Then open http://localhost:8420
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fireguard"))
from fireguard.types import Chunk, RetrievalSet

from fireguard import Firewall

PORT = 8420
STATIC_DIR = Path(__file__).parent

print("Loading fireguard (first run downloads the ONNX model, ~5s)...")
FIREWALL = Firewall()
print("Ready.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the console clean -- this is a demo, not a service

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html")
        else:
            self.send_error(404)

    def _serve_file(self, name: str, content_type: str):
        path = STATIC_DIR / name
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/scan":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = self._run_scan(payload)
            status, body = 200, result
        except Exception as e:
            status, body = 400, {"error": str(e)}

        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _run_scan(self, payload: dict) -> dict:
        query = payload.get("query", "")
        chunks = [
            Chunk(text=c["text"], source_uri=c["source_uri"], tier=c.get("tier", "open_web"))
            for c in payload.get("chunks", [])
        ]
        result = FIREWALL.scan(RetrievalSet(query=query, chunks=chunks))

        order = {c["source_uri"]: i for i, c in enumerate(payload.get("chunks", []))}
        all_chunks = result.approved_chunks + result.flagged_chunks
        all_chunks.sort(key=lambda c: order.get(c.source_uri, 0))

        return {
            "verdict": result.verdict,
            "reasons": result.reasons,
            "chunks": [
                {
                    "source_uri": c.source_uri,
                    "tier": c.tier,
                    "text": c.text,
                    "trust": c.trust,
                    "threshold": c.threshold,
                    "injection_score": c.injection_score,
                    "flagged": c.flagged,
                    "flag_reasons": c.flag_reasons,
                }
                for c in all_chunks
            ],
        }


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"Demo running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
