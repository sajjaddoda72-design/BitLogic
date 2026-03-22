"""FastAPI server for local development of the Bitcoin trading simulator.

The production deployment on Netlify runs entirely client-side (the
simulation engine is ported to JavaScript inside index.html).  This
server is only needed for local development and testing of the Python
backend.

Usage:
    python server.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bitcoin_trading_sim import run_simulation

app = FastAPI(title="CryptoSim API", version="1.0.0")

# Allow all origins so the frontend can call the API from any context
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent

# Serve static files (manifest.json, sw.js, icons/, etc.)
app.mount("/icons", StaticFiles(directory=str(STATIC_DIR / "icons")), name="icons")


@app.get("/manifest.json")
def serve_manifest() -> FileResponse:
    """Serve the PWA manifest."""
    return FileResponse(STATIC_DIR / "manifest.json", media_type="application/json")


@app.get("/sw.js")
def serve_sw() -> FileResponse:
    """Serve the service worker."""
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the main frontend HTML page."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/simulate")
def simulate(
    days: int = Query(default=90, ge=7, le=730, description="Lookback period in days"),
    volatility: float = Query(
        default=0.03, ge=0.001, le=0.20, description="Daily volatility (0.01 = 1%)"
    ),
    leverage: float = Query(
        default=1.0, ge=1.0, le=100.0, description="Leverage multiplier"
    ),
) -> JSONResponse:
    """Run the trading simulation and return results as JSON."""
    result = run_simulation(days=days, volatility=volatility, leverage=leverage)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
