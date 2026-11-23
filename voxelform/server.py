"""Local-only demo with a persistent worker and an active-time clock."""

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .context import ContextLibrary
from .session import Session

ROOT = Path(__file__).resolve().parents[1]
library = ContextLibrary()
torch.set_num_threads(max(1, int(os.getenv("VOXELFORM_THREADS", "1"))))

session = Session(library=library)
app = FastAPI(title="VoxelForm RL")
app.mount("/assets", StaticFiles(directory=ROOT / "public"), name="assets")
# Serve the actual sibling renderer and its dependencies, without copying or
# modifying them. VoxelForm supplies only its own massing overlay adapter.
viewer_public = library.root / "frontend/public"
if not (viewer_public / "scripts/urban_viewer.js").is_file():
    viewer_public = library.root / "public"
for directory in ("scripts", "styles", "vendor"):
    app.mount(
        f"/shared-viewer/{directory}",
        StaticFiles(directory=viewer_public / directory),
        name=f"shared-viewer-{directory}",
    )


@app.get("/")
def home():
    return FileResponse(ROOT / "public/index.html")


@app.get("/api/state")
def state():
    return session.state()


@app.get("/api/sites")
def sites():
    return [
        dict(
            siteId=r["site_id"],
            city=r.get("city_name", "Urban site"),
            cityCode=r.get("city_code", ""),
            tier=r.get("area_tier", ""),
            area=r.get("site_area_m2"),
        )
        for r in library.catalog()
    ]


@app.post("/api/control")
async def control(request: Request):
    data = await request.json()
    command = data.get("command")
    with session.lock:
        if not session.ready:
            raise HTTPException(409, "Environment is still preparing")
        try:
            if command == "run":
                session.start(training=bool(data.get("training", True)))
            elif command == "pause":
                session.pause()
            elif command == "step":
                session.start(training=bool(data.get("training", False)), single=True)
            elif command == "masks":
                session.configure_masks(
                    data.get("mode"),
                    data.get("deepInterior", False),
                    data.get("maximumInteriorVoids", 1),
                    data.get("maximumEditVoxels", 256),
                    data.get("editAllocation", "probability"),
                )
            elif command == "grid":
                session.configure_grid(data.get("alignment", "world"))
            elif command == "reset":
                session.pause()
                session.next_episode()
            elif command == "settings":
                session.configure_far(
                    data.get("targetFar"),
                    data.get("densityPreset", "custom"),
                    data.get("farTolerance"),
                )
            elif command == "site":
                session.change_site(
                    data.get("source", "cached_osm"),
                    data.get("siteId"),
                    data.get("plan", "rectangle"),
                )
            else:
                raise ValueError("Unknown command")
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    return dict(ok=True)


@app.get("/api/policy")
def save():
    with session.lock:
        if not session.ready:
            raise HTTPException(409, "Not ready")
        buffer = io.BytesIO()
        torch.save(session.policy.state(), buffer)
    return Response(
        buffer.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="voxelform-policy.pt"'},
    )


@app.post("/api/policy")
async def load(request: Request):
    raw = await request.body()
    if len(raw) > 20_000_000:
        raise HTTPException(413, "Checkpoint too large")
    with session.lock:
        session.pause()
        try:
            session.load_policy(
                torch.load(
                    io.BytesIO(raw), map_location=session.device, weights_only=True
                )
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    return dict(ok=True)
