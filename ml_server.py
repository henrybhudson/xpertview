import asyncio
import io
import json
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor  # official API
import cv2

# ---------------------------------------
# Config
# ---------------------------------------
MAX_FRAMES_BUFFER = 120 
DOWNSCALE_W = 960 
MASK_THRESHOLD = 0.0 
CLIP_BACK_FRAMES = 30 
INFERENCE_LOCK_TIMEOUT = 10.0

# ---------------------------------------
# Utilities
# ---------------------------------------
def bbox_from_polygon(poly_xy01: List[Tuple[float, float]], W: int, H: int) -> np.ndarray:
    xs = [p[0] * W for p in poly_xy01]
    ys = [p[1] * H for p in poly_xy01]
    x0, x1 = max(0, int(min(xs))), min(W-1, int(max(xs)))
    y0, y1 = max(0, int(min(ys))), min(H-1, int(max(ys)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)

def centroid_of_polygon(poly_xy01: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in poly_xy01]
    ys = [p[1] for p in poly_xy01]
    return (float(sum(xs)/len(xs)), float(sum(ys)/len(ys)))

def mask_to_polygon_norm(mask: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Return a simplified normalised polygon for the largest connected component."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # find contours on a binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    # approximate polygon to reduce points
    eps = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    H, W = mask.shape[:2]
    poly = []
    for p in approx.reshape(-1, 2):
        nx, ny = p[0] / W, p[1] / H
        poly.append((float(nx), float(ny)))
    return poly

# ---------------------------------------
# SAM2 wrapper
# ---------------------------------------
class Sam2Tracker:
    """
    Minimal wrapper around SAM2VideoPredictor for promptable video object segmentation.
    We rebuild a short 'clip' (last N frames) on demand and propagate to the latest frame.
    This is efficient enough for a PoC at 720p and ~10–15 FPS.
    """
    def __init__(self, model_id: str = "facebook/sam2-hiera-large"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2VideoPredictor.from_pretrained(model_id).to(device)
        self.device = device

    def track_polygon_over_clip(
        self,
        frames_bgr: List[np.ndarray],      # list of HxWx3 uint8 (BGR or RGB both fine; PIL-converted below)
        anchor_idx: int,                   # index in frames_bgr to which the coach's prompt refers
        poly_xy01: List[Tuple[float, float]]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Run SAM2 over a short clip, adding a box + point prompt at anchor frame,
        then propagate to the end; return polygon mask on the last frame (normalised).
        """
        if not frames_bgr:
            return None
        # SAM2 expects numpy (RGB). Ensure RGB.
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else f for f in frames_bgr]
        H, W = frames_rgb[0].shape[:2]

        # init state (can be list of frames)
        with torch.inference_mode():
            state = self.predictor.init_state(frames_rgb)

            # Build prompts at the anchor frame
            box = bbox_from_polygon(poly_xy01, W, H)
            cx, cy = centroid_of_polygon(poly_xy01)
            # Use a centre positive click to reduce ambiguity
            pts = np.array([[cx, cy]], dtype=np.float32)  # normalised in [0,1]
            labels = np.array([1], dtype=np.int64)

            # New API supports normalised coords; pass frame_idx and an obj_id (1)
            # NOTE: this signature is consistent with SAM2 community wrappers and copies of the predictor.
            # Official docs show add_new_points_or_box(state, <prompts>); both paths exist.
            self.predictor.add_new_points_or_box(
                state,
                frame_idx=int(anchor_idx),
                obj_id=1,
                points=pts,
                labels=labels,
                box=box,
                normalize_coords=True,
            )

            # Propagate from anchor to the end (so we get latest frame)
            # This generator yields (frame_idx, object_ids, masks_logits)
            last_mask = None
            last_was = None
            for f_idx, obj_ids, masks_logits in self.predictor.propagate_in_video(
                state, start_frame_idx=int(anchor_idx)
            ):
                # pick our object id (1)
                obj_ids = list(obj_ids)
                if 1 in obj_ids:
                    k = obj_ids.index(1)
                    # convert logits to binary mask
                    mlog = masks_logits[k]
                    if isinstance(mlog, torch.Tensor):
                        m = (mlog > MASK_THRESHOLD).detach().to("cpu").numpy().astype(np.uint8)
                    else:
                        m = (mlog > 0).astype(np.uint8)
                    last_mask = m
                    last_was = f_idx

            if last_mask is None:
                return None

            # Return polygon on the last processed frame
            poly = mask_to_polygon_norm(last_mask)
            return poly

# ---------------------------------------
# FastAPI + WebSockets
# ---------------------------------------
app = FastAPI()

@dataclass
class RoomState:
    frames: Deque[np.ndarray]         # recent frames buffer
    last_size: Tuple[int, int]        # (H, W)
    last_ts: float
    pending_poly: Optional[List[Tuple[float, float]]]  # normalised polygon when coach circles
    tracker: Optional[Sam2Tracker]
    lock: asyncio.Lock

rooms: Dict[str, RoomState] = {}

def get_room(room_id: str) -> RoomState:
    if room_id not in rooms:
        rooms[room_id] = RoomState(
            frames=deque(maxlen=MAX_FRAMES_BUFFER),
            last_size=(0, 0),
            last_ts=0.0,
            pending_poly=None,
            tracker=Sam2Tracker(),   # load once per room (small number of rooms for PoC)
            lock=asyncio.Lock(),
        )
    return rooms[room_id]

@app.websocket("/ml/v1")
async def ml_ws(ws: WebSocket):
    await ws.accept()
    room_id: Optional[str] = None

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    data = json.loads(msg["text"])
                    t = data.get("type")

                    if t == "join-room":
                        room_id = data["roomId"]
                        _ = get_room(room_id)  # ensure created
                        continue

                    if t == "frame":
                        # Next message is the JPEG blob
                        binmsg = await ws.receive_bytes()
                        room = get_room(room_id)
                        img = Image.open(io.BytesIO(binmsg)).convert("RGB")
                        arr = np.array(img)  # HxWx3 RGB
                        # Convert to BGR for OpenCV ops later; SAM2 accepts RGB list
                        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                        room.frames.append(arr_bgr)
                        room.last_size = (arr_bgr.shape[0], arr_bgr.shape[1])
                        room.last_ts = float(data.get("ts", 0.0))

                        # If we have an active prompt, (re)run short-clip tracking and return latest mask
                        if room.pending_poly is not None:
                            # Build a short clip ending at the newest frame
                            clip = list(room.frames)[-CLIP_BACK_FRAMES:]
                            anchor_idx = len(clip) - 1  # prompt refers to "now"
                            try:
                                async with asyncio.timeout(INFERENCE_LOCK_TIMEOUT):
                                    async with room.lock:
                                        poly = room.tracker.track_polygon_over_clip(
                                            clip, anchor_idx, room.pending_poly
                                        )
                            except TimeoutError:
                                poly = None

                            if poly:
                                await ws.send_text(json.dumps({
                                    "type": "mask",
                                    "objectId": 1,
                                    "polygon": poly,
                                    "conf": 1.0  # SAM2 doesn't emit a single scalar conf; keep API simple
                                }))
                        continue

                    if t == "prompt":
                        # Coach circled something: store the normalised polygon and run once immediately
                        room = get_room(room_id)
                        room.pending_poly = [(float(x), float(y)) for x, y in data["polygon"]]

                        # Run immediate feedback on latest frame buffer if present
                        if len(room.frames) > 0:
                            clip = list(room.frames)[-CLIP_BACK_FRAMES:]
                            anchor_idx = len(clip) - 1
                            try:
                                async with asyncio.timeout(INFERENCE_LOCK_TIMEOUT):
                                    async with room.lock:
                                        poly = room.tracker.track_polygon_over_clip(
                                            clip, anchor_idx, room.pending_poly
                                        )
                            except TimeoutError:
                                poly = None

                            if poly:
                                await ws.send_text(json.dumps({
                                    "type": "mask",
                                    "objectId": 1,
                                    "polygon": poly,
                                    "conf": 1.0
                                }))
                        continue

                # ignore stray binary (we always pair after a 'frame' header)
            else:
                # ping/pong etc.
                pass

    except WebSocketDisconnect:
        # Clean-up is optional for PoC; keep room state for a while
        return
