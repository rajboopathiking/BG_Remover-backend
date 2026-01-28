import io
import time
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from torchvision import transforms
from transformers import AutoModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fast MODNet Background Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------
# FIX for Windows Proactor issues
# ---------------------------
# Force SelectorEventLoop (prevents _call_connection_lost crash)
# import asyncio
# if asyncio.get_event_loop_policy().__class__.__name__ == "WindowsProactorEventLoopPolicy":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------
# Performance tuning
# ---------------------------
torch.set_num_threads(1)

# ---------------------------
# Load MODNet once
# ---------------------------
print("Loading MODNet model...")
device = torch.device("cpu")

model = AutoModel.from_pretrained(
    "boopathiraj/MODNet",
    trust_remote_code=True
)

model.to(device)
model.eval()

print("Model loaded.")

# ---------------------------
# FastAPI app
# ---------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "Fast MODNet Background Remover",
        "default_output": "jpeg",
        "ref_size": 512
    }

# ---------------------------
# Preprocess
# ---------------------------
def preprocess_pil(image: Image.Image, ref_size: int = 512):
    image = image.convert("RGB")
    w, h = image.size

    scale = ref_size / max(h, w)
    scale = min(scale, 1.0)

    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)

    pad_x = (ref_size - new_w) // 2
    pad_y = (ref_size - new_h) // 2

    padded = Image.new("RGB", (ref_size, ref_size), (0, 0, 0))
    padded.paste(image_resized, (pad_x, pad_y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    tensor = transform(padded).unsqueeze(0)

    meta = {
        "orig_size": (w, h),
        "new_size": (new_w, new_h),
        "pad": (pad_x, pad_y)
    }

    return tensor, meta

# ---------------------------
# Postprocess
# ---------------------------
def postprocess_matte(matte: np.ndarray, meta: dict):
    w, h = meta["orig_size"]
    new_w, new_h = meta["new_size"]
    pad_x, pad_y = meta["pad"]

    matte = matte[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

    return matte

# ---------------------------
# Core background removal
# ---------------------------
def remove_background(image: Image.Image, background: str = "white"):
    inp, meta = preprocess_pil(image)
    inp = inp.to(device)

    with torch.no_grad():
        _, _, matte = model(inp, True)

    matte = matte[0, 0].cpu().numpy()
    matte = postprocess_matte(matte, meta)

    alpha = (matte * 255).astype(np.uint8)

    # Light edge cleanup
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    img_np = np.array(image)

    # Transparent output
    if background == "transparent":
        alpha_3 = np.expand_dims(alpha, axis=2)
        rgba = np.concatenate([img_np, alpha_3], axis=2)
        return Image.fromarray(rgba, mode="RGBA"), "png"

    # White background (fast JPEG)
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_f = np.expand_dims(alpha_f, axis=2)

    white_bg = np.ones_like(img_np) * 255
    result = img_np * alpha_f + white_bg * (1 - alpha_f)
    result = result.astype(np.uint8)

    return Image.fromarray(result), "jpeg"

# ---------------------------
# API Endpoint with timing
# ---------------------------
@app.post("/remove")
async def remove_bg(
    file: UploadFile = File(...),
    background: str = Query(default="white", enum=["white", "transparent"])
):
    t0 = time.time()

    image_bytes = await file.read()
    t1 = time.time()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t2 = time.time()

    result_img, out_type = remove_background(image, background=background)
    t3 = time.time()

    # OPTIONAL: Limit output resolution (huge speed win)
    MAX_OUTPUT = 1280
    w, h = result_img.size
    scale = min(MAX_OUTPUT / max(w, h), 1.0)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        result_img = result_img.resize(new_size, Image.BILINEAR)

    buf = io.BytesIO()

    if out_type == "png":
        result_img.save(buf, format="PNG")
        media_type = "image/png"
    else:
        result_img.save(buf, format="JPEG", quality=90, subsampling=0)
        media_type = "image/jpeg"

    buf.seek(0)
    t4 = time.time()

    print(f"""
    ---- Timing ----
    Read upload      : {t1 - t0:.2f}s
    Decode image     : {t2 - t1:.2f}s
    Inference+post   : {t3 - t2:.2f}s
    Encode+resize    : {t4 - t3:.2f}s
    TOTAL SERVER     : {t4 - t0:.2f}s
    Input size (MB) : {len(image_bytes)/1024/1024:.2f}
    Output size (MB): {buf.getbuffer().nbytes/1024/1024:.2f}
    ----------------
    """)

    return StreamingResponse(buf, media_type=media_type)

# uvicorn.run(
#         "main:app",
#         host="127.0.0.1",
#         port=8080, 
#         workers=1,
#         log_level="info",
#         loop="asyncio",
#         reload=True
#     )