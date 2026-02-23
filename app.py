# we use diffusers local inpainting pipeline for sd2-community and similar models; falls back to HF Inference API if needed)
import os, logging, requests 
from io import BytesIO
from base64 import b64decode
from typing import Optional, Any, List, Dict 
from PIL import Image
import torch
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient, InferenceTimeoutError
from dotenv import load_dotenv 
try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception:
    # as a fallback, DiffusionPipeline may work for some models, but inpaint pipeline preferred.
    from diffusers import DiffusionPipeline as StableDiffusionInpaintPipeline  

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

HF_TOKEN = os.getenv("HF_TOKEN")
HF_INPAINT_MODEL = os.getenv("HF_INPAINT_MODEL", "sd2-community/stable-diffusion-2-inpainting")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "Salesforce/blip-image-captioning-base")
FALLBACK_INPAINT_MODEL = os.getenv("FALLBACK_INPAINT_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

_http = requests.Session()
_http.headers.update({"User-Agent": "img-gen/1.0"})
_hf_client: Optional[InferenceClient] = None
_PIPELINES: Dict[str, Any] = {}  # cache of loaded diffusers pipelines


def get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set.")
        _hf_client = InferenceClient(token=HF_TOKEN)
    return _hf_client
 

class Captioner:
    def __init__(self, model_name: str = CAPTION_MODEL_NAME):
        # prefer CPU for small caption model if GPU not available; huggingface transformers will pick device if moved.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading caption model %s on device %s", model_name, self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=64)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

_captioner = Captioner()
 
# helper func.(s) 
def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def download_image_from_url(url: str) -> Image.Image:
    resp = _http.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def _pil_from_component(value: Any) -> Optional[Image.Image]:
    """
    Convert Gradio Image / ImageEditor values (dict / PIL.Image / bytes / np.array / url) into PIL.Image.
    """
    if value is None:
        return None

    # ImageEditor returns dicts: prefer 'composite' then 'background' then first layer
    if isinstance(value, dict):
        composite = value.get("composite")
        if composite:
            return _pil_from_component(composite)
        background = value.get("background")
        if background:
            return _pil_from_component(background)
        layers = value.get("layers")
        if layers and isinstance(layers, (list, tuple)) and len(layers) > 0:
            return _pil_from_component(layers[0])
        return None

    # already a PIL image
    if isinstance(value, Image.Image):
        return value

    # raw bytes
    if isinstance(value, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(value)).convert("RGB")
        except Exception:
            return None

    # string: URL or path
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            try:
                return download_image_from_url(value)
            except Exception:
                return None
        # Try local file path
        try:
            return Image.open(value).convert("RGB")
        except Exception:
            pass

    # numpy array -> PIL
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            return Image.fromarray(value)
    except Exception:
        pass

    return None

def _parse_hf_image_result(result: Any) -> Image.Image:
    """
    Parse output from InferenceClient.image_to_image into PIL.Image (RGB).
    Supports: PIL.Image, bytes, dicts with 'images'/'image' or base64/data URLs or remote URLs.
    """
    if isinstance(result, Image.Image):
        return result.convert("RGB")

    if isinstance(result, (bytes, bytearray)):
        return Image.open(BytesIO(result)).convert("RGB")

    if isinstance(result, dict):
        candidate_keys = [
            "image", "images",
            "generated_image", "generated_images",
            "image_base64", "b64_json",
            "data", "result", "output"
        ]
        candidates = []
        for k in candidate_keys:
            if k in result:
                candidates.append(result[k])
        if not candidates:
            # fallback to first non-empty dict value
            for v in result.values():
                if v is not None:
                    candidates.append(v)
                    break
        if not candidates:
            raise ValueError(f"HF result dict empty/unrecognized: keys={list(result.keys())}")

        val = candidates[0]

        if isinstance(val, (list, tuple)) and len(val) > 0:
            val = val[0]

        if isinstance(val, (bytes, bytearray)):
            return Image.open(BytesIO(val)).convert("RGB")

        if isinstance(val, Image.Image):
            return val.convert("RGB")

        if isinstance(val, str):
            # data URL
            if val.startswith("data:"):
                try:
                    b64part = val.split(",", 1)[1]
                    raw = b64decode(b64part)
                    return Image.open(BytesIO(raw)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Failed to parse data URL image: {e}")
            # remote URL
            if val.startswith("http://") or val.startswith("https://"):
                try:
                    return download_image_from_url(val)
                except Exception as e:
                    raise ValueError(f"Failed to download image at URL returned from HF: {e}")
            # try plain base64
            try:
                raw = b64decode(val, validate=True)
                try:
                    return Image.open(BytesIO(raw)).convert("RGB")
                except Exception:
                    pass
            except Exception:
                pass

        if isinstance(val, dict):
            return _parse_hf_image_result(val)

        raise ValueError(f"Unrecognized image payload in HF response (type={type(val)}).")

    raise ValueError(f"Unrecognized HF image result type: {type(result)}")

def _is_repo_not_found_exception(e: Exception) -> bool:
    """
    Heuristic to detect repository-not-found / 404 / model access errors from HF client.
    """
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            status = getattr(resp, "status_code", None)
            if status == 404:
                return True
        except Exception:
            pass

    msg = str(e).lower()
    indicators = [
        "repository not found",
        "repo not found",
        "404",
        "not found",
        "model not found",
        "could not find model",
        "no inference provider found",
        "access denied",
        "gated",
    ]
    return any(ind in msg for ind in indicators)


# def _repo_access_issue_message(e: Exception) -> Optional[str]:
#     resp = getattr(e, "response", None)
#     if resp is not None:
#         status = getattr(resp, "status_code", None)
#         if status == 404:
#             return "404: model not found on Hugging Face — check model id or that the model is public."
#         if status in (401, 403):
#             return ("401/403: authentication/access denied. Make sure your HF_TOKEN is the same account "
#                     "that accepted any gated model terms, and that the token has read scope.")
#     msg = str(e).lower()
#     if "gated" in msg or "access denied" in msg:
#         return ("Model appears gated/blocked. Open the model page while logged in and accept license / access.")
#     return None


# diffusers pipeline helper 
def _device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # mps support (Apple silicon) — use bfloat16/float16 only if supported; fallback to float32
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS has limited dtype support; use float32
        return "mps", torch.float32
    return "cpu", torch.float32


def load_local_pipeline(model_id: str):
    """
    Try to load a local diffusers inpainting pipeline for model_id.
    On success, cache and return pipeline.
    Raises Exception if not loadable.
    """
    if model_id in _PIPELINES:
        return _PIPELINES[model_id]

    device, dtype = _device_and_dtype()
    logger.info("Attempting to load local diffusers pipeline for %s (device=%s, dtype=%s)", model_id, device, dtype)

    # Try the inpaint pipeline class first
    try:
        # If specific pipeline class is available, use it
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # disable safety checker for faster loading in many Spaces (optional)
        )
    except Exception as e:
        logger.warning("StableDiffusionInpaintPipeline.from_pretrained failed for %s: %s", model_id, e)
        # Try generic DiffusionPipeline fallback if available
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        except Exception as e2:
            logger.exception("Generic DiffusionPipeline.from_pretrained also failed for %s: %s", model_id, e2)
            raise

    # Move to device
    try:
        if device == "cuda":
            pipe.to("cuda")
        elif device == "mps":
            pipe.to("mps")
        else:
            pipe.to("cpu")
    except Exception as e:
        logger.warning("Warning: moving pipeline to device %s raised: %s", device, e)

    # Try to enable memory efficient attention if supported
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _PIPELINES[model_id] = pipe
    logger.info("Loaded and cached pipeline for %s", model_id)
    return pipe


# gen. func. with fallback logic (local diffusers -> HF Inference) 
def generate_image(upload_method, img_url, uploaded_img, mask_img, negative_prompt, hf_model, mask_is_keep):
    if not HF_TOKEN:
        # we still can try local pipelines without HF_TOKEN
        logger.info("HF_TOKEN not set. Local pipelines only.")
    # Load base image (either from URL or uploaded)
    try:
        if upload_method == "URL":
            base_img = download_image_from_url(img_url)
        else:
            base_img = _pil_from_component(uploaded_img)
            if base_img is None:
                raise ValueError("Uploaded image could not be parsed.")
    except Exception as e:
        logger.exception("Image load failed")
        return None, "", f"Image load failed: {e}"

    # caption gen.
    try:
        caption = _captioner.caption(base_img)
        prompt = (
            f"Replace the background to match: '{caption}'. "
            "Keep the masked object unchanged."
        )
    except Exception as e:
        logger.exception("Caption failed")
        return None, "", f"Caption failed: {e}"

    # Prepare mask PIL (diffusers expects mask_image as RGBA/L for inpaint)
    mask_pil = _pil_from_component(mask_img)
    # If mask_is_keep is True, we may need to invert mask depending on model expectations.
    # Many inpainting pipelines expect white/255 = regions to inpaint. If mask editor marks the object white (keep),
    # we may need to invert. Here we assume mask image white = area to inpaint. The UI checkbox exists in case you wanna to invert.
    if mask_pil is not None:
        mask_for_pipe = mask_pil.convert("L")
        if mask_is_keep:
            # If the mask represents the object to keep (white), invert so white = area to inpaint.
            # (This choice depends on how the Gradio editor produces the mask; adjust if you observe reversed behavior.)
            mask_for_pipe = Image.eval(mask_for_pipe, lambda px: 255 - px)
    else:
        mask_for_pipe = None

    # models to try: explicit hf_model input -> default -> fallback
    models_to_try: List[str] = []
    if hf_model and hf_model.strip():
        models_to_try.append(hf_model.strip())
    models_to_try.append(HF_INPAINT_MODEL)
    if FALLBACK_INPAINT_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_INPAINT_MODEL)

    last_exc: Optional[Exception] = None
    tried_models: List[str] = []

    for model_candidate in models_to_try:
        tried_models.append(model_candidate)
        # First attempt local diffusers pipeline (works for sd2-community model you provided)
        try:
            logger.info("Attempting local diffusers for model: %s", model_candidate)
            pipe = load_local_pipeline(model_candidate)
            # Run the pipeline
            # For diffusers inpaint, typical call is pipe(prompt=..., image=..., mask_image=..., negative_prompt=..., num_inference_steps=25)
            pipe_args = dict(
                prompt=prompt,
                image=base_img,
                mask_image=mask_for_pipe,
                negative_prompt=negative_prompt or None,
            )
            # remove None entries
            pipe_args = {k: v for k, v in pipe_args.items() if v is not None}
            logger.info("Running pipeline %s with args: prompt_len=%d, mask_present=%s", model_candidate, len(prompt), mask_for_pipe is not None)
            output = pipe(**pipe_args)
            # many pipelines return a PipelineOutput-like object with .images
            if hasattr(output, "images"):
                output_img = output.images[0]
            elif isinstance(output, list) and len(output) > 0:
                output_img = output[0]
            else:
                # maybe it returned a dict
                if isinstance(output, dict):
                    # try parse
                    # If it contains "images" key, use it
                    imgs = output.get("images") or output.get("outputs") or output.get("result")
                    if isinstance(imgs, (list, tuple)):
                        output_img = imgs[0]
                    else:
                        raise ValueError("Unexpected pipeline output format.")
                else:
                    raise ValueError("Unexpected pipeline output type.")
            status_msg = f"Success (local diffusers model used: {model_candidate})"
            return output_img.convert("RGB"), caption, status_msg
        except Exception as e:
            logger.exception("Local pipeline failed for %s: %s", model_candidate, e)
            last_exc = e
            # If the error looks like model not found / cannot access / no such repo, try the next candidate (fallback).
            # If it's a GPU/Memory error, surface a helpful message.
            if "No such file or directory" in str(e) or "repository" in str(e).lower() or "not found" in str(e).lower():
                logger.warning("Model %s not found locally or repo not present: %s", model_candidate, e)
                # try next candidate (fall through)
                continue
            # If it fails due to out-of-memory or CUDA not available, give a clear message and continue to try other models
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                return None, caption, f"Inpainting failed for {model_candidate}: CUDA out of memory. Try a smaller model or enable GPU for this Space."
            # Otherwise, fallback to trying the InferenceClient below (if HF_TOKEN present)
        # If local pipeline failed, and HF_TOKEN exists, try HF Inference API
        if HF_TOKEN:
            try:
                logger.info("Attempting HF Inference API for model: %s", model_candidate)
                client = get_hf_client()
                base_bytes = image_to_bytes(base_img)
                mask_bytes = image_to_bytes(mask_for_pipe) if mask_for_pipe is not None else None
                result = client.image_to_image(
                    base_bytes,
                    model=model_candidate,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    mask_image=mask_bytes,
                )
                output_img = _parse_hf_image_result(result)
                status_msg = f"Success (Inference API model used: {model_candidate})"
                return output_img, caption, status_msg
            except Exception as e:
                logger.exception("Inference API call failed for %s: %s", model_candidate, e)
                last_exc = e
                if _is_repo_not_found_exception(e):
                    # try next model candidate
                    logger.warning("Model %s not found / inaccessible via Inference API, trying next.", model_candidate)
                    continue
                if isinstance(e, InferenceTimeoutError):
                    return None, caption, f"Inpainting failed (timeout) using {model_candidate}: {e}"
                # otherwise continue to next candidate
                continue

    # If we get here, all candidates failed
    err_text = str(last_exc) if last_exc is not None else "Unknown error"
    return None, caption, f"Inpainting failed for tried models {tried_models}: {err_text}"


# ui
def invert_box(upload_method):
    if upload_method == "URL":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks() as demo:
    gr.Markdown("## Caption-based Background Regeneration (with local diffusers inpainting if available)")

    with gr.Row():
        upload_method = gr.Radio(["URL", "Upload"], value="URL")
        img_url = gr.Textbox(label="Image URL")
        uploaded_img = gr.Image(type="pil", visible=False)

    upload_method.change(invert_box, upload_method, [img_url, uploaded_img])

    mask_img = gr.ImageEditor(type="pil", label="Mask")

    mask_is_keep = gr.Checkbox(label="Mask shows object to KEEP (invert for inpainting)", value=True)

    negative_prompt = gr.Textbox(value="blur, ugly")

    hf_model = gr.Textbox(value=HF_INPAINT_MODEL, label="HF Model (try custom override)")

    generate_btn = gr.Button("Generate")

    caption_box = gr.Textbox(label="Generated Caption")
    status = gr.Textbox(label="Status")
    output_image = gr.Image(type="pil")

    generate_btn.click(
        generate_image,
        inputs=[upload_method, img_url, uploaded_img, mask_img, negative_prompt, hf_model, mask_is_keep],
        outputs=[output_image, caption_box, status],
    )

if __name__ == "__main__":
    demo.launch()