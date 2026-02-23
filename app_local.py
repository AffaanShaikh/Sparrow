"""
- Uses a local transformers captioning model to generate captions for input images.
- Using the image + generated caption we generate an output image using a local diffusers inpainting pipeline.
- optionally treat the painted mask as the OBJECT TO KEEP (i.e. preserve the masked object while regenerating the background
  based on the generated caption). This is implemented by inverting the mask before sending to the inpainting model.

Environment variables:
    - HF_TOKEN (required to call HF Inference client)
    - EAGER_LOAD_CAPTION_MODEL (optional) : "1" to load caption model at startup
"""

import os, sys, logging, requests
from io import BytesIO
from base64 import b64encode, b64decode
from typing import Optional, Tuple, Any, Dict
from PIL import Image, UnidentifiedImageError
import math
import torch
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient, InferenceTimeoutError
try:
    from diffusers import DiffusionPipeline
except Exception:
    DiffusionPipeline = None
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=getattr(logging, "INFO", logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("log-file.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_INPAINT_MODEL = os.getenv("HF_INPAINT_MODEL", "stabilityai/stable-diffusion-2-inpainting")
CAPTION_MODEL_NAME = os.getenv("CAPTION_MODEL_NAME", "Salesforce/blip-image-captioning-base")

if not HF_TOKEN:
    logger.error("HF_TOKEN is not set. Please set your Hugging Face token in HF_TOKEN.")
else:
    logger.info("HF_TOKEN found in environment (will be used for InferenceClient).")

logger.info("Default HF inpainting model: %s", HF_INPAINT_MODEL)
logger.info("Caption model: %s", CAPTION_MODEL_NAME)

_http = requests.Session()
_http.headers.update({"User-Agent": "img-gen/1.0"})

_hf_client: Optional[InferenceClient] = None
def get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set (required to call the Hugging Face Inference API).")
        _hf_client = InferenceClient(token=HF_TOKEN)
    return _hf_client

class Captioner:
    def __init__(self, model_name: str = CAPTION_MODEL_NAME, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self._loaded = False
        logger.info("Captioner initialized (model=%s device=%s)", self.model_name, self.device)

    def load(self):
        if self._loaded:
            return
        logger.info("Loading caption model '%s' (may take a moment)...", self.model_name)
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("Caption model loaded.")
        except Exception as e:
            logger.exception("Failed to load caption model '%s': %s", self.model_name, e)
            raise

    def caption(self, image: Image.Image, max_length: int = 64, num_beams: int = 4) -> str:
        if image is None:
            raise ValueError("No image provided for captioning.")
        if not self._loaded:
            self.load()
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            logger.info("Caption generated: %s", caption)
            return caption
        except Exception as e:
            logger.exception("Captioning failed: %s", e)
            raise

_captioner: Optional[Captioner] = None
def get_captioner() -> Captioner:
    global _captioner
    if _captioner is None:
        _captioner = Captioner()
        if os.getenv("EAGER_LOAD_CAPTION_MODEL", "0") == "1":
            _captioner.load()
    return _captioner

# helpers
def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def image_to_b64(img: Image.Image) -> str:
    return b64encode(image_to_bytes(img, fmt="JPEG")).decode("utf-8")

def download_image_from_url(url: str) -> Image.Image:
    resp = _http.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img

def _ensure_pil_image_or_none(obj: Any) -> Optional[Image.Image]:
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj.convert("RGBA")
    if isinstance(obj, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(obj)).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Could not open mask bytes: {e}")
    if hasattr(obj, "read"):
        try:
            return Image.open(obj).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Could not open mask file-like object: {e}")
    if isinstance(obj, dict):
        for key in ("image", "mask", "img", "data", "composite"):
            if key in obj:
                v = obj[key]
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return Image.open(BytesIO(v)).convert("RGBA")
                    except Exception:
                        pass
                if isinstance(v, str):
                    if v.startswith("data:"):
                        try:
                            payload = v.split(",", 1)[1]
                            raw = b64decode(payload)
                            return Image.open(BytesIO(raw)).convert("RGBA")
                        except Exception:
                            pass
                    if len(v) > 100:
                        try:
                            raw = b64decode(v)
                            return Image.open(BytesIO(raw)).convert("RGBA")
                        except Exception:
                            pass
        for v in obj.values():
            if isinstance(v, str):
                if v.startswith("data:"):
                    try:
                        payload = v.split(",", 1)[1]
                        raw = b64decode(payload)
                        return Image.open(BytesIO(raw)).convert("RGBA")
                    except Exception:
                        continue
                if len(v) > 100:
                    try:
                        raw = b64decode(v)
                        return Image.open(BytesIO(raw)).convert("RGBA")
                    except Exception:
                        continue
    if isinstance(obj, str):
        if obj.startswith("data:"):
            try:
                payload = obj.split(",", 1)[1]
                raw = b64decode(payload)
                return Image.open(BytesIO(raw)).convert("RGBA")
            except Exception as e:
                raise ValueError(f"Could not decode data URI mask: {e}")
        if len(obj) > 100:
            try:
                raw = b64decode(obj)
                return Image.open(BytesIO(raw)).convert("RGBA")
            except Exception as e:
                raise ValueError(f"Could not decode base64 mask string: {e}")
    raise ValueError(f"Unsupported mask type: {type(obj)}")

# resize helper: scale image and mask up to nearest multiple of 64 (models expect multiples of 64)
def _to_multiple_of_64_dims(w: int, h: int) -> Tuple[int,int]:
    new_w = int(math.ceil(w / 64.0) * 64)
    new_h = int(math.ceil(h / 64.0) * 64)
    # avoid degenerate values
    new_w = max(64, new_w)
    new_h = max(64, new_h)
    return new_w, new_h

def _resize_image_and_mask_to_model(base: Image.Image, mask: Optional[Image.Image]) -> Tuple[Image.Image, Optional[Image.Image], Tuple[int,int]]:
    orig_size = (base.width, base.height)
    target_w, target_h = _to_multiple_of_64_dims(base.width, base.height)
    if (target_w, target_h) == orig_size:
        return base, mask, orig_size
    logger.info("Resizing input image from %s to model-friendly %s (multiple of 64).", orig_size, (target_w,target_h))
    base_resized = base.resize((target_w, target_h), resample=Image.LANCZOS)
    mask_resized = None
    if mask is not None:
        # keep mask binary; use NEAREST then threshold
        mask_resized = mask.resize((target_w, target_h), resample=Image.NEAREST)
    return base_resized, mask_resized, orig_size

def _prepare_mask_png_bytes_for_hf(mask_img: Image.Image, invert: bool = False) -> bytes:
    """
    Convert mask image to single-channel PNG bytes where white (255) indicates area to inpaint.
    If invert=True, flip the mask (useful if endpoint expects opposite convention or if user painted
    the object-to-keep and we want to invert so the outside becomes the inpaint area).
    """
    # convert to L and threshold
    m = mask_img.convert("L").point(lambda p: 255 if p > 127 else 0)
    if invert:
        m = m.point(lambda p: 255 - p)
    return image_to_bytes(m, fmt="PNG")

# diffusers local path (unchanged except returns RGB)
_diffusers_pipes: Dict[str, "DiffusionPipeline"] = {}

def call_local_diffusers_inpaint(
    model_id: str,
    base_image: Image.Image,
    mask_image: Optional[Image.Image],
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    mask_invert: bool = False,  # New: whether to invert mask for pipeline
) -> Image.Image:
    if DiffusionPipeline is None:
        raise RuntimeError("diffusers is not installed.")
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32

    pipe = _diffusers_pipes.get(model_id)
    if pipe is None:
        logger.info("Loading diffusers pipeline '%s'..., device=%s dtype=%s", model_id, device, torch_dtype)
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        try:
            pipe.to(device)
        except Exception:
            logger.warning("Failed to .to(%s) pipeline; continuing.", device)
        _diffusers_pipes[model_id] = pipe
        logger.info("Diffusers pipeline loaded.")
    # prepare mask for diffusers (L and threshold)
    mask_for_pipe = None
    if mask_image is not None:
        if mask_invert:
            # invert binary mask: white->black, black->white
            mask_for_pipe = mask_image.convert("L").point(lambda p: 0 if p > 127 else 255)
        else:
            mask_for_pipe = mask_image.convert("L").point(lambda p: 255 if p > 127 else 0)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        image=base_image,
        mask_image=mask_for_pipe,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
    )
    if hasattr(result, "images"):
        return result.images[0].convert("RGB")
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], Image.Image):
        return result[0].convert("RGB")
    raise RuntimeError("Unexpected diffusers pipeline return type.")

def call_hf_inference_inpaint(
    model_id: str,
    base_image: Image.Image,
    mask_image: Optional[Image.Image],
    prompt: str,
    negative_prompt: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    mask_invert: bool = False,  # New: pass True if mask marks the object to keep (so invert before sending)
) -> Image.Image:
    """
    Resizes input to model-friendly dims (multiples of 64), converts mask to a clear binary PNG,
    sends both base image and mask to the HF InferenceClient, then rescales output back to original size.

    mask_invert=True means the provided mask marks the object to KEEP; invert it so white==area_to_inpaint.
    """
    client = get_hf_client()

    # preserve original size and resize to multiples of 64
    base_to_send, mask_to_send, orig_size = _resize_image_and_mask_to_model(base_image, mask_image)

    base_bytes = image_to_bytes(base_to_send, fmt="PNG")
    mask_bytes = None
    if mask_to_send is not None:
        # standard: white = area to replace (inpaint). If mask_invert True, flip user mask.
        mask_bytes = _prepare_mask_png_bytes_for_hf(mask_to_send, invert=mask_invert)

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    if height:
        kwargs["height"] = int(height)
    if width:
        kwargs["width"] = int(width)

    # Many HF handlers accept 'mask' or 'mask_image' — include both keys (client will forward).
    if mask_bytes is not None:
        kwargs["mask"] = mask_bytes
        kwargs["mask_image"] = mask_bytes

    logger.debug("Calling HF Inference client.image_to_image model=%s kwargs_keys=%s", model_id, list(kwargs.keys()))
    try:
        result = client.image_to_image(base_bytes, model=model_id, **kwargs)

        # handle raw bytes -> PIL
        if isinstance(result, Image.Image):
            out = result.convert("RGB")
        elif isinstance(result, (bytes, bytearray)):
            out = Image.open(BytesIO(result)).convert("RGB")
        elif isinstance(result, dict):
            # attempt to extract base64 image
            if "images" in result and isinstance(result["images"], list) and len(result["images"]) > 0:
                img_b64 = result["images"][0]
                if isinstance(img_b64, str):
                    if img_b64.startswith("data:"):
                        img_b64 = img_b64.split(",", 1)[1]
                    out = Image.open(BytesIO(b64decode(img_b64))).convert("RGB")
                else:
                    raise RuntimeError("HF returned non-string image data.")
            elif "image" in result and isinstance(result["image"], str):
                v = result["image"]
                if v.startswith("http"):
                    out = download_image_from_url(v)
                else:
                    if v.startswith("data:"):
                        v = v.split(",", 1)[1]
                    out = Image.open(BytesIO(b64decode(v))).convert("RGB")
            elif "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                first = result["data"][0]
                if isinstance(first, dict):
                    for key in ("b64_json", "base64", "image"):
                        if key in first and isinstance(first[key], str):
                            img_b64 = first[key]
                            if img_b64.startswith("data:"):
                                img_b64 = img_b64.split(",", 1)[1]
                            out = Image.open(BytesIO(b64decode(img_b64))).convert("RGB")
                            break
                    else:
                        raise RuntimeError("Unrecognized 'data' format in HF response.")
                else:
                    raise RuntimeError("Unrecognized 'data' list entry in HF response.")
            else:
                # fallback scan
                for v in result.values():
                    if isinstance(v, str) and len(v) > 100:
                        try:
                            maybe_bytes = b64decode(v)
                            out = Image.open(BytesIO(maybe_bytes)).convert("RGB")
                            break
                        except Exception:
                            continue
                else:
                    raise RuntimeError(f"Unrecognized HF inference response format (dict keys: {list(result.keys())}).")
        else:
            raise RuntimeError(f"Unexpected HF inference return type: {type(result)}")

        # If we resized input earlier, rescale output back to original image size
        if out.size != orig_size:
            logger.info("Resizing model output %s back to original image size %s.", out.size, orig_size)
            out = out.resize(orig_size, resample=Image.LANCZOS)

        return out

    except InferenceTimeoutError as e:
        logger.exception("Inference timed out: %s", e)
        raise
    except Exception as e:
        logger.exception("HF Inference call failed: %s", e)
        raise

# core generate_image
def generate_image(
    upload_method: str,
    img_url: str,
    uploaded_img: Optional[Image.Image],
    mask_img: Optional[Image.Image],
    negative_prompt: str,
    hf_model: str,
    mask_is_keep: bool, # If True, the painted mask marks the object to keep; invert mask before inpainting
) -> Tuple[Optional[Image.Image], str]:
    logger.info("generate_image called (upload_method=%s model=%s mask_is_keep=%s)", upload_method, hf_model, mask_is_keep)

    if not HF_TOKEN:
        return None, "HF_TOKEN not set. Set HF_TOKEN in environment to use Hugging Face Inference API."

    try:
        if upload_method == "URL":
            if not img_url:
                return None, "", "Please provide an image URL."
            base_img = download_image_from_url(img_url)
        else:
            if uploaded_img is None:
                return None, "", "Please upload an image."
            base_img = uploaded_img.convert("RGB")
    except UnidentifiedImageError:
        return None, "", "Provided file/URL is not a valid image."
    except Exception as e:
        logger.exception("Failed to fetch/prepare base image: %s", e)
        return None, "", f"Failed to fetch/prepare base image: {e}"

    try:
        captioner = get_captioner()
        caption = captioner.caption(base_img)
        # a focused prompt instruction
        prompt = (
            f"Replace the background of the image to match this description: \"{caption}\". "
            "Keep the object inside the painted region exactly in place (do not modify or move it). "
            "Generate a coherent, semantically appropriate and artistically aligned background consistent with the scene and the description."
        )
    except Exception as e:
        logger.exception("Captioning failed: %s", e)
        return None, "", f"Captioning failed: {e}"

    if isinstance(mask_img, dict):
        mask_pil = mask_img.get("composite", None)
    else:
        mask_pil = mask_img

    try:
        mask_pil = _ensure_pil_image_or_none(mask_pil)
    except ValueError as e:
        logger.exception("Failed to normalize mask: %s", e)
        return None, "", f"Invalid mask image: {e}"

    try:
        model_id = hf_model or HF_INPAINT_MODEL
        # call HF inpainting (handles resizing internally and preserves final size)
        output_img = call_hf_inference_inpaint(
            model_id=model_id,
            base_image=base_img,
            mask_image=mask_pil,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            mask_invert=mask_is_keep,  # invert if mask marks the object to keep
        )
        return output_img, caption, "Success (HF Inference)"
    except Exception as hf_exc:
        logger.exception("HF inpainting failed: %s", hf_exc)
        # fallback to local diffusers if possible
        msg = str(hf_exc).lower()
        if "not found" in msg or "repositorynotfound" in msg or "no inference provider" in msg or "not accessible" in msg:
            try:
                fallback_model = "sd2-community/stable-diffusion-2-inpainting"
                output_img = call_local_diffusers_inpaint(
                    model_id=fallback_model,
                    base_image=base_img,
                    mask_image=mask_pil,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    mask_invert=mask_is_keep,
                )
                return output_img, caption, "Success (local diffusers fallback)"
            except Exception as diff_exc:
                logger.exception("Local diffusers fallback failed: %s", diff_exc)
                return None, caption, f"HF inpainting failed: {hf_exc} | Local diffusers fallback failed: {diff_exc}"
        return None, caption, f"HF inpainting failed: {hf_exc}"

# UI helpers (unchanged)
def invert_box(upload_method: str):
    if upload_method == "URL":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)

def build_ui():
    with gr.Blocks(title="Local caption + HF inpainting (preserve object, regenerate background)") as demo:
        gr.Markdown("### Caption-based Image Generation - Uses: Background generation, Object removal, Scene modification")
        gr.Markdown(
            "The app generates a caption locally and uses it as the prompt for an HF inpainting model. "
            "Draw a white mask with the sketch tool. Use the checkbox if your painted mask marks the object you WANT TO KEEP "
            "(white = preserved object): the system will invert the mask so the background outside the object is regenerated."
        )

        with gr.Row():
            upload_method = gr.Radio(choices=["URL", "Upload"], value="URL", label="Image input")
            img_url = gr.Textbox(label="Image URL")
            uploaded_img = gr.Image(type="pil", label="Upload Image", visible=False)
            upload_method.change(invert_box, inputs=upload_method, outputs=[img_url, uploaded_img])

        with gr.Row():
            mask_img = gr.ImageEditor(type="pil", label="Mask (use the editor — paint white where relevant)", visible=True)

        with gr.Row():
            # If checked: user's white-painted mask indicates the object to KEEP (so the code inverts the mask).
            mask_is_keep = gr.Checkbox(label="Keep masked object (invert mask): white = the object to preserve", value=True)

        with gr.Row():
            negative_prompt = gr.Textbox(
                label="Negative Prompt (optional)",
                value="disfigured, deformed, ugly, floating, blur, haze, poorly drawn",
            )

        with gr.Row():
            hf_model = gr.Textbox(label="Hugging Face inpainting model id", value=os.getenv("HF_INPAINT_MODEL", HF_INPAINT_MODEL))
            generate_btn = gr.Button("Generate (HF Inference)")

        caption_box = gr.Textbox(label="Generated Caption", interactive=False)

        status = gr.Textbox(label="Status", interactive=False)
        output_image = gr.Image(type="pil", label="Output Image")

        def _wrap_generate(upload_method, img_url, uploaded_img, mask_img, negative_prompt, hf_model, mask_is_keep):
            img, caption, msg = generate_image(upload_method, img_url, uploaded_img, mask_img, negative_prompt, hf_model, mask_is_keep)
            return img, caption, msg

        generate_btn.click(
            fn=_wrap_generate,
            inputs=[upload_method, img_url, uploaded_img, mask_img, negative_prompt, hf_model, mask_is_keep],
            outputs=[output_image, caption_box, status],
        )

    return demo

if __name__ == "__main__":
    logger.info("Starting Gradio app. Torch device: %s", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=6009, debug=True)