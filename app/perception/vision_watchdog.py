import asyncio
import base64
import binascii
import io
import logging
import threading
import time

from PIL import Image

logger = logging.getLogger("vision_watchdog")


class VisionWatchdog:
    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str = "auto",
        max_new_tokens: int = 8,
        timeout_seconds: float | None = None,
    ):
        self.model_id = model
        self.device_config = device
        self.torch_dtype_config = torch_dtype
        self.attn_implementation_config = attn_implementation
        self.max_new_tokens = max_new_tokens
        # Kept for config compatibility. Generation runs in-process, so this is not used.
        self.timeout_seconds = timeout_seconds

        self._disabled_until = 0.0
        self._last_error_signature: str | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._processor = None
        self._model = None
        self._device = None

    async def evaluate_screen(self, base64_image: str) -> bool:
        prompt = (
        "Examine this screenshot. Is there a highly noticeable new event that demands "
        "attention? Examples include: a prominent notification, an incoming message or call, "
        "an error popup, a completed loading screen, a finished download, or a major "
        "visual change (like a video ending or a checkout confirming). "
        "Answer exactly YES or NO."
    )
        return await self._evaluate(prompt, base64_image, source="screen")

    async def evaluate_webcam(self, base64_image: str) -> bool:
        prompt = (
            "Examine this webcam frame. Did the user just sit down, raise their hand, "
            "look directly at the camera with a confused expression, or hold an object "
            "up to the lens? Answer exactly YES or NO."
        )
        return await self._evaluate(prompt, base64_image, source="webcam")

    async def _evaluate(self, prompt: str, base64_image: str, source: str) -> bool:
        if not base64_image:
            return False
        if time.monotonic() < self._disabled_until:
            return False

        loop = asyncio.get_running_loop()
        try:
            response_text = await loop.run_in_executor(
                None,
                self._generate,
                prompt,
                base64_image,
            )
        except Exception as exc:
            self._log_failure_once(source, exc)
            self._disabled_until = time.monotonic() + 15.0
            return False

        normalized = response_text.strip().upper()
        logger.debug("Vision watchdog %s response: %r", source, response_text)
        return normalized.startswith("YES")

    def _generate(self, prompt: str, base64_image: str) -> str:
        image = self._decode_image(base64_image)
        processor, model, device = self._ensure_model_loaded()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)

        with self._generate_lock:
            import torch

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        prompt_token_count = inputs["input_ids"].shape[1]
        generated_only = generated_ids[:, prompt_token_count:]
        generated_texts = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )
        return generated_texts[0] if generated_texts else ""

    def _ensure_model_loaded(self):
        if self._processor is not None and self._model is not None and self._device is not None:
            return self._processor, self._model, self._device

        with self._load_lock:
            if self._processor is not None and self._model is not None and self._device is not None:
                return self._processor, self._model, self._device

            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            device = self._resolve_device(torch)
            torch_dtype = self._resolve_torch_dtype(torch, device)
            attn_implementation = self._resolve_attn_implementation(device)

            logger.info(
                "Loading SmolVLM watchdog model %s on %s (dtype=%s, attn=%s)",
                self.model_id,
                device,
                torch_dtype,
                attn_implementation,
            )
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                # _attn_implementation=attn_implementation,
            ).to(device)
            model.eval()

            self._processor = processor
            self._model = model
            self._device = device
            return processor, model, device

    def _resolve_device(self, torch):
        if self.device_config and self.device_config != "auto":
            return self.device_config
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_torch_dtype(self, torch, device: str):
        dtype = (self.torch_dtype_config or "auto").lower()
        if dtype == "auto":
            return torch.bfloat16 if device == "cuda" else torch.float32
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported vision_watchdog.torch_dtype: {self.torch_dtype_config}")

    def _resolve_attn_implementation(self, device: str) -> str:
        if self.attn_implementation_config and self.attn_implementation_config != "auto":
            return self.attn_implementation_config
        return "flash_attention_2" if device == "cuda" else "eager"

    def _decode_image(self, base64_image: str) -> Image.Image:
        normalized = base64_image.strip()
        if normalized.startswith("data:") and "," in normalized:
            _, normalized = normalized.split(",", 1)

        try:
            image_bytes = base64.b64decode(normalized, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Vision watchdog image data is not valid base64") from exc

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    def _log_failure_once(self, source: str, exc: Exception) -> None:
        signature = f"{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature:
            return

        self._last_error_signature = signature
        logger.warning(
            "Vision watchdog evaluation failed for %s frame using %s: %s",
            source,
            self.model_id,
            exc,
        )
