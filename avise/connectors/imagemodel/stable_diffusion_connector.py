"""Stable Diffusion Connector for AVISE.

Connects the AVISE framework to a locally running Stable Diffusion
model via the HuggingFace diffusers library.
"""

import os
import logging
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional
from io import BytesIO

logger = logging.getLogger(__name__)


class StableDiffusionConnector:
    """Connector for Stable Diffusion image generation models.

    Runs locally using the HuggingFace diffusers library.
    Saves generated images to disk and returns image paths + base64.

    Args:
        model_id: HuggingFace model ID (e.g. 'stabilityai/stable-diffusion-2-1')
        image_size: Output image size in pixels (default 512)
        num_inference_steps: Number of denoising steps (default 30)
        guidance_scale: CFG guidance scale (default 7.5)
        output_dir: Directory to save generated images
        device: Device to run on ('cuda', 'cpu', 'mps')
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        image_size: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        output_dir: str = "avise-reports/images",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None

        # Auto-detect device
        if device:
            self.device = device
        else:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"StableDiffusionConnector initialized on device: {self.device}")

    def load_model(self):
        """Load the Stable Diffusion pipeline. Called once before execution."""
        if self.pipe is not None:
            return

        try:
            import torch
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

            logger.info(f"Loading model: {self.model_id}")

            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # We handle safety evaluation ourselves
                requires_safety_checker=False,
            )

            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe = self.pipe.to(self.device)

            # Memory optimizations for T4
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except Exception:
                    logger.info("xformers not available, skipping")

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, negative_prompt: str = "") -> dict:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to guide away from certain content

        Returns:
            Dict containing:
                - image_path: Path to saved image file
                - image_base64: Base64 encoded image string
                - prompt: Original prompt used
                - model_id: Model used for generation
                - was_filtered: Whether SD's internal safety filter triggered
                - error: Error message if generation failed
        """
        if self.pipe is None:
            self.load_model()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"generated_{timestamp}.png"
        image_path = self.output_dir / image_filename

        try:
            import torch

            logger.info(f"Generating image for prompt: {prompt[:80]}...")

            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=self.image_size,
                    width=self.image_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                )

            image = result.images[0]

            # Check if SD's internal filter returned a black image
            # SD returns a black image when its NSFW filter triggers
            was_filtered = self._is_black_image(image)
            if was_filtered:
                logger.warning(f"SD internal filter triggered for: {prompt[:60]}...")

            # Save image
            image.save(image_path)
            logger.info(f"Image saved to: {image_path}")

            # Encode to base64 for evaluator
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {
                "image_path": str(image_path),
                "image_base64": image_base64,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_id": self.model_id,
                "was_filtered": was_filtered,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "image_path": None,
                "image_base64": None,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_id": self.model_id,
                "was_filtered": False,
                "error": str(e),
            }

    def _is_black_image(self, image, threshold: float = 0.01) -> bool:
        """Check if image is blank/black — SD's way of refusing a prompt.

        Args:
            image: PIL Image
            threshold: Mean pixel value below which image is considered blank

        Returns:
            True if image appears to be blank/black (filtered)
        """
        import numpy as np
        img_array = np.array(image).astype(float) / 255.0
        return float(img_array.mean()) < threshold

    def unload_model(self):
        """Free GPU memory after evaluation is complete."""
        if self.pipe is not None:
            import torch
            del self.pipe
            self.pipe = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Model unloaded and GPU memory cleared")