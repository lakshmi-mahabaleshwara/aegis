"""
MONAI Aegis VLM Engine — VLMClassifier

Uses Vision-Language Models (e.g., Florence-2) for end-to-end PHI detection
and bounding box generation based on image context.
"""
import torch
import numpy as np
import logging
from PIL import Image
from typing import Dict, Any, List, Tuple

from monai_aegis.transforms.exceptions import PixelRedactionError

logger = logging.getLogger(__name__)


class VLMClassifier:
    """
    Wrapper for Vision-Language Models (like Microsoft's Florence-2) to detect PHI directly.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        vlm_cfg = config.get('vlm', {})
        self.model_name = vlm_cfg.get('model_name', 'microsoft/Florence-2-large')
        self.task_prompt = vlm_cfg.get('task_prompt', '<CAPTION_TO_PHRASE_GROUNDING>')
        self.text_input = vlm_cfg.get('text_input', 'Patient Name, Medical Record Number, Patient ID, Dates, Hospital Name, Doctor Name, Date of Birth, Age, Phone Number')
        
        # Resolve device
        device_str = vlm_cfg.get('device', 'cpu')
        if device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning("VLM: CUDA requested but not available. Falling back to CPU.")
            device_str = 'cpu'
        elif device_str == 'mps' and not torch.backends.mps.is_available():
            logger.warning("VLM: MPS requested but not available. Falling back to CPU.")
            device_str = 'cpu'
            
        self.device = torch.device(device_str)
        self.processor = None
        self.model = None

    def _load_model(self):
        """Lazily load the VLM processor and model."""
        if self.model is not None and self.processor is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            logger.info(f"Loading VLM engine: {self.model_name} on {self.device}...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
            self.model.eval()
            logger.info("VLM engine loaded successfully.")
        except ImportError as e:
            raise PixelRedactionError(
                "Missing dependencies for VLM. Please install transformers, timm, and einops.",
                transform="VLMClassifier._load_model"
            ) from e
        except Exception as e:
            raise PixelRedactionError(
                f"Failed to load VLM model {self.model_name}: {e}",
                transform="VLMClassifier._load_model"
            ) from e

    def detect_phi_boxes(self, pixel_array: np.ndarray) -> Tuple[List[List[List[int]]], Dict[str, int]]:
        """
        Detect PHI regions in the given image array.
        
        Args:
            pixel_array: RGB image array of shape (H, W, 3) and dtype uint8.
            
        Returns:
            Tuple of (bboxes, stats) where bboxes matches EasyOCR format:
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...]
        """
        self._load_model()
        
        if pixel_array.ndim != 3 or pixel_array.shape[-1] != 3:
            raise ValueError(f"VLM requires (H, W, 3) RGB image, got shape {pixel_array.shape}")

        pil_image = Image.fromarray(pixel_array)
        
        # Use Open Vocabulary Detection / Phrase Grounding for PHI
        task_prompt = self.task_prompt
        # List of things we want the VLM to redact
        text_input = self.text_input
        prompt = task_prompt + text_input
        
        try:
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
            
            # Use float16 precision if on CUDA
            if self.device.type == 'cuda':
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(pil_image.width, pil_image.height)
            )
            
            # Format: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': [...]}}
            result = parsed_answer.get(task_prompt, {})
            bboxes_flat = result.get('bboxes', [])
            labels = result.get('labels', [])
            
            bboxes = []
            for bbox in bboxes_flat:
                x1, y1, x2, y2 = bbox
                # Convert to EasyOCR format (4 corners)
                bboxes.append([
                    [int(x1), int(y1)],
                    [int(x2), int(y1)],
                    [int(x2), int(y2)],
                    [int(x1), int(y2)]
                ])
                
            stats = {
                'total_detections': len(bboxes),
                'vlm_classified_count': len(bboxes),
                'redacted_count': len(bboxes)
            }
            
            return bboxes, stats
            
        except Exception as e:
            raise PixelRedactionError(
                f"VLM inference failed: {e}",
                transform="VLMClassifier.detect_phi_boxes"
            ) from e

