# VLM Design Choices

This document explains the architectural decision to use Microsoft's **Florence-2** model over massive conversational VLMs like LLaVA or Qwen-VL for the Aegis pixel redaction pipeline.

## Model Comparison

| Feature / Metric | Florence-2 (Base/Large) | LLaVA (1.5 / 1.6) | Qwen-VL (Chat) |
|------------------|-------------------------|-------------------|----------------|
| **Primary Design Focus** | Computer Vision, Visual Grounding, Region Captioning | Conversational AI, Visual Q&A, Chatbots | Conversational AI, Visual Q&A |
| **Parameter Count** | ~230M (Base) / ~770M (Large) | 7B to 13B | 7B |
| **Hardware Requirements** | CPU or Cheap GPU (T4, RTX 3060) | Datacenter GPU (A100, V100, 24GB+ VRAM) | Datacenter GPU (A100, 24GB+ VRAM) |
| **Inference Speed** | Very Fast (ideal for high-throughput batching) | Slow (generative auto-regressive overhead) | Slow |
| **Bounding Box Output** | Native structured coordinate arrays (`[x1, y1, x2, y2]`) | Requires complex prompting; outputs free-text requiring regex | Requires complex prompting; hallucinates boxes |
| **Zero-Shot OCR Grounding** | Excellent. Pre-trained on text-rich imagery to link text to pixels. | Poor. Often reads text but struggles to map it to pixel locations accurately. | Moderate. Better than LLaVA but still conversational. |

## Detailed Justification

### 1. Precision Bounding Boxes
For a medical de-identification pipeline, redaction requires extremely precise spatial coordinates. Florence-2 was explicitly trained using the `<CAPTION_TO_PHRASE_GROUNDING>` task, which natively accepts a text phrase (e.g., "Patient Name") and directly returns a structured JSON payload of exact bounding box coordinates. Conversational VLMs typically return prose (e.g., "The patient's name is located at the top left of the screen"), which cannot be reliably parsed into a redaction mask.

### 2. Speed and Hardware Constraints
Aegis must be capable of processing thousands of DICOMs or Video frames securely at the edge. A 7-billion parameter model is too large to run efficiently on standard hospital hardware or edge ultrasound carts. Florence-2's extremely small footprint (~770M parameters) allows it to run at high speeds even on consumer-grade hardware.

### 3. "Zero-Trust" Edge Deployment
Because Florence-2 is small enough to run locally without major infrastructure, it ensures that Protected Health Information (PHI) never leaves the hospital's internal network. This makes Aegis significantly more secure than relying on proprietary cloud APIs like Google Cloud DLP or AWS Comprehend.

## Task Prompts

Florence-2 is a multi-task model that requires specific, literal task strings (not placeholders or conversational English) to determine what visual task to perform. The following prompts are commonly used:

- `'<CAPTION_TO_PHRASE_GROUNDING>'`: This is what we use as the default. It tells the model to look at the `text_input` list and find bounding boxes exclusively for those items.
- `'<OCR_WITH_REGION>'`: If you set this, Florence-2 will just blindly OCR all text on the screen and return bounding boxes, ignoring your `text_input`.
- `'<DENSE_REGION_CAPTION>'`: Tells the model to draw boxes around generic objects it recognizes and label them.
