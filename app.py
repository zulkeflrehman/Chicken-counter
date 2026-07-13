"""
Poultry Farm Chicken Counter
-----------------------------
A YOLO-based object detection app that counts every visible chicken
in an uploaded farm photo (PNG, JPG, JPEG, WEBP, BMP - any format PIL supports).

Deploy target: HuggingFace Spaces (Gradio SDK)
"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# --- Robust class detection ---------------------------------------------
# Instead of hardcoding a class name (which broke once already when the
# Roboflow project was renamed), we auto-detect the "whole chicken" class
# at startup: any class whose name contains "chicken" but does NOT refer
# to a body part (head, comb, eye, leg, wing, etc).
EXCLUDE_KEYWORDS = ["head", "comb", "eye", "leg", "wing", "beak", "feather"]

def find_whole_chicken_class_id(names: dict) -> int:
    candidates = []
    for idx, name in names.items():
        lname = name.lower()
        if "chicken" in lname or "hen" in lname or "broiler" in lname or "poultry" in lname:
            if not any(kw in lname for kw in EXCLUDE_KEYWORDS):
                candidates.append((idx, name))
    if not candidates:
        # Fall back to class 0 if nothing matches (single-class models)
        return 0
    # Prefer the shortest matching name (usually the "clean" whole-object class)
    candidates.sort(key=lambda x: len(x[1]))
    return candidates[0][0]

CHICKEN_CLASS_ID = find_whole_chicken_class_id(model.names)
CHICKEN_CLASS_NAME = model.names[CHICKEN_CLASS_ID]
print(f"Detected classes: {model.names}")
print(f"Counting class: '{CHICKEN_CLASS_NAME}' (id={CHICKEN_CLASS_ID})")


def count_chickens(image: Image.Image, confidence: float):
    if image is None:
        return None, "Please upload an image first."

    try:
        results = model(image, conf=confidence, verbose=False)
    except Exception as e:
        return None, f"Could not process this image: {e}"

    boxes = results[0].boxes
    chicken_indices = [i for i, b in enumerate(boxes) if int(b.cls[0]) == CHICKEN_CLASS_ID]
    count = len(chicken_indices)

    # Draw only the whole-chicken boxes (skip head/comb boxes visually)
    result_copy = results[0]
    if len(chicken_indices) < len(boxes):
        keep_mask = np.array([int(b.cls[0]) == CHICKEN_CLASS_ID for b in boxes])
        result_copy.boxes = boxes[keep_mask]

    annotated_bgr = result_copy.plot(line_width=2, font_size=12)
    annotated_rgb = annotated_bgr[..., ::-1]
    output_image = Image.fromarray(annotated_rgb)

    if count == 0:
        message = "No chickens detected. Try a clearer or closer image, or lower the confidence threshold."
    elif count == 1:
        message = "We detected 1 chicken in the poultry farm."
    else:
        message = f"We detected {count} chickens in the poultry farm."

    if chicken_indices:
        avg_conf = float(np.mean([boxes[i].conf[0].item() for i in chicken_indices]))
        message += f"\n\nAverage detection confidence: {avg_conf:.0%}"

    return output_image, message


with gr.Blocks(title="Poultry Farm Chicken Counter", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐔 Poultry Farm Chicken Counter
        Upload a photo of a poultry farm and the model will detect and count every visible chicken.
        Supports PNG, JPG, JPEG, WEBP, and BMP images.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Farm Image")
            confidence_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Detection confidence threshold",
                info="Lower = catches more chickens but may include false positives. Higher = stricter, may miss some."
            )
            submit_btn = gr.Button("Count Chickens", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="Detection Result")
            text_output = gr.Textbox(label="Result", lines=3)

    gr.Markdown(
        """
        ---
        *This model was trained on a custom poultry-farm dataset. Accuracy may vary with
        image quality, lighting, and chicken density. For production use, verify counts
        against a manual sample before relying on results.*
        """
    )

    submit_btn.click(
        fn=count_chickens,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, text_output]
    )
    image_input.change(
        fn=count_chickens,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    demo.launch()
