import gradio as gr
import torch
from PIL import Image

from model import DigitCNN
from preprocess import preprocess

# Use the GPU if there is one, otherwise the CPU
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

# Load the trained model once at startup, so every prediction is fast
model = DigitCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))
model.eval()  # turns off Dropout and makes BatchNorm use its saved statistics


def predict(drawing):
    """Return the probability of each digit and an enlarged view of what the model sees."""
    composite = drawing["composite"] if drawing else None  # the drawing as one flat image
    if composite is None:
        return None, None

    tensor, small = preprocess(composite)
    if tensor is None:  # the canvas is empty, so there is nothing to predict
        return None, None

    with torch.no_grad():
        probabilities = model(tensor.to(device)).softmax(dim=1)[0]  # turns logits into probabilities

    scores = {str(digit): probability.item() for digit, probability in enumerate(probabilities)}
    return scores, small.resize((280, 280), Image.NEAREST)  # NEAREST keeps the pixels sharp


with gr.Blocks(title="Digit Recognizer") as demo:
    gr.Markdown("# Digit Recognizer\nDraw a digit from 0 to 9.")

    with gr.Row():
        sketchpad = gr.Sketchpad(
            label="Draw here",
            type="pil",  # gives us PIL images, which is what preprocess expects
            canvas_size=(400, 400),
            fixed_canvas=True,
            brush=gr.Brush(default_size=10, colors=["#000000"], color_mode="fixed"),  # thick black brush
            layers=False,
            transforms=(),
        )
        with gr.Column():
            prediction = gr.Label(label="Prediction", num_top_classes=3)
            gr.Markdown("**What the model sees (28×28)**")  # above the image, so it can't cover the digit
            model_view = gr.Image(show_label=False, buttons=[], format="png")

    # Predict whenever the drawing changes (Sketchpad reports this when a stroke ends);
    # always_last skips outdated updates
    sketchpad.change(
        predict,
        inputs=sketchpad,
        outputs=[prediction, model_view],
        show_progress="hidden",
        trigger_mode="always_last",
    )

if __name__ == "__main__":
    demo.launch()
