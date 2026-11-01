import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

# The same scaling as training: pixels go from 0-255 to 0-1
to_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


def preprocess(drawing):
    """Turn a PIL image of the drawing into a tensor of shape [1, 1, 28, 28] for the model.

    Returns the tensor and the 28x28 image, so the app can show what the model sees.
    Returns (None, None) if nothing is drawn.
    """
    # Step 1: flatten onto a white background (the canvas may be transparent), then make it grayscale
    drawing = drawing.convert("RGBA")
    background = Image.new("RGBA", drawing.size, "white")
    gray = Image.alpha_composite(background, drawing).convert("L")

    # Step 2: invert the colors, because MNIST digits are white on a black background
    inverted = ImageOps.invert(gray)

    # Step 3: if the canvas is empty, stop here (getbbox finds no non-black pixels)
    bbox = inverted.getbbox()
    if bbox is None:
        return None, None

    # Step 4: crop to the bounding box of the drawn pixels
    cropped = inverted.crop(bbox)

    # Step 5: resize so the longest side is 20 px, keeping the aspect ratio
    width, height = cropped.size
    scale = 20 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))  # at least 1 px per side
    resized = cropped.resize(new_size, Image.LANCZOS)

    # A very faint drawing can fade to nothing when shrunk, so treat that as empty too
    if resized.getbbox() is None:
        return None, None

    # Step 6: find the center of mass (the average pixel position, weighted by brightness)
    pixels = np.array(resized, dtype=np.float32)
    rows, cols = np.indices(pixels.shape)
    center_x = (cols * pixels).sum() / pixels.sum()
    center_y = (rows * pixels).sum() / pixels.sum()

    # Paste so the center of mass lands on the image center (13.5), keeping the whole digit inside
    left = min(max(round(13.5 - center_x), 0), 28 - resized.width)
    top = min(max(round(13.5 - center_y), 0), 28 - resized.height)
    small = Image.new("L", (28, 28), 0)
    small.paste(resized, (left, top))

    # Step 7: scale to 0-1, then add the batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
    tensor = to_tensor(small).unsqueeze(0)

    return tensor, small
