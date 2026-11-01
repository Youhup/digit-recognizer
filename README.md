# Handwritten Digit Recognizer

Draw a digit with your mouse and a small PyTorch CNN, trained on MNIST, predicts it in a Gradio web app.
the app runs locally (see [Run it locally](#run-it-locally)).

<!-- Replace this line with a screenshot or GIF of the app -->
*Screenshot coming soon.*

## Model

A small CNN with 207,018 parameters. It takes a `[batch, 1, 28, 28]` image and outputs 10 logits, one per digit.

| Part | Layers | Output shape |
|---|---|---|
| Block 1 | Conv2d (16 filters) → BatchNorm2d → ReLU → MaxPool2d | `[batch, 16, 14, 14]` |
| Block 2 | Conv2d (32 filters) → BatchNorm2d → ReLU → MaxPool2d | `[batch, 32, 7, 7]` |
| Classifier | Flatten → Linear (128) → ReLU → Dropout (0.25) → Linear (10) | `[batch, 10]` |

**Final test accuracy: 99.08%** on the 10,000 MNIST test images.

It was trained for 5 epochs with Adam (learning rate 1e-3), batch size 64 and `CrossEntropyLoss`. Training images get small random rotations (up to ±10°) and shifts (up to 10%). Training takes about 4 minutes on a CPU.

## Run it locally

Install the dependencies (Python 3):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally, retrain the model. The trained weights are already included as `mnist_cnn.pth`, and retraining overwrites them:

```bash
python train.py
```

Launch the app, then open http://127.0.0.1:7860:

```bash
python app.py
```

Click the **Brush** icon next to the canvas before drawing, and again after clearing it. The prediction updates each time you finish a stroke, and the app also shows the 28×28 image the model receives.

## Project structure

| File | Purpose |
|---|---|
| `model.py` | The CNN class |
| `train.py` | Trains on MNIST, evaluates and saves the weights |
| `preprocess.py` | Converts a drawing into an MNIST-like tensor |
| `app.py` | The Gradio interface |
| `mnist_cnn.pth` | The trained weights |

## What I learned

**High test accuracy doesn't guarantee good results on real input.** The model scored about 99% on MNIST, but that only applies to images that look like MNIST. A drawing on a canvas doesn't: it's dark on light, can be any size, and can be anywhere on the canvas. Testing the app with scripted mouse drawings before preprocessing was complete, a digit drawn large in the center was read correctly, while the same digit drawn small in a corner was misread or barely recognized.

**Preprocessing closed that gap.** Each drawing now goes through the same steps:

1. Flatten it onto a white background and convert it to grayscale
2. Invert the colors, because MNIST digits are white on black
3. Stop if the canvas is empty, because a model always gives an answer, even for a blank image
4. Crop to the drawn pixels
5. Resize so the longest side is 20 px, keeping the proportions
6. Center it on a 28×28 black image by its center of mass, the way MNIST was prepared
7. Scale pixel values to 0–1, the same as in training

To check this, I placed real MNIST test digits on a canvas at different sizes and positions and ran them through these steps. Accuracy stayed at about 98.9%, whatever the size or position.

**The steps only work together.** Adding them one at a time sometimes made results worse. Cropping without keeping the proportions stretched a straight "1" into an almost solid white square, which the model read as a 0 with 99% confidence.

**Data augmentation prepares the model for imperfect input.** Small random rotations and shifts during training show the model digits that aren't perfectly straight or centered, as hand-drawn digits often aren't.

**Details outside the model matter too.** Every digit is scaled to the same size, so a small drawing's strokes get enlarged along with it. A thick brush turned small digits into blobs, and choosing a thinner one improved accuracy on very small simulated drawings from 43.5% to 96%.
