import gradio as gr
from model.unet import load_model
import numpy as np

model = load_model("model.pth")

def predict(file):
    # Replace with real inference
    return np.random.randint(0, 3, (128, 128))

gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload MRI"),
    outputs="image",
    title="Brain Tumor Segmentation"
).launch()