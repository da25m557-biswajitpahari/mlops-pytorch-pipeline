import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import get_model


preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv(
        "MODEL_PATH", "checkpoints/classifier_v1.pt"
    )
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True
    )

    model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    app.state.model = model
    app.state.class_names = checkpoint["class_names"]

    yield

    app.state.model = None


app = FastAPI(title="CIFAR-10 Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read at most 5 MB plus one byte to detect oversized uploads.
    contents = image.file.read(5 * 1024 * 1024 + 1)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")

    try:
        with Image.open(io.BytesIO(contents)) as uploaded:
            inputs = preprocess(uploaded.convert("RGB")).unsqueeze(0)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid or unsupported image"
        ) from exc

    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]

    class_names = app.state.class_names
    predicted_index = int(probabilities.argmax().item())

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(
                class_names, probabilities.tolist()
            )
        },
    }
