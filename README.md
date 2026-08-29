# MLOps PyTorch Pipeline

This project implements a PyTorch-based CIFAR-10 training pipeline, checkpoint management, FastAPI model serving, and automated model tests.

##Architecture

![Structured training logs](docs/images/architecture-diagram.png)

## Model Training

Run the training pipeline:

```bash
python -m src.train
```

The training process produces structured JSON logs for every epoch.

### Structured Training Logs

![Structured training logs](docs/images/training-json-logs.png)

### Checkpoint Verification

The best-performing model checkpoint is saved as:

```text
checkpoints/classifier_v1.pt
```

The checkpoint was successfully reloaded and verified.

![Checkpoint verification](docs/images/checkpoint-verification.png)

## Model Serving

Start the FastAPI application:

```bash
MODEL_PATH=checkpoints/classifier_v1.pt \
uvicorn src.serve:app --host 127.0.0.1 --port 8080
```

Keep this terminal running while testing the API.

### Health Endpoint

Open another terminal and run:

```bash
curl -i http://127.0.0.1:8080/health
```

![API health check](docs/images/api-health.png)

### Prediction Endpoint

Run:

```bash
curl -X POST \
  http://127.0.0.1:8080/predict \
  -F "image=@test_image.png"
```

The response contains the predicted class, confidence, and probabilities for all 10 CIFAR-10 classes.

![API prediction result](docs/images/api-prediction.png)

## Testing

Run the automated tests:

```bash
python -m pytest -v
```

All three automated PyTorch model tests should pass.

![PyTorch automated tests](docs/images/pytorch-tests.png)
