import pytest
import torch
from src.model import get_model


def test_model_output_shape():
    model = get_model()
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 32, 32))

    assert outputs.shape == (2, 10)
    assert torch.isfinite(outputs).all()


def test_model_backward():
    model = get_model()
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 9])

    loss = torch.nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_unsupported_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model(architecture="unknown")
