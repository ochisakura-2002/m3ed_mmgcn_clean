"""S2D training checkpoints contain only the model state, never run metadata."""

from collections.abc import Mapping


def require_tensor_state(state):
    import torch

    if (not isinstance(state, Mapping) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   for key, value in state.items())):
        raise ValueError("Expected a pure tensor state_dict; mixed checkpoints require explicit external conversion")
    return state


def save_model_weights(model, path):
    import torch

    torch.save(require_tensor_state(model.state_dict()), path)


def reload_model_weights(model, path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(require_tensor_state(state), strict=True)
