from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.multidag_cl.paper_reimplementation.attention import RelationAwareAttention
from models.multidag_cl.paper_reimplementation.dag_layer import DualGRUNodeUpdate


def test_attention_matches_hand_calculated_logits_softmax_and_relation_values() -> None:
    attention = RelationAwareAttention(1).double()
    with torch.no_grad():
        attention.score_linear.weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        attention.score_linear.bias.zero_()
        attention.W_same.weight.fill_(2.0)
        attention.W_different.weight.fill_(3.0)
    result = attention(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[[1.0], [3.0]]], dtype=torch.float64),
        torch.tensor([[1, 1]], dtype=torch.int64),
        torch.tensor([[1, 0]], dtype=torch.int64),
    )
    denominator = math.exp(3.0) + math.exp(5.0)
    expected_weights = torch.tensor(
        [[math.exp(3.0) / denominator, math.exp(5.0) / denominator]],
        dtype=torch.float64,
    )
    expected_message = expected_weights[0, 0] * 2.0 + expected_weights[0, 1] * 9.0
    torch.testing.assert_close(result.logits, torch.tensor([[3.0, 5.0]], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(result.weights, expected_weights, rtol=0, atol=1e-12)
    torch.testing.assert_close(result.message, expected_message.reshape(1, 1), rtol=0, atol=1e-12)


def test_attention_masks_nonpredecessors_and_handles_empty_rows_without_nan() -> None:
    torch.manual_seed(5)
    attention = RelationAwareAttention(2)
    result = attention(
        torch.tensor([[0.2, -0.1], [1.0, 2.0]]),
        torch.tensor(
            [
                [[1.0, 0.0], [4.0, 5.0], [0.0, 2.0]],
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            ]
        ),
        torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.int64),
        torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.int64),
    )
    assert result.weights[0, 1].item() == 0.0
    torch.testing.assert_close(result.weights[0].sum(), torch.tensor(1.0))
    assert torch.count_nonzero(result.weights[1]).item() == 0
    assert torch.count_nonzero(result.message[1]).item() == 0
    assert torch.isfinite(result.weights).all()
    assert torch.isfinite(result.message).all()


def test_attention_with_zero_predecessor_axis_returns_exact_empty_and_zero() -> None:
    attention = RelationAwareAttention(3)
    result = attention(
        torch.randn(2, 3),
        torch.empty(2, 0, 3),
        torch.empty(2, 0, dtype=torch.bool),
        torch.empty(2, 0, dtype=torch.bool),
    )
    assert result.logits.shape == (2, 0)
    assert result.weights.shape == (2, 0)
    assert torch.count_nonzero(result.message).item() == 0


def test_all_attention_and_relation_parameters_receive_finite_gradients() -> None:
    torch.manual_seed(7)
    attention = RelationAwareAttention(3)
    result = attention(
        torch.randn(1, 3),
        torch.randn(1, 3, 3),
        torch.tensor([[1, 1, 1]], dtype=torch.bool),
        torch.tensor([[1, 0, 1]], dtype=torch.bool),
    )
    result.message.square().sum().backward()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


class RecordingCell(nn.Module):
    def __init__(self, hidden_dim: int, offset: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.offset = offset
        self.calls = []

    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        self.calls.append((input_tensor.clone(), hidden.clone()))
        return input_tensor + 10.0 * hidden + self.offset


def test_dual_gru_recording_cells_prove_input_hidden_order_and_sum() -> None:
    update = DualGRUNodeUpdate(2)
    node = RecordingCell(2, 1.0)
    context = RecordingCell(2, 2.0)
    update.node_gru = node
    update.context_gru = context
    previous = torch.tensor([[1.0, 2.0]])
    message = torch.tensor([[3.0, 4.0]])
    result = update(previous, message)
    assert torch.equal(node.calls[0][0], previous)
    assert torch.equal(node.calls[0][1], message)
    assert torch.equal(context.calls[0][0], message)
    assert torch.equal(context.calls[0][1], previous)
    expected_node = previous + 10.0 * message + 1.0
    expected_context = message + 10.0 * previous + 2.0
    torch.testing.assert_close(result.node_state, expected_node)
    torch.testing.assert_close(result.context_state, expected_context)
    torch.testing.assert_close(result.state, expected_node + expected_context)
    assert result.state.shape == previous.shape


def test_dual_gru_first_node_zero_message_uses_the_same_formula() -> None:
    update = DualGRUNodeUpdate(2)
    node = RecordingCell(2, 0.0)
    context = RecordingCell(2, 0.0)
    update.node_gru = node
    update.context_gru = context
    previous = torch.tensor([[2.0, -1.0]])
    zero_message = torch.zeros_like(previous)
    update(previous, zero_message)
    assert torch.equal(node.calls[0][1], zero_message)
    assert torch.equal(context.calls[0][0], zero_message)


def test_real_dual_gru_parameters_have_finite_gradients() -> None:
    torch.manual_seed(13)
    update = DualGRUNodeUpdate(3)
    result = update(torch.randn(2, 3), torch.randn(2, 3))
    result.state.square().mean().backward()
    for name, parameter in update.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
