"""Source-equation checks and causal boundaries, independent of real IEMOCAP."""

import pytest
import torch

from models.multidag_cl.graph_core_variants.coop import CooperativeAttention, CoopCore
from models.multidag_cl.graph_core_variants.dcn import DCNCore, causal_shift_factors
from models.multidag_cl.graph_core_variants.track import TrackCore
from models.multidag_cl.paper_reimplementation.attention import RelationAwareAttention
from models.multidag_cl.paper_reimplementation.contracts import SpeakerRelations
from models.multidag_cl.paper_reimplementation.graph import CausalPredecessorBuilder, SpeakerRelationBuilder

CORES = (TrackCore, CoopCore, DCNCore)


def graph_inputs(hidden=8):
    torch.manual_seed(100)
    speakers = torch.tensor([[4, 9, 4, 9, 4], [9, 9, 4, -1, -1]])
    mask = speakers >= 0
    h0 = torch.randn(2, 5, hidden) * mask[..., None]
    adjacency = CausalPredecessorBuilder().build(speakers, mask)
    relations = SpeakerRelationBuilder.build(speakers, mask)
    return h0, adjacency, relations, mask


def prefix_inputs(inputs, length=3):
    h, a, r, m = inputs
    return (h[:1, :length], a[:1, :length, :length],
            SpeakerRelations(r.same_speaker[:1, :length, :length],
                             r.different_speaker[:1, :length, :length]), m[:1, :length])


@pytest.mark.parametrize("core_type", CORES)
def test_four_depths_padding_determinism_and_finite_backward(core_type):
    inputs = graph_inputs()
    inputs[0].requires_grad_()
    core = core_type(8)
    states = core(*inputs)
    assert len(states) == 4
    again = core(*inputs)
    for h, repeat in zip(states, again):
        assert h.shape == (2, 5, 8) and torch.isfinite(h).all()
        assert torch.count_nonzero(h[~inputs[-1]]) == 0
        torch.testing.assert_close(h, repeat, rtol=0, atol=0)
    sum(h.square().sum() for h in states).backward()
    assert torch.isfinite(inputs[0].grad).all()
    assert torch.count_nonzero(inputs[0].grad[~inputs[-1]]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in core.parameters())


@pytest.mark.parametrize("core_type", CORES)
def test_u4_u5_perturbation_cannot_change_u1_u2_u3(core_type):
    inputs = graph_inputs()
    core = core_type(8).eval()
    changed = inputs[0].clone()
    changed[0, 3:] = 1000 * torch.randn_like(changed[0, 3:])
    before, after = core(*inputs), core(changed, *inputs[1:])
    for left, right in zip(before, after):
        torch.testing.assert_close(left[0, :3], right[0, :3], rtol=0, atol=0)
    assert any(not torch.equal(left[0, 3:], right[0, 3:]) for left, right in zip(before, after))


@pytest.mark.parametrize("core_type", CORES)
def test_prefix_extension_and_future_graph_metadata_invariance(core_type):
    inputs = graph_inputs()
    core = core_type(8).eval()
    full, short = core(*inputs), core(*prefix_inputs(inputs))
    for left, right in zip(full, short):
        torch.testing.assert_close(left[:1, :3], right, rtol=1e-6, atol=1e-6)
    speakers = torch.tensor([[4, 9, 4, 4, 9], [9, 9, 4, -1, -1]])
    future_adj = inputs[1].clone()
    future_adj[0, 3:] = False
    future_rel = SpeakerRelationBuilder.build(speakers, inputs[-1])
    future = core(inputs[0], future_adj, future_rel, inputs[-1])
    for left, right in zip(full, future):
        torch.testing.assert_close(left[0, :3], right[0, :3], rtol=0, atol=0)


@pytest.mark.parametrize("core_type", CORES)
def test_future_feature_jacobian_is_exactly_zero(core_type):
    inputs = graph_inputs()
    inputs[0].requires_grad_()
    core = core_type(8)
    sum(state[0, :3].sum() for state in core(*inputs)).backward()
    assert torch.count_nonzero(inputs[0].grad[0, 3:]) == 0
    assert torch.count_nonzero(inputs[0].grad[0, :3]) > 0


@pytest.mark.parametrize("core_type", CORES)
@pytest.mark.parametrize("case", ["future", "self", "padded_edge", "overlap", "incomplete", "nonprefix", "nan", "padding"])
def test_invalid_graphs_are_rejected(core_type, case):
    h, a, r, m = graph_inputs()
    if case == "future":
        a[0, 0, 4] = True
    elif case == "self":
        a[0, 0, 0] = True
    elif case == "padded_edge":
        a[1, 4, 0] = True
    elif case == "overlap":
        r.different_speaker[0, 0, 0] = True
    elif case == "incomplete":
        r.same_speaker[0, 0, 0] = False
    elif case == "nonprefix":
        m[0, 1] = False
    elif case == "nan":
        h[0, 0, 0] = float("nan")
    else:
        h[1, 4] = 1
    with pytest.raises((TypeError, ValueError)):
        core_type(8)(h, a, r, m)


@pytest.mark.parametrize("core_type", CORES)
def test_singleton_batch_isolation_and_arbitrary_speaker_id_renaming(core_type):
    inputs = graph_inputs()
    core = core_type(8)
    short = prefix_inputs(inputs, 1)
    assert all(h.shape == (1, 1, 8) and torch.isfinite(h).all() for h in core(*short))
    full = core(*inputs)
    alone = (inputs[0][:1], inputs[1][:1],
             SpeakerRelations(inputs[2].same_speaker[:1], inputs[2].different_speaker[:1]), inputs[3][:1])
    for left, right in zip(full, core(*alone)):
        torch.testing.assert_close(left[:1], right, rtol=1e-6, atol=1e-6)
    renamed = torch.tensor([[81, 12, 81, 12, 81], [12, 12, 81, -1, -1]])
    rel = SpeakerRelationBuilder.build(renamed, inputs[-1])
    for left, right in zip(full, core(inputs[0], inputs[1], rel, inputs[-1])):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_track_persistent_source_equation_and_readout():
    h, a, r, mask = graph_inputs()
    core = TrackCore(8, initial_residual=0.1)
    operator = a.float() + torch.diag_embed(mask.float())
    operator /= operator.sum(-1, keepdim=True).clamp_min(1)
    first = r.same_speaker[:, 0]
    initial = (h * first[..., None], h * (mask & ~first)[..., None])
    messages = initial
    expected = []
    for readout in core.readouts:
        messages = tuple(torch.bmm(operator, value) + .1 * start
                         for value, start in zip(messages, initial))
        same = torch.where(first[..., None], messages[0], messages[1])
        other = torch.where(first[..., None], messages[1], messages[0])
        expected.append(readout(torch.cat((same, other), -1)) * mask[..., None])
    for actual, reference in zip(core(h, a, r, mask), expected):
        torch.testing.assert_close(actual, reference)


def test_track_rejects_three_speaker_groups():
    h, a, _, mask = graph_inputs()
    speakers = torch.tensor([[0, 1, 2, 0, 1], [1, 1, 0, -1, -1]])
    with pytest.raises(ValueError, match="two speaker"):
        TrackCore(8)(h, a, SpeakerRelationBuilder.build(speakers, mask), mask)


def test_coop_uniform_policy_is_quarter_message_without_renormalization():
    torch.manual_seed(100)
    coop = CooperativeAttention(8)
    parent = RelationAwareAttention(8)
    parent.load_state_dict({k: v for k, v in coop.state_dict().items() if not k.startswith("policy.")})
    with torch.no_grad():
        coop.policy.weight.zero_()
        coop.policy.bias.zero_()
    query, history = torch.randn(2, 8), torch.randn(2, 1, 8)
    edge, same = torch.ones(2, 1, dtype=torch.bool), torch.tensor([[True], [False]])
    base, soft = parent(query, history, edge, same), coop(query, history, edge, same)
    torch.testing.assert_close(soft.message, base.message * .25)
    torch.testing.assert_close(soft.weights, base.weights * .25)
    soft.message.square().sum().backward()
    assert coop.policy.bias.grad is not None and torch.count_nonzero(coop.policy.bias.grad)
    with torch.no_grad():
        coop.policy.bias[:] = torch.tensor([-100., -100., -100., 100.])
    assert torch.count_nonzero(coop(query, history, edge, same).message) == 0


def test_pdcn_matches_exact_source_formula_on_every_induced_prefix():
    h, a, r, mask = graph_inputs(hidden=4)
    # A sparse, branching DAG also exercises common ancestors beyond a chain.
    a[0].zero_()
    a[0, 1, 0] = a[0, 2, 0] = a[0, 3, 1] = a[0, 4, 2] = True
    a[0, 4, 3] = True
    core = DCNCore(4)
    actual = core(h, a, r, mask)
    for batch in range(2):
        for target in range(int(mask[batch].sum())):
            n = target + 1
            adj = a[batch, :n, :n].float()
            adj = adj / adj.sum(-1, keepdim=True).clamp_min(1)
            inv = torch.eye(n) - adj
            w = torch.linalg.inv(inv)
            reach = torch.eye(n)
            for power in range(1, n):
                reach = reach + torch.linalg.matrix_power(a[batch, :n, :n].float(), power)
            expected = [h.new_zeros(4) for _ in range(4)]
            for k in range(n):
                selector = torch.diag((reach[k] > 0).float())
                shift = w @ selector @ inv
                assert torch.count_nonzero(torch.triu(shift, diagonal=1)) == 0
                torch.testing.assert_close(shift @ shift, shift)
                branch = shift @ h[batch, :n]
                for depth, layer in enumerate(core.layers):
                    branch = torch.relu(layer(branch))
                    expected[depth] = expected[depth] + branch[-1]
            for depth in range(4):
                torch.testing.assert_close(actual[depth][batch, target], expected[depth], rtol=1e-5, atol=1e-6)


def test_ancestor_selectors_and_shift_factors_are_feature_independent():
    h, a, _, mask = graph_inputs()
    w, inv, reach = causal_shift_factors(a, mask)
    torch.testing.assert_close(w @ inv, torch.eye(5).expand(2, -1, -1))
    assert reach.dtype == torch.bool
    assert torch.equal(reach[0], torch.ones(5, 5, dtype=torch.bool).tril())
    assert not reach[1, 3:].any() and not reach[1, :, 3:].any()
