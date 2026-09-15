"""Small independent edge-alignment fixtures for anatomical gain annotations."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("prepare_connectorch_groups", Path(__file__).resolve().parents[1]
                                             / "scripts/prepare_connectorch_groups.py")
GROUPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROUPS)


def alignment_fixture():
    ids = np.array([10, 20, 30, 40, 50, 60], np.int64)
    superclass = np.array(["cb_intrinsic", "optic_lobe_intrinsic", "cb_sensory", "ascending_neuron",
                           "descending_neuron", "cb_intrinsic"])
    types = np.array(["A", "excluded", "A", "", "B", ""])
    ptr = np.array([0, 3, 4, 6, 8, 10, 11], np.int64)
    source = np.array([0, 2, 4, 0, 0, 3, 2, 4, 0, 3, 4], np.int64)
    weight = np.array([4, 0, -2, 1, 1, -3, 2, -4, 5, -2, -1], np.float32)
    checkpoint = {
        "brain.w_offsets": np.array([0, 3, 4, 6, 7, 8], np.int64),
        "brain.w_indices": np.array([0, 1, 3, 2, 1, 3, 2, 3], np.int64),
        "brain.w_values": np.array([4, 0, -2, -3, 2, -4, -2, -1], np.float32) * np.float32(.1),
        "brain.in_index": np.array([0, 2], np.int64),
        "brain.out_index": np.array([4], np.int64),
    }
    return [ids, superclass, types, ptr, source, weight, checkpoint]


def test_verified_subset_alignment_preserves_omissions_zeros_and_unknown_types():
    fixture = alignment_fixture()
    original = copy.deepcopy(fixture)
    arrays, meta = GROUPS.verify_alignment(*fixture)
    np.testing.assert_array_equal(arrays["node_ids"], [10, 30, 40, 50, 60])
    np.testing.assert_array_equal(arrays["node_type_labels"], ["type:A", "type:B", "unknown_id:40", "unknown_id:60"])
    np.testing.assert_array_equal(arrays["node_type_index"], [0, 0, 2, 1, 3])
    assert meta["nodes"] == 5 and meta["checkpoint_edges"] == 8
    assert meta["source_induced_edges"] == 10 and meta["source_edges_not_in_checkpoint"] == 2
    assert meta["zero_checkpoint_edges"] == 1
    assert meta["untyped_nodes"] == 2 and meta["node_type_groups"] == 4
    assert meta["factorized_gain_parameters"] == 8
    assert meta["source_to_checkpoint_global_scale"] == pytest.approx(.1)
    assert meta["max_weight_absolute_error"] < 1e-7
    assert meta["frozen_buffers_sha256"] == {key: GROUPS.array_hash(value) for key, value in fixture[-1].items()}
    for actual, expected in zip(fixture[:-1], original[:-1]):
        np.testing.assert_array_equal(actual, expected)
    for name in fixture[-1]:
        np.testing.assert_array_equal(fixture[-1][name], original[-1][name])


@pytest.mark.parametrize("change,error", [
    ("id_order", "unique sorted"),
    ("duplicate_ids", "unique sorted"),
    ("id_dtype", "integer"),
    ("wrong_class", "dimensions differ"),
    ("missing_endpoint", "absent"),
    ("edge_order", "canonically sorted"),
    ("edge_duplicate", "canonically sorted"),
    ("wrong_sign", "signs disagree"),
    ("changed_zero", "signs disagree"),
    ("wrong_magnitude", "single global scaling"),
])
def test_rejects_misaligned_annotation_or_checkpoint(change, error):
    args = alignment_fixture()
    checkpoint = args[-1]
    if change == "id_order":
        args[0][[0, 1]] = args[0][[1, 0]]
    elif change == "duplicate_ids":
        args[0][1] = args[0][0]
    elif change == "id_dtype":
        args[0] = args[0].astype(np.float64)
    elif change == "wrong_class":
        args[1][1] = "cb_intrinsic"
    elif change == "missing_endpoint":
        checkpoint["brain.w_indices"][2] = 2
    elif change == "edge_order":
        checkpoint["brain.w_indices"][:2] = [1, 0]
    elif change == "edge_duplicate":
        checkpoint["brain.w_indices"][1] = 0
    elif change == "wrong_sign":
        checkpoint["brain.w_values"][0] *= -1
    elif change == "changed_zero":
        checkpoint["brain.w_values"][1] = .1
    elif change == "wrong_magnitude":
        checkpoint["brain.w_values"][0] *= 1.1
    with pytest.raises(ValueError, match=error):
        GROUPS.verify_alignment(*args)


def test_rejects_ambiguous_source_endpoints():
    args = alignment_fixture()
    args[4][1] = args[4][0]
    with pytest.raises(ValueError, match="ambiguous edge endpoints"):
        GROUPS.verify_alignment(*args)


def test_frozen_hash_distinguishes_dtype_shape_and_values():
    a = np.array([1, 2], dtype=np.int64)
    variants = [a, a.astype(np.int32), a.reshape(1, 2), np.array([1, 3], dtype=np.int64)]
    assert len({GROUPS.array_hash(value) for value in variants}) == len(variants)


def test_cli_pinned_source_rejects_different_but_sorted_body_ids(tmp_path, monkeypatch):
    # Edge structure alone cannot authenticate absolute biological IDs. The CLI
    # must pin the annotated source bytes before it calls structural alignment.
    from safetensors.numpy import save_file
    args = alignment_fixture()
    graph = tmp_path / "graph.npz"
    model = tmp_path / "model.safetensors"
    def write_graph():
        np.savez(graph, ids=args[0], superclass=args[1], node_type=args[2],
                 incoming_ptr=args[3], incoming_pre=args[4], incoming_weight=args[5])
    write_graph()
    save_file(args[-1], model)
    monkeypatch.setattr(GROUPS, "GRAPH_SHA256", GROUPS.file_hash(graph))
    monkeypatch.setattr(GROUPS, "MODEL_SHA256", GROUPS.file_hash(model))
    args[0] += 1000
    write_graph()
    monkeypatch.setattr("sys.argv", ["prepare_connectorch_groups.py", "--graph", str(graph),
                                   "--model", str(model), "--output", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="Pinned source/checkpoint file hash mismatch"):
        GROUPS.main()
    assert not (tmp_path / "output").exists()
