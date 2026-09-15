"""Small asymmetric graphs verify export semantics without neural workloads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

SOURCE = Path(__file__).resolve().parents[1] / "scripts/prepare_plastic_graph.py"
SPEC = importlib.util.spec_from_file_location("prepare_plastic_graph", SOURCE)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def fixture_graph():
    # Includes signed, duplicate, zero, self, and unreachable edges. The full
    # graph must retain all of these, even when they cannot carry useful input.
    pre = np.array([0, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 7, 8], np.int32)
    post = np.array([1, 1, 2, 3, 4, 5, 5, 6, 6, 6, 6, 2, 8], np.int32)
    weight = np.array([2, .125, .5, -.75, 0, 1.25, -.25, -1, 2, 1, 1, 3, 5], np.float32)
    types = np.array(["R1-R6", "hDeltaB", "hDeltaH", "hDeltaA", "hDeltaI", "hDeltaG",
                      "DN", "hDeltaB", "DN"])
    graph = {"ids": np.arange(100, 109, dtype=np.int64),
             "ptr": np.r_[0, np.cumsum(np.bincount(pre, minlength=9))].astype(np.int64),
             "post": post, "weight": weight, "retina": np.array([0], np.int32),
             "superclass": np.array(["other"] * 6 + ["descending_neuron", "other", "descending_neuron"]),
             "uv": np.array([[.25, .75]], np.float32)}
    return graph, types


def test_incoming_orientation_and_all_canonical_edges_survive():
    graph, types = fixture_graph()
    original = {key: value.copy() for key, value in graph.items()}
    exported, meta = prep.build_export(graph, types)
    a = csr_matrix((exported["incoming_weight"], exported["incoming_pre"],
                    exported["incoming_ptr"]), shape=(9, 9))
    at = csr_matrix((exported["weight"], exported["post"], exported["ptr"]), shape=(9, 9))
    np.testing.assert_allclose(a @ np.arange(1, 10), [0, 2.25, 25, -1.5, 0, 2, 16, 0, 45])
    np.testing.assert_array_equal(a.toarray(), at.toarray().T)
    assert a.nnz == at.nnz == 13  # the duplicate is not coalesced
    np.testing.assert_array_equal(np.sort(exported["incoming_edge_ids"]), np.arange(13))
    incoming_post = np.repeat(np.arange(9), np.diff(exported["incoming_ptr"]))
    np.testing.assert_array_equal(incoming_post, graph["post"][exported["incoming_edge_ids"]])
    np.testing.assert_array_equal(exported["incoming_weight"], graph["weight"][exported["incoming_edge_ids"]])
    for key, value in original.items():
        np.testing.assert_array_equal(exported[key], value)
        assert prep.array_digest([exported[key]]) == prep.array_digest([value])
    assert meta["raw_weight_diagnostics"]["zero_weight_edges"] == 1
    assert exported["incoming_abs_sum"][5] == 1.5
    assert exported["incoming_signed_sum"][5] == 1.


def test_exact_candidate_ids_groups_and_directed_unreachable_paths():
    graph, types = fixture_graph()
    exported, meta = prep.build_export(graph, types)
    np.testing.assert_array_equal(exported["candidate_edge_ids"], [2, 3, 4, 5, 6, 11])
    np.testing.assert_array_equal(exported["candidate_group"], [0, 1, 2, 3, 3, 0])
    np.testing.assert_array_equal(exported["candidate_pre"], [1, 1, 1, 1, 1, 7])
    np.testing.assert_array_equal(exported["candidate_post"], [2, 3, 4, 5, 5, 2])
    np.testing.assert_array_equal(exported["descending"], [6, 8])
    np.testing.assert_array_equal(exported["hops_from_retina"], [0, 1, 2, 2, 2, 2, 3, -1, -1])
    np.testing.assert_array_equal(exported["hops_to_descending"], [3, 2, 1, 1, 1, 1, 0, 2, 0])
    np.testing.assert_array_equal(exported["candidate_hops_retina_via_edge_to_descending"], [3, 3, 3, 3, 3, -1])
    assert meta["candidate_groups"][0]["candidate_edges"] == 2
    assert meta["candidate_groups"][0]["retina_to_connected_presynaptic"]["unreachable"] == 1
    assert meta["candidate_groups"][2]["zero_weight_edges"] == 1
    assert meta["reachability"]["retina_to_descending"]["unreachable"] == 1


def test_annotation_join_is_exact_and_preserves_null():
    ids = np.array([20, 10, 30], np.int64)
    result = prep.align_annotation_types(ids, np.array([10, 30, 20, 99]),
                                         ["hDeltaH", None, "hDeltaB", "unused"])
    np.testing.assert_array_equal(result, ["hDeltaB", "hDeltaH", ""])
    assert result.dtype.kind == "U"
    with pytest.raises(ValueError, match="unique"):
        prep.align_annotation_types(ids, np.array([10, 10, 20]), ["a", "b", "c"])
    with pytest.raises(ValueError, match="lacks an exact"):
        prep.align_annotation_types(ids, np.array([10, 20]), ["a", "b"])
    with pytest.raises(ValueError, match="integer"):
        prep.align_annotation_types(ids, np.array([10., 20., 30.]), ["a", "b", "c"])


def test_missing_type_fails_instead_of_inventing_alias():
    graph, types = fixture_graph()
    types[2] = "hΔH"
    with pytest.raises(ValueError, match="aliases are forbidden"):
        prep.build_export(graph, types)


def test_existing_type_with_no_candidate_edges_is_reported():
    graph, types = fixture_graph()
    # Keep the I neuron and edge, but change its presynaptic source to another
    # node through a structurally valid CSR pointer. No graph edge is deleted.
    graph["ptr"] = np.array([0, 1, 4, 8, 9, 10, 11, 11, 12, 13], np.int64)
    exported, meta = prep.build_export(graph, types)
    assert len(exported["post"]) == 13
    assert meta["candidate_groups"][2]["zero_edges"]
    assert meta["candidate_groups"][2]["candidate_edges"] == 0


def test_hash_rejects_object_pointer_arrays_and_handles_empty_arrays():
    with pytest.raises(ValueError, match="Object arrays"):
        prep.array_digest([np.array(["hDeltaB"], dtype=object)])
    assert len(prep.array_digest([np.array([], np.int64)])) == 64
    assert prep.array_digest([np.array([1], np.int32)]) != prep.array_digest([np.array([1], np.int64)])


def test_power_diagnostic_uses_signed_incoming_matrix():
    diagonal = {"ids": np.array([1, 2], np.int64), "incoming_ptr": np.array([0, 1, 2], np.int64),
                "incoming_pre": np.array([0, 1], np.int32), "incoming_weight": np.array([-3, 1], np.float32)}
    result = prep.power_diagnostics(diagonal, iterations=30)
    assert result["last_norm_ratio"] == pytest.approx(3.)
    assert result["trace"][-1]["rayleigh"] == pytest.approx(-3.)
    assert result["trace"][-1]["relative_eigen_residual"] < 1e-10
    assert not prep.power_diagnostics(diagonal, iterations=0)["performed"]
