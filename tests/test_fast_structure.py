"""Selected fast edges must remain exact members of the canonical graph."""
from copy import deepcopy
import json

import numpy as np
import pytest

from fly_wordbrain.fast_structure import (
    ARRAY_KEYS, ORIGINAL_GROUPS, array_digest, build_structure, family_masks,
    load_structure, save_structure, validate_structure,
)


def toy_graph():
    node_type = np.asarray(["R1-R6", "hDeltaB", "hDeltaH", "hDeltaA", "hDeltaI", "hDeltaG",
                            "KCg-m", "MBON01", "FC1A", "LC4", "central_unknown", "DNexample"])
    superclass = np.asarray(["ol_sensory"] + ["cb_intrinsic"] * 8 + ["visual_projection", "cb_intrinsic", "descending_neuron"])
    # Two KC→MBON parallel canonical edges intentionally exercise duplicate
    # endpoints: distinct original IDs must never be coalesced.
    triples = [(0, 1, 1.), (0, 6, 1.), (0, 9, 1.),
               (1, 2, 1.), (1, 3, 1.), (1, 4, 1.), (1, 5, 1.),
               (2, 8, 1.), (3, 8, 1.), (4, 8, 1.), (5, 8, 1.),
               (6, 7, 5.), (6, 7, 5.), (7, 10, 5.), (7, 11, -4.),
               (8, 2, -3.), (8, 11, 4.), (9, 11, 7.), (9, 10, 5.),
               (10, 11, -8.), (11, 11, 2.)]
    triples.sort(key=lambda row: row[0])
    pre, post = np.asarray([row[0] for row in triples], np.int64), np.asarray([row[1] for row in triples], np.int32)
    weight = np.asarray([row[2] for row in triples], np.float32)
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=len(node_type)))].astype(np.int64)

    def distances(seeds, reverse=False):
        result = np.full(len(node_type), -1, np.int32)
        queue = list(seeds)
        result[queue] = 0
        for node in queue:
            targets = pre[post == node] if reverse else post[pre == node]
            for target in targets:
                if result[target] == -1:
                    result[target] = result[node] + 1
                    queue.append(int(target))
        return result

    original = np.flatnonzero(pre == 1).astype(np.int64)
    return {"ptr": ptr, "post": post, "weight": weight, "node_type": node_type, "superclass": superclass,
            "candidate_edge_ids": original, "candidate_group": np.arange(4, dtype=np.int8),
            "hops_from_retina": distances([0]), "hops_to_descending": distances([11], reverse=True)}


def rehash(arrays, metadata):
    metadata["selection_array_sha256"] = array_digest([arrays[name] for name in ARRAY_KEYS])


def test_deterministic_groups_preserve_original_edges_weights_signs_and_rank_ties():
    graph = toy_graph()
    original_graph = {key: value.copy() for key, value in graph.items()}
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=15)
    again, again_metadata = build_structure(graph, "graph-sha", target_edges=15)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], again[key])
    assert metadata == again_metadata
    assert metadata["group_names"][:4] == list(ORIGINAL_GROUPS)
    assert len(arrays["candidate_edge_ids"]) == 15
    np.testing.assert_array_equal(arrays["candidate_edge_ids"][:4], graph["candidate_edge_ids"])
    np.testing.assert_array_equal(arrays["candidate_group"][:4], graph["candidate_group"])
    assert (arrays["candidate_weight"] < 0).any()
    assert validate_structure(arrays, metadata, graph, "graph-sha")
    for key in graph:
        np.testing.assert_array_equal(graph[key], original_graph[key])
    # A one-new-edge budget first gives the KC pool one slot; equal score/hops
    # chooses the lower canonical edge ID, not an unstable sort.
    one, _ = build_structure(graph, "graph-sha", target_edges=5)
    kc_ids = np.flatnonzero((np.repeat(np.arange(12), np.diff(graph["ptr"])) == 6) & (graph["post"] == 7))
    assert one["candidate_edge_ids"][-1] == min(kc_ids)


def test_every_new_edge_matches_its_priority_owned_anatomical_family_and_sign():
    graph = toy_graph()
    pre = np.repeat(np.arange(12, dtype=np.int64), np.diff(graph["ptr"]))
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=15)
    allowed = {}
    free = np.ones(len(pre), bool)
    free[graph["candidate_edge_ids"]] = False
    for family, mask, _ in family_masks(graph, pre):
        allowed[family] = mask & free
        free[mask] = False
    for group in metadata["groups"][4:]:
        ids = arrays["candidate_edge_ids"][arrays["candidate_group"] == group["index"]]
        assert allowed[group["family"]][ids].all()
        expected_positive = group["sign"] == "positive"
        assert ((graph["weight"][ids] > 0) == expected_positive).all()
        assert (graph["hops_from_retina"][pre[ids]] >= 0).all()
        assert (graph["hops_to_descending"][graph["post"][ids]] >= 0).all()


def test_zero_unreachable_edges_are_ineligible_even_if_their_weight_would_rank_first():
    graph = toy_graph()
    # KC candidates both disappear; no forced fill from these forbidden IDs.
    graph["weight"][11] = 0.
    graph["hops_from_retina"][6] = -1
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=13)
    assert 11 not in arrays["candidate_edge_ids"] and 12 not in arrays["candidate_edge_ids"]
    assert "KC_to_MBON/positive" not in metadata["group_names"]


@pytest.mark.parametrize("field", ["candidate_pre", "candidate_post", "candidate_weight"])
def test_endpoint_or_baseweight_tampering_fails_even_with_updated_selection_digest(field):
    graph = toy_graph()
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=15)
    arrays[field][-1] += 1
    rehash(arrays, metadata)
    with pytest.raises(ValueError, match="endpoint|base weights"):
        validate_structure(arrays, metadata, graph, "graph-sha")


def test_original_membership_groupids_duplicates_and_cross_graph_are_rejected():
    graph = toy_graph()
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=15)
    with pytest.raises(ValueError, match="different canonical graph"):
        validate_structure(arrays, metadata, graph, "other-graph")
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["candidate_group"][0] = 1
    altered_metadata = deepcopy(metadata)
    rehash(changed, altered_metadata)
    with pytest.raises(ValueError, match="Groups|original candidate"):
        validate_structure(changed, altered_metadata, graph, "graph-sha")
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["candidate_edge_ids"][-1] = changed["candidate_edge_ids"][0]
    with pytest.raises(ValueError, match="unique canonical IDs"):
        validate_structure(changed, metadata, graph, "graph-sha")
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["candidate_group"][-1] = 999
    with pytest.raises(ValueError, match="Groups"):
        validate_structure(changed, metadata, graph, "graph-sha")


def test_small_sidecar_roundtrip_hashes_and_original_archive_not_rewritten(tmp_path):
    graph = toy_graph()
    arrays, metadata = build_structure(graph, "graph-sha", target_edges=15)
    path = tmp_path / "selected.npz"
    receipt = save_structure(path, arrays, metadata)
    assert receipt["sidecar_bytes"] < 10000
    loaded, loaded_metadata = load_structure(path, graph, "graph-sha")
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])
    assert loaded_metadata["original_candidate_edges"] == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["selected.json", "selected.npz"]
    with pytest.raises(ValueError, match="overwrite"):
        save_structure(path, arrays, metadata)
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="archive checksum"):
        load_structure(path, graph, "graph-sha")


def test_insufficient_or_invalid_budgets_fail_without_new_edges():
    graph = toy_graph()
    for count in (3, 21, 16385, 5.5):
        with pytest.raises(ValueError):
            build_structure(graph, "graph-sha", target_edges=count)
