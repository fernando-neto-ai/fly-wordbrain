"""The chess board encoding, which training depends on and cannot re-derive.

Preparation resolves positions into packed bytes once; the trainer only ever sees those
bytes and unpacks them with numpy. If the packing and the unpacking disagree, nothing
downstream can notice -- the model simply learns from a corrupted board. So the round trip
is tested directly, and the move vocabulary is pinned to the size the reference uses.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_wordbrain.chess_encoding import (EXTRA_FEATURES, FEATURES, PACKED_BOARD_BYTES,
                                          PIECE_PLANES, SQUARES, build_move_vocabulary,
                                          features_from_packed, pack_board, unpack_board,
                                          win_probability)


class EncodingTests(unittest.TestCase):
    def test_feature_width_matches_the_reference(self):
        self.assertEqual(FEATURES, 780)
        self.assertEqual(PIECE_PLANES * SQUARES + EXTRA_FEATURES, FEATURES)

    def test_packing_round_trips_for_every_piece_code(self):
        codes = np.arange(13, dtype=np.uint8)[None, :].repeat(5, 0)
        codes = np.pad(codes, ((0, 0), (0, SQUARES - 13)))
        self.assertTrue(np.array_equal(unpack_board(pack_board(codes)), codes))

    def test_packing_is_two_squares_per_byte(self):
        self.assertEqual(pack_board(np.zeros((3, SQUARES), np.uint8)).shape,
                         (3, PACKED_BOARD_BYTES))

    def test_packing_refuses_codes_outside_the_alphabet(self):
        codes = np.zeros((1, SQUARES), np.uint8)
        codes[0, 0] = 13
        with self.assertRaises(ValueError):
            pack_board(codes)

    def test_each_occupied_square_sets_exactly_one_plane(self):
        generator = np.random.default_rng(1729)
        codes = generator.integers(0, 13, size=(16, SQUARES)).astype(np.uint8)
        features = features_from_packed(pack_board(codes), np.zeros((16, 2), np.uint8))
        planes = features[:, :PIECE_PLANES * SQUARES]
        self.assertTrue(np.array_equal(planes.sum(1), (codes > 0).sum(1).astype(np.float32)))
        # A piece must land in its own plane, not merely somewhere.
        row, square = 0, int(np.nonzero(codes[0])[0][0])
        plane = int(codes[0, square]) - 1
        self.assertEqual(features[row, plane * SQUARES + square], 1.0)

    def test_castling_and_en_passant_occupy_their_own_slots(self):
        extras = np.array([[0b1010, 0], [0b0001, 5]], np.uint8)
        features = features_from_packed(pack_board(np.zeros((2, SQUARES), np.uint8)), extras)
        base = PIECE_PLANES * SQUARES
        self.assertEqual(list(features[0, base:base + 4]), [0, 1, 0, 1])
        self.assertEqual(list(features[1, base:base + 4]), [1, 0, 0, 0])
        # Zero means "no en-passant square" and must light nothing.
        self.assertEqual(features[0, base + 4:].sum(), 0.0)
        self.assertEqual(features[1, base + 4:].sum(), 1.0)
        self.assertEqual(features[1, base + 4 + 4], 1.0)

    def test_move_vocabulary_matches_the_reference_size_and_is_unique(self):
        vocabulary = build_move_vocabulary()
        self.assertEqual(len(vocabulary), 1968)
        self.assertEqual(len(set(vocabulary)), 1968)
        for uci in ("e2e4", "g1f3", "e1g1", "a7a8q", "b7a8n"):
            self.assertIn(uci, vocabulary)
        # A knight cannot be reached by a queen line, and no piece moves three files
        # and one rank, so this pair belongs to no piece's geometry.
        self.assertNotIn("a1d2", vocabulary)

    def test_win_probability_is_centred_and_monotone(self):
        self.assertAlmostEqual(float(win_probability(0)), 0.5)
        values = win_probability([-1000, -100, 0, 100, 1000])
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertAlmostEqual(float(win_probability(300) + win_probability(-300)), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
