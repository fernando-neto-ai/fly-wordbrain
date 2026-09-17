"""Board and move encoding shared by the chess data preparation and the trainer.

Nothing here imports python-chess. Preparation resolves positions into a packed byte
format once; training and evaluation unpack it with numpy alone, so the machine that
trains never needs a chess library and its pinned environment stays untouched.

Encoding, chosen to match the ChessFly reference so results are comparable:

    board    64 squares, 4 bits each, 0 empty and 1..12 for the piece types in the
             order P N B R Q K for the side to move then the same for the opponent,
             packed two squares per byte -> 32 bytes per position
    extras   castling rights for both sides (4 bits) and the en-passant file, stored as
             0 for none or 1..8 for the file
    features 780 floats: 12 piece planes x 64 squares, then 4 castling flags and 8
             en-passant file flags, all zero when there is no en-passant square
    moves    1968 entries: every from/to pair a queen or knight could traverse on an
             empty board, plus the four promotion choices for pawn pushes and captures
             reaching the last rank

Positions are stored already mirrored so the side to move is always "white": the
network is only ever asked one question, and the value is always from the mover's view.
"""
import numpy as np

SQUARES = 64
PIECE_PLANES = 12
EXTRA_FEATURES = 4 + 8          # castling rights, en-passant file flags
FEATURES = PIECE_PLANES * SQUARES + EXTRA_FEATURES      # 780
PACKED_BOARD_BYTES = SQUARES // 2                        # 32


def build_move_vocabulary():
    """The 1968 geometric moves, in a fixed order. Requires python-chess (preparation only)."""
    import chess
    moves = []
    for frm in chess.SQUARES:
        for to in chess.SQUARES:
            if frm == to:
                continue
            reachable = chess.BB_SQUARES[to] & (
                chess.BB_RANK_ATTACKS[frm][0] | chess.BB_FILE_ATTACKS[frm][0]
                | chess.BB_DIAG_ATTACKS[frm][0] | chess.BB_KNIGHT_ATTACKS[frm])
            if not reachable:
                continue
            moves.append(chess.Move(frm, to).uci())
            if chess.square_rank(to) in (0, 7) and abs(chess.square_file(frm) - chess.square_file(to)) <= 1:
                rank = chess.square_rank(frm)
                if (chess.square_rank(to) == 7 and rank == 6) or (chess.square_rank(to) == 0 and rank == 1):
                    for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                        moves.append(chess.Move(frm, to, promotion=piece).uci())
    return moves


def pack_board(piece_codes):
    """Pack 64 piece codes in 0..12 into 32 bytes, low nibble first."""
    codes = np.asarray(piece_codes, dtype=np.uint8).reshape(-1, SQUARES)
    if codes.max(initial=0) > 12:
        raise ValueError("Piece codes must be 0..12")
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def unpack_board(packed):
    """Inverse of pack_board: (N, 32) bytes -> (N, 64) piece codes."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, PACKED_BOARD_BYTES)
    codes = np.empty((packed.shape[0], SQUARES), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def features_from_packed(packed, extras, out=None):
    """(N, 32) + (N, 2) -> (N, 780) float32 board features."""
    codes = unpack_board(packed)
    # A corrupt corpus otherwise surfaces as an out-of-bounds write into the feature
    # block, which reads as a shape bug anywhere but here. The check is over 32 bytes a
    # row and is free next to the sparse matmul that follows.
    if codes.size and codes.max() > PIECE_PLANES:
        raise ValueError(f"Packed board holds piece code {int(codes.max())}; "
                         f"valid codes are 0..{PIECE_PLANES}")
    count = codes.shape[0]
    features = np.zeros((count, FEATURES), dtype=np.float32) if out is None else out
    features[:] = 0.0
    occupied = codes > 0
    rows, squares = np.nonzero(occupied)
    planes = codes[rows, squares].astype(np.int64) - 1
    features[rows, planes * SQUARES + squares] = 1.0
    extras = np.asarray(extras, dtype=np.uint8).reshape(-1, 2)
    base = PIECE_PLANES * SQUARES
    for bit in range(4):
        features[:, base + bit] = (extras[:, 0] >> bit) & 1
    # extras[:, 1] is 0 for no en-passant square, else the file numbered 1..8; an
    # all-zero block therefore means "none" and needs no slot of its own.
    stored = extras[:, 1].astype(np.int64)
    present = np.nonzero(stored)[0]
    features[present, base + 4 + stored[present] - 1] = 1.0
    return features


def win_probability(centipawns):
    """Lichess' logistic conversion from centipawns to a win probability in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-0.00368208 * np.asarray(centipawns, dtype=np.float64)))
