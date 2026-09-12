#!/usr/bin/env python3
"""Causal cache-transplant study of inferred chess state in frozen Qwen3.5.

This standalone runner implements ``docs/qwen35_chess_memory_plan.md``.  It
keeps the language model frozen, uses exact python-chess labels, and refuses to
score held-out data until the cache implementation and development gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qwen35-chess-memory-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/qwen35-chess-memory-cache")

import numpy as np
import torch


SCHEMA_VERSION = 1
STUDY_ID = "qwen35-chess-memory-v1"
PLAN_PATH = Path("docs/qwen35_chess_memory_plan.md")
DEFAULT_CONFIG_PATH = Path("data/qwen35_chess_memory/config.json")
DEFAULT_MANIFEST_PATH = Path("data/qwen35_chess_memory/manifest.json")
HEADER_TEXT = "Chess game from the standard starting position. Moves are in UCI notation.\nMoves: "
QUERY_TEXT = "\nNext move:"
PIECE_CLASSES = ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
PIECE_TO_INDEX = {value: index for index, value in enumerate(PIECE_CLASSES)}
SQUARE_NAMES = tuple(f"{file_name}{rank}" for rank in range(1, 9) for file_name in "abcdefgh")
HEAD_CLASS_COUNTS = (13,) * 64 + (2,) + (2,) * 4 + (65,)
HEAD_NAMES = SQUARE_NAMES + ("side_to_move", "castle_WK", "castle_WQ", "castle_BK", "castle_BQ", "en_passant")
HEAD_OFFSETS = tuple(np.cumsum((0,) + HEAD_CLASS_COUNTS).tolist())
CONDITION_IDS = ("AA", "BB", "BA", "AB")
HORIZONS = (0, 1, 2, 4)


class ProtocolError(RuntimeError):
    """Raised before an experimental invariant can be violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path.cwd() / value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return sha256_bytes("\x1f".join(str(part) for part in parts).encode("utf-8"))


def stable_bucket(modulus: int, *parts: Any) -> int:
    return int(stable_hash(*parts)[:16], 16) % modulus


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"required file is missing: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_new_or_resume(path: Path, resume: bool) -> Any | None:
    if not path.exists():
        return None
    if not resume:
        raise ProtocolError(f"artifact already exists: {path}; pass --resume to verify and reuse it")
    return read_json(path) if path.suffix == ".json" else path


def package_versions(names: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_payload() -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "scikit-learn", "matplotlib", "python-chess", "zstandard")
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def initialize_run(config_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / "config.json"
    payload = read_json(config_path)
    if config_copy.exists() and read_json(config_copy) != payload:
        raise ProtocolError("run directory is bound to a different config")
    if not config_copy.exists():
        atomic_write_json(config_copy, payload)
    if not (run_dir / "time_log.csv").exists():
        atomic_write_text(run_dir / "time_log.csv", "stage,kind,seconds,note\n")
    if not (run_dir / "decisions.md").exists():
        atomic_write_text(
            run_dir / "decisions.md",
            "# Decision log\n\nRecord protocol amendments, failures, and schedule decisions here as they occur.\n",
        )


def log_time(run_dir: Path, stage: str, started: float, note: str = "") -> None:
    with (run_dir / "time_log.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow((stage, "wall_clock_stage", f"{time.monotonic() - started:.6f}", note))


@dataclass(frozen=True)
class StudyConfig:
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        value = cls(read_json(resolve_path(path)))
        value.validate()
        return value

    def validate(self) -> None:
        required = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "seed": 11,
            "bootstrap_seed": 41,
            "bootstrap_replicates": 10000,
        }
        for key, expected in required.items():
            if self.raw.get(key) != expected:
                raise ProtocolError(f"config {key} must equal {expected!r}")
        model = self.raw.get("model", {})
        if model.get("primary_id") != "Qwen/Qwen3.5-4B-Base":
            raise ProtocolError("primary model differs from the frozen plan")
        if model.get("fallback_id") != "Qwen/Qwen3.5-2B-Base":
            raise ProtocolError("fallback model differs from the frozen plan")
        revisions = model.get("revisions", {})
        if set(revisions) != {model.get("primary_id"), model.get("fallback_id")} or any(
            not isinstance(value, str) or len(value) != 40 for value in revisions.values()
        ):
            raise ProtocolError("primary and fallback models require separate immutable revisions")
        if model.get("dtype") != "float16" or model.get("device") != "cuda":
            raise ProtocolError("model precision/device differs from the frozen plan")
        if model.get("attn_implementation") != "eager" or int(model.get("max_tokens", -1)) != 512:
            raise ProtocolError("attention backend or token limit differs from the frozen plan")
        representation = self.raw.get("representation", {})
        if representation != {
            "header_text": HEADER_TEXT,
            "query_text": QUERY_TEXT,
            "candidate_prefix": " ",
            "candidate_suffix": "\n",
            "add_special_tokens": False,
        }:
            raise ProtocolError("text representation differs from the frozen plan")
        if tuple(self.raw.get("continuation_horizons", [])) != HORIZONS:
            raise ProtocolError("continuation horizons differ from the frozen plan")
        probe = self.raw.get("probe", {})
        if probe.get("ridge_grid") != [0.1, 1.0, 10.0, 100.0, 1000.0]:
            raise ProtocolError("ridge grid differs from the frozen plan")
        if probe.get("feature_std_floor") != 1e-6:
            raise ProtocolError("probe standard-deviation floor differs from the frozen plan")
        source = self.raw.get("source", {})
        if source.get("expected_sha256") != "aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635":
            raise ProtocolError("Lichess archive checksum differs from the frozen source")

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def model(self) -> Mapping[str, Any]:
        return self.raw["model"]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.raw["source"]

    @property
    def counts(self) -> Mapping[str, int]:
        return self.raw["counts"]

    @property
    def gates(self) -> Mapping[str, Any]:
        return self.raw["gates"]


def stage_context(args: argparse.Namespace) -> tuple[StudyConfig, Path, Path]:
    config_path = resolve_path(args.config)
    config = StudyConfig.load(config_path)
    run_dir = resolve_path(args.run_dir)
    initialize_run(config_path, run_dir)
    return config, config_path, run_dir


def import_chess() -> Any:
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise ProtocolError("python-chess is required; install requirements_qwen35_chess_memory.txt") from exc
    return chess


def render_context(moves: Sequence[str]) -> str:
    if not moves:
        raise ProtocolError("a chess history must contain at least one move")
    return HEADER_TEXT + " ".join(str(move) for move in moves)


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in values]


def tokenize_context_query(tokenizer: Any, context_text: str) -> tuple[list[int], list[int], list[int]]:
    context_ids = encode_ids(tokenizer, context_text)
    complete_ids = encode_ids(tokenizer, context_text + QUERY_TEXT)
    if complete_ids[: len(context_ids)] != context_ids:
        raise ProtocolError("context token IDs are not a prefix after the query is appended")
    query_ids = complete_ids[len(context_ids) :]
    expected_query = encode_ids(tokenizer, QUERY_TEXT)
    if query_ids != expected_query:
        raise ProtocolError("contextual query token IDs differ from the shared query IDs")
    if not query_ids:
        raise ProtocolError("query tokenization is empty")
    return context_ids, query_ids, complete_ids


def candidate_continuation_ids(tokenizer: Any, context_text: str, move: str) -> list[int]:
    _, _, prefix_ids = tokenize_context_query(tokenizer, context_text)
    complete = context_text + QUERY_TEXT + " " + move + "\n"
    complete_ids = encode_ids(tokenizer, complete)
    if complete_ids[: len(prefix_ids)] != prefix_ids:
        raise ProtocolError(f"candidate {move!r} changes the context/query token prefix")
    result = complete_ids[len(prefix_ids) :]
    if not result:
        raise ProtocolError(f"candidate {move!r} has no continuation tokens")
    return result


def advance_suffix_ids(tokenizer: Any, previous_moves: Sequence[str], next_move: str) -> list[int]:
    previous = encode_ids(tokenizer, render_context(previous_moves))
    advanced = encode_ids(tokenizer, render_context((*previous_moves, next_move)))
    if advanced[: len(previous)] != previous:
        raise ProtocolError(f"appending move {next_move!r} changes the cached token prefix")
    suffix = advanced[len(previous) :]
    if not suffix:
        raise ProtocolError(f"appending move {next_move!r} produces no tokens")
    return suffix


def token_span_for_moves(tokenizer: Any, context_text: str) -> list[int]:
    if not getattr(tokenizer, "is_fast", False):
        raise ProtocolError("the orderless input baseline requires a fast tokenizer with offsets")
    encoded = tokenizer(context_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    start = len(HEADER_TEXT)
    selected = [index for index, (left, right) in enumerate(offsets) if int(right) > start and int(left) < len(context_text)]
    if not selected:
        raise ProtocolError("could not locate move tokens in the rendered context")
    return selected


def token_span_for_last_move(tokenizer: Any, context_text: str, last_move: str) -> list[int]:
    if not getattr(tokenizer, "is_fast", False):
        raise ProtocolError("the last-move input baseline requires a fast tokenizer with offsets")
    encoded = tokenizer(context_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    start = len(context_text) - len(last_move)
    selected = [index for index, (left, right) in enumerate(offsets) if int(right) > start and int(left) < len(context_text)]
    if not selected:
        raise ProtocolError("could not locate the last move's contextual tokens")
    return selected


def strict_board_from_moves(moves: Sequence[str]) -> Any:
    chess = import_chess()
    board = chess.Board()
    for ply, uci in enumerate(moves, 1):
        try:
            move = chess.Move.from_uci(str(uci))
        except ValueError as exc:
            raise ProtocolError(f"invalid UCI move at ply {ply}: {uci!r}") from exc
        if move == chess.Move.null() or move not in board.legal_moves:
            raise ProtocolError(f"illegal/null move at ply {ply}: {uci!r}")
        board.push(move)
    return board


def piece_labels(board: Any) -> list[int]:
    chess = import_chess()
    labels: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        labels.append(PIECE_TO_INDEX["empty" if piece is None else piece.symbol()])
    return labels


def board_targets(board: Any) -> list[int]:
    chess = import_chess()
    values = piece_labels(board)
    values.append(0 if board.turn == chess.WHITE else 1)
    values.extend(
        [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ]
    )
    values.append(0 if board.ep_square is None else int(board.ep_square) + 1)
    if len(values) != 70:
        raise AssertionError("board target vector must contain 70 heads")
    return values


def target_key(targets: Sequence[int]) -> str:
    return stable_hash("piece-and-rule-state", *[int(value) for value in targets])


def six_field_fen(board: Any) -> str:
    return str(board.fen(en_passant="fen"))


def move_touched_squares(board: Any, move: Any) -> set[int]:
    chess = import_chess()
    touched = {int(move.from_square), int(move.to_square)}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:
            touched.update((chess.square(7, rank), chess.square(5, rank)))
        else:
            touched.update((chess.square(0, rank), chess.square(3, rank)))
    if board.is_en_passant(move):
        captured = move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
        touched.add(int(captured))
    return touched


def touched_square_history(moves: Sequence[str]) -> list[set[int]]:
    chess = import_chess()
    board = chess.Board()
    result: list[set[int]] = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ProtocolError(f"illegal move while computing touched squares: {uci}")
        result.append(move_touched_squares(board, move))
        board.push(move)
    return result


def material_signature(board: Any) -> dict[str, int]:
    return {symbol: piece_labels(board).count(index) for index, symbol in enumerate(PIECE_CLASSES[1:], 1)}


def partition_for_game(full_moves: Sequence[str], seed: int = 11) -> str:
    bucket = stable_bucket(100, seed, " ".join(full_moves))
    if bucket <= 9:
        return "pilot"
    if bucket <= 59:
        return "train"
    if bucket <= 74:
        return "development"
    return "test"


def position_payload(
    full_moves: Sequence[str],
    cutoff: int,
    game_id: str,
    source_index: int,
    split: str,
    tokenizer: Any,
    seed: int = 11,
    pretokenized: tuple[Sequence[int], Sequence[int], Sequence[int]] | None = None,
) -> dict[str, Any]:
    history = tuple(full_moves[:cutoff])
    board = strict_board_from_moves(history)
    text = render_context(history)
    if pretokenized is None:
        context_ids, query_ids, full_ids = tokenize_context_query(tokenizer, text)
    else:
        context_ids = [int(value) for value in pretokenized[0]]
        query_ids = [int(value) for value in pretokenized[1]]
        full_ids = [int(value) for value in pretokenized[2]]
        if full_ids != context_ids + query_ids:
            raise ProtocolError("batched token contract is not context followed by the fixed query")
    targets = board_targets(board)
    return {
        "position_id": stable_hash(game_id, cutoff)[:24],
        "game_id": game_id,
        "source_game_hash": stable_hash(" ".join(full_moves)),
        "partition_bucket": stable_bucket(100, seed, " ".join(full_moves)),
        "source_index": int(source_index),
        "split": split,
        "cutoff_ply": int(cutoff),
        "moves": list(history),
        "observed_next_moves": list(full_moves[cutoff : cutoff + 4]),
        "context_text": text,
        "context_token_ids": context_ids,
        "query_token_ids": query_ids,
        "context_query_token_ids": full_ids,
        "context_token_length": len(context_ids),
        "total_token_length": len(full_ids),
        "fen": six_field_fen(board),
        "targets": targets,
        "target_key": target_key(targets),
        "legal_moves": sorted(move.uci() for move in board.legal_moves),
        "side_to_move": "white" if board.turn else "black",
        "piece_count": len(board.piece_map()),
        "in_check": bool(board.is_check()),
        "halfmove_clock": int(board.halfmove_clock),
        "fullmove_number": int(board.fullmove_number),
        "material": material_signature(board),
    }


def state_difference_count(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    return sum(int(a) != int(b) for a, b in zip(first["targets"][:64], second["targets"][:64]))


def contextual_candidate_ids(tokenizer: Any, position: Mapping[str, Any], move: str) -> list[int] | None:
    try:
        return candidate_continuation_ids(tokenizer, str(position["context_text"]), move)
    except ProtocolError:
        return None


def find_opposing_candidates(
    tokenizer: Any,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any] | None:
    first_legal = set(str(value) for value in first["legal_moves"])
    second_legal = set(str(value) for value in second["legal_moves"])
    first_only = sorted(first_legal - second_legal, key=lambda move: stable_hash(first["position_id"], second["position_id"], "a", move))
    second_only = sorted(second_legal - first_legal, key=lambda move: stable_hash(first["position_id"], second["position_id"], "b", move))
    first_candidates: list[tuple[str, list[int]]] = []
    second_candidates: list[tuple[str, list[int]]] = []
    for move in first_only:
        ids_first = contextual_candidate_ids(tokenizer, first, move)
        ids_second = contextual_candidate_ids(tokenizer, second, move)
        if ids_first is not None and ids_first == ids_second:
            first_candidates.append((move, ids_first))
    for move in second_only:
        ids_first = contextual_candidate_ids(tokenizer, first, move)
        ids_second = contextual_candidate_ids(tokenizer, second, move)
        if ids_first is not None and ids_first == ids_second:
            second_candidates.append((move, ids_first))
    for first_move, first_ids in first_candidates:
        for second_move, second_ids in second_candidates:
            if len(first_ids) == len(second_ids):
                return {
                    "a_move": first_move,
                    "b_move": second_move,
                    "a_candidate_ids": first_ids,
                    "b_candidate_ids": second_ids,
                    "candidate_token_count": len(first_ids),
                }
    return None


def pair_match_key(position: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(position["cutoff_ply"]),
        int(position["context_token_length"]),
        str(position["side_to_move"]),
        int(position["piece_count"]),
        bool(position["in_check"]),
    )


def token_hamming_distance(first: Sequence[int], second: Sequence[int]) -> int:
    if len(first) != len(second):
        raise ProtocolError("token Hamming distance requires equal-length histories")
    return sum(int(a) != int(b) for a, b in zip(first, second))


def annotate_pair_history(pair: MutableMapping[str, Any], first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    chess = import_chess()
    source_squares = {
        chess.Move.from_uci(str(pair["a_move"])).from_square,
        chess.Move.from_uci(str(pair["b_move"])).from_square,
    }
    histories = (touched_square_history(first["moves"]), touched_square_history(second["moves"]))
    last_four = set().union(*(set().union(*history[-4:]) if history[-4:] else set() for history in histories))
    pair["older_state_subgroup"] = not bool(source_squares & last_four)
    pair["source_square_last_touched"] = {}
    for square in sorted(source_squares):
        name = chess.square_name(square)
        pair["source_square_last_touched"][name] = []
        for history in histories:
            last = max((ply for ply, touched in enumerate(history, 1) if square in touched), default=None)
            pair["source_square_last_touched"][name].append(last)
    pair["shared_final_two_moves"] = list(first["moves"][-2:]) == list(second["moves"][-2:])


def advance_position(position: Mapping[str, Any], suffix_moves: Sequence[str], tokenizer: Any) -> dict[str, Any]:
    moves = list(position["moves"])
    advance_ids: list[int] = []
    for move in suffix_moves:
        ids = advance_suffix_ids(tokenizer, moves, move)
        advance_ids.extend(ids)
        moves.append(move)
    board = strict_board_from_moves(moves)
    text = render_context(moves)
    context_ids, query_ids, full_ids = tokenize_context_query(tokenizer, text)
    if context_ids != list(position["context_token_ids"]) + advance_ids:
        raise ProtocolError("incremental continuation IDs do not equal the full advanced context")
    targets = board_targets(board)
    return {
        "position_id": stable_hash("advanced", position["position_id"], *suffix_moves)[:24],
        "game_id": position.get("game_id"),
        "split": position.get("split"),
        "cutoff_ply": len(moves),
        "moves": moves,
        "advance_token_ids": advance_ids,
        "context_text": text,
        "context_token_ids": context_ids,
        "query_token_ids": query_ids,
        "context_query_token_ids": full_ids,
        "context_token_length": len(context_ids),
        "total_token_length": len(full_ids),
        "fen": six_field_fen(board),
        "targets": targets,
        "target_key": target_key(targets),
        "legal_moves": sorted(move.uci() for move in board.legal_moves),
        "side_to_move": "white" if board.turn else "black",
        "piece_count": len(board.piece_map()),
        "in_check": bool(board.is_check()),
        "halfmove_clock": int(board.halfmove_clock),
        "fullmove_number": int(board.fullmove_number),
        "material": material_signature(board),
    }


def find_shared_continuation(
    tokenizer: Any,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any] | None:
    for donor, observed in (("A", first["observed_next_moves"]), ("B", second["observed_next_moves"])):
        suffix = list(observed[:4])
        if len(suffix) < 4:
            continue
        try:
            advanced_a = {horizon: advance_position(first, suffix[:horizon], tokenizer) for horizon in (1, 2, 4)}
            advanced_b = {horizon: advance_position(second, suffix[:horizon], tokenizer) for horizon in (1, 2, 4)}
        except ProtocolError:
            continue
        if any(advanced_a[h]["targets"] == advanced_b[h]["targets"] for h in (1, 2, 4)):
            continue
        if any(advanced_a[h]["advance_token_ids"] != advanced_b[h]["advance_token_ids"] for h in (1, 2, 4)):
            continue
        horizons: dict[str, Any] = {}
        failed = False
        for horizon in (1, 2, 4):
            candidates = find_opposing_candidates(tokenizer, advanced_a[horizon], advanced_b[horizon])
            if candidates is None:
                failed = True
                break
            horizons[str(horizon)] = {
                "a": advanced_a[horizon],
                "b": advanced_b[horizon],
                "candidates": candidates,
            }
        if failed:
            continue
        return {"source": donor, "moves": suffix, "horizons": horizons}
    return None


def select_cutoffs(full_moves: Sequence[str], split: str, game_id: str) -> list[int]:
    if split == "train":
        bands = ((12, 19), (20, 27), (28, 35), (36, 44))
        band = bands[stable_bucket(len(bands), game_id, "band")]
        wanted_parity = stable_bucket(2, game_id, "turn")
        values = [ply for ply in range(band[0], min(band[1], len(full_moves)) + 1) if ply % 2 == wanted_parity]
        if not values:
            return []
        return [values[stable_bucket(len(values), game_id, "cutoff")]]
    maximum = min(40, len(full_moves) - 4)
    if maximum < 12:
        return []
    available = list(range(12, maximum + 1))
    ordered = sorted(available, key=lambda ply: stable_hash(game_id, split, ply))
    return sorted(ordered[: min(3, len(ordered))])


def iter_lichess_games(path: Path, maximum_games: int | None = None) -> Iterator[dict[str, Any]]:
    chess = import_chess()
    try:
        import zstandard
    except ImportError as exc:
        raise ProtocolError("zstandard is required to stream the Lichess archive") from exc
    with path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            source_index = 0
            valid_index = 0
            while maximum_games is None or valid_index < maximum_games:
                try:
                    game = chess.pgn.read_game(text)
                except Exception as exc:
                    source_index += 1
                    yield {"source_index": source_index, "error": f"parse_exception:{type(exc).__name__}"}
                    continue
                if game is None:
                    break
                source_index += 1
                if game.errors:
                    yield {"source_index": source_index, "error": "pgn_errors"}
                    continue
                variant = str(game.headers.get("Variant", "Standard"))
                if variant not in ("", "Standard") or game.headers.get("FEN"):
                    yield {"source_index": source_index, "error": "nonstandard_start_or_variant"}
                    continue
                board = chess.Board()
                moves: list[str] = []
                valid = True
                for move in game.mainline_moves():
                    if move == chess.Move.null() or move not in board.legal_moves:
                        valid = False
                        break
                    moves.append(move.uci())
                    board.push(move)
                if not valid or len(moves) < 16:
                    yield {"source_index": source_index, "error": "illegal_or_short_game"}
                    continue
                game_hash = stable_hash(" ".join(moves))
                valid_index += 1
                yield {
                    "source_index": source_index,
                    "valid_index": valid_index,
                    "game_id": game_hash[:24],
                    "game_hash": game_hash,
                    "moves": moves,
                }


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, handle)
    os.replace(temporary, path)


def resolve_hf_snapshot(model_id: str, requested_revision: str, allow_remote_downloads: bool) -> tuple[str, Path, list[dict[str, Any]]]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ProtocolError("huggingface-hub is required to resolve the tokenizer revision") from exc
    if allow_remote_downloads:
        revision = str(HfApi().model_info(model_id, revision=requested_revision).sha)
    else:
        revision = requested_revision
        if len(revision) != 40:
            raise ProtocolError("offline prepare requires a 40-character immutable model revision")
    patterns = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template*",
    )
    snapshot = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=list(patterns),
            local_files_only=not allow_remote_downloads,
        )
    )
    files = [
        {"path": str(path.relative_to(snapshot)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    return revision, snapshot, files


def load_tokenizer(snapshot_path: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ProtocolError("transformers is required to load the frozen tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), use_fast=True, local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ProtocolError("the frozen study requires the fast tokenizer for baseline token spans")
    return tokenizer


def build_candidate_pools(
    archive_path: Path,
    tokenizer: Any,
    maximum_games: int | None,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], int, int, dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    scanned = 0
    valid_scanned = 0
    seen_games: set[str] = set()
    resource_fixture: dict[str, Any] | None = None
    shared_query_ids = encode_ids(tokenizer, QUERY_TEXT)
    for game in iter_lichess_games(archive_path, maximum_games):
        scanned = max(scanned, int(game["source_index"]))
        valid_scanned = max(valid_scanned, int(game.get("valid_index", 0)))
        if "error" in game:
            exclusions[str(game["error"])] += 1
            continue
        if game["game_hash"] in seen_games:
            exclusions["duplicate_game"] += 1
            continue
        seen_games.add(str(game["game_hash"]))
        split = partition_for_game(game["moves"], seed)
        # Once an exact maximum-size valid workload exists, later games cannot
        # improve this engineering fixture and are not retokenized for it.
        if resource_fixture is None or int(resource_fixture["planned_workload_tokens"]) < max_tokens:
            for cutoff in range(len(game["moves"]), 0, -1):
                try:
                    fixture = position_payload(
                        game["moves"], cutoff, game["game_id"], int(game["source_index"]), split, tokenizer
                    )
                    board = strict_board_from_moves(fixture["moves"])
                    candidate_move = sorted(move.uci() for move in board.legal_moves)[0]
                    candidate_ids = candidate_continuation_ids(tokenizer, fixture["context_text"], candidate_move)
                except (ProtocolError, IndexError):
                    continue
                workload_tokens = int(fixture["total_token_length"]) + len(candidate_ids)
                if workload_tokens <= max_tokens:
                    fixture["resource_candidate_move"] = candidate_move
                    fixture["resource_candidate_ids"] = candidate_ids
                    fixture["planned_workload_tokens"] = workload_tokens
                    if resource_fixture is None or workload_tokens > int(resource_fixture["planned_workload_tokens"]):
                        resource_fixture = fixture
                    break
        cutoffs = select_cutoffs(game["moves"], split, game["game_id"])
        if not cutoffs:
            exclusions["no_eligible_cutoff"] += 1
            continue
        texts = [render_context(game["moves"][:cutoff]) for cutoff in cutoffs]
        context_batches = tokenizer(texts, add_special_tokens=False)["input_ids"]
        complete_batches = tokenizer(
            [text + QUERY_TEXT for text in texts], add_special_tokens=False
        )["input_ids"]
        for cutoff, context_batch, complete_batch in zip(cutoffs, context_batches, complete_batches):
            try:
                context_ids = [int(value) for value in context_batch]
                complete_ids = [int(value) for value in complete_batch]
                if complete_ids[: len(context_ids)] != context_ids:
                    raise ProtocolError("batched context IDs are not a prefix after appending the query")
                query_ids = complete_ids[len(context_ids) :]
                if query_ids != shared_query_ids:
                    raise ProtocolError("batched contextual query IDs differ from the shared query")
                position = position_payload(
                    game["moves"],
                    cutoff,
                    game["game_id"],
                    int(game["source_index"]),
                    split,
                    tokenizer,
                    pretokenized=(context_ids, query_ids, complete_ids),
                )
            except ProtocolError:
                exclusions["token_or_position_error"] += 1
                continue
            if int(position["total_token_length"]) + 16 > max_tokens:
                exclusions["over_token_limit"] += 1
                continue
            pools[split].append(position)
    for split in pools:
        pools[split].sort(key=lambda row: stable_hash(seed, split, row["position_id"]))
    if resource_fixture is None:
        raise ProtocolError("no valid resource fixture was found under the 512-token ceiling")
    return dict(pools), exclusions, scanned, valid_scanned, resource_fixture


def select_training_positions(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    reserved_targets: set[str],
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        ply = int(row["cutoff_ply"])
        band = next(name for low, high, name in ((12, 19, "12-19"), (20, 27, "20-27"), (28, 35, "28-35"), (36, 44, "36-44")) if low <= ply <= high)
        cells[(str(row["side_to_move"]), band)].append(row)
    selected: list[dict[str, Any]] = []
    used_games: set[str] = set()
    cell_order = sorted(cells)
    while len(selected) < count:
        progressed = False
        for cell in cell_order:
            while cells[cell]:
                row = cells[cell].pop(0)
                if row["game_id"] in used_games or row["target_key"] in reserved_targets:
                    continue
                selected.append(dict(row))
                used_games.add(str(row["game_id"]))
                reserved_targets.add(str(row["target_key"]))
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) < count:
        raise ProtocolError(f"only {len(selected)} unique balanced training states were available; need {count}")
    return selected


def make_pair_payload(
    split: str,
    index: int,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    orientation = stable_bucket(2, split, first["position_id"], second["position_id"])
    a, b = (first, second) if orientation == 0 else (second, first)
    candidates = find_opposing_candidates(tokenizer, a, b)
    if candidates is None:
        return None
    pair: dict[str, Any] = {
        "pair_id": f"{split}-{index:04d}-{stable_hash(a['position_id'], b['position_id'])[:8]}",
        "split": split,
        "a_position_id": a["position_id"],
        "b_position_id": b["position_id"],
        "changed_square_count": state_difference_count(a, b),
        "token_hamming_distance": token_hamming_distance(a["context_token_ids"], b["context_token_ids"]),
        **candidates,
    }
    annotate_pair_history(pair, a, b)
    return pair


def select_pairs(
    split: str,
    candidates: Sequence[Mapping[str, Any]],
    target_count: int,
    tokenizer: Any,
    reserved_targets: set[str],
    used_games: set[str] | None = None,
    want_continuations: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    used = set() if used_games is None else used_games
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["game_id"] not in used and row["target_key"] not in reserved_targets:
            groups[pair_match_key(row)].append(row)
    proposals: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: stable_hash(split, key, row["position_id"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 : left_index + 101]:
                if left["game_id"] == right["game_id"] or state_difference_count(left, right) < 2:
                    continue
                proposals.append((stable_hash(split, left["position_id"], right["position_id"]), left, right))
    proposals.sort(key=lambda value: value[0])

    positions: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    pair_cache: dict[str, dict[str, Any] | None] = {}
    selected: list[tuple[str, dict[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []

    def eligible_base_pair(
        proposal_hash: str,
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]] | None:
        if first["game_id"] in used or second["game_id"] in used:
            return None
        if first["target_key"] in reserved_targets or second["target_key"] in reserved_targets:
            return None
        if proposal_hash not in pair_cache:
            pair_cache[proposal_hash] = make_pair_payload(split, 0, first, second, tokenizer)
            if pair_cache[proposal_hash] is None:
                exclusions["no_opposing_matched_candidates"] += 1
        cached = pair_cache[proposal_hash]
        if cached is None:
            return None
        pair = dict(cached)
        a = first if pair["a_position_id"] == first["position_id"] else second
        b = second if a is first else first
        return pair, a, b

    def commit_pair(
        proposal_hash: str,
        pair: dict[str, Any],
        a: Mapping[str, Any],
        b: Mapping[str, Any],
        continuation: Mapping[str, Any] | None = None,
    ) -> None:
        if continuation is not None:
            pair["continuation"] = dict(continuation)
            for horizon_row in continuation["horizons"].values():
                reserved_targets.add(str(horizon_row["a"]["target_key"]))
                reserved_targets.add(str(horizon_row["b"]["target_key"]))
        selected.append((proposal_hash, pair, a, b))
        for row in (a, b):
            positions[str(row["position_id"])] = dict(row)
            reserved_targets.add(str(row["target_key"]))
            used.add(str(row["game_id"]))

    # Continuation eligibility is rare. Search it prospectively before filling
    # ordinary pairs so an early target-count stop cannot masquerade as an
    # archive-wide continuation shortfall.
    continuation_count = 0
    if want_continuations:
        for proposal_hash, first, second in proposals:
            if continuation_count >= want_continuations:
                break
            candidate = eligible_base_pair(proposal_hash, first, second)
            if candidate is None:
                continue
            pair, a, b = candidate
            continuation = find_shared_continuation(tokenizer, a, b)
            if continuation is None:
                continue
            future_keys = {
                str(horizon_row[donor]["target_key"])
                for horizon_row in continuation["horizons"].values()
                for donor in ("a", "b")
            }
            if future_keys & reserved_targets or len(future_keys) != 6:
                exclusions["continuation_target_collision"] += 1
                continue
            commit_pair(proposal_hash, pair, a, b, continuation)
            continuation_count += 1

    for proposal_hash, first, second in proposals:
        if len(selected) >= target_count:
            break
        candidate = eligible_base_pair(proposal_hash, first, second)
        if candidate is None:
            continue
        pair, a, b = candidate
        commit_pair(proposal_hash, pair, a, b)

    selected.sort(key=lambda value: value[0])
    pairs: list[dict[str, Any]] = []
    for index, (_, pair, _, _) in enumerate(selected):
        pair["pair_id"] = (
            f"{split}-{index:04d}-{stable_hash(pair['a_position_id'], pair['b_position_id'])[:8]}"
        )
        pairs.append(pair)
    if len(pairs) < target_count:
        exclusions["pair_shortfall"] = target_count - len(pairs)
    if want_continuations and continuation_count < want_continuations:
        exclusions["continuation_shortfall"] = want_continuations - continuation_count
    return pairs, positions, exclusions


def select_transposition_triplets(
    candidates: Sequence[Mapping[str, Any]],
    target_count: int,
    tokenizer: Any,
    reserved_targets: set[str],
    used_games: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    used = used_games
    exact: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    structural: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["game_id"] in used or row["target_key"] in reserved_targets:
            continue
        exact[(row["fen"], row["cutoff_ply"], row["context_token_length"])].append(row)
        structural[pair_match_key(row)].append(row)
    proposals: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for key, rows in exact.items():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda row: stable_hash("transposition", key, row["position_id"]))
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 : i + 21]:
                if first["game_id"] != second["game_id"] and first["moves"] != second["moves"]:
                    proposals.append((stable_hash(first["position_id"], second["position_id"]), first, second))
    proposals.sort(key=lambda value: value[0])
    triplets: list[dict[str, Any]] = []
    positions: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    for _, first, same in proposals:
        if len(triplets) >= target_count:
            break
        if first["game_id"] in used or same["game_id"] in used:
            continue
        same_distance = token_hamming_distance(first["context_token_ids"], same["context_token_ids"])
        b_rows = sorted(
            structural[pair_match_key(first)],
            key=lambda row: stable_hash(first["position_id"], same["position_id"], row["position_id"]),
        )
        chosen_b = None
        chosen_candidates = None
        for candidate in b_rows:
            if candidate["game_id"] in used or candidate["game_id"] in (first["game_id"], same["game_id"]):
                continue
            if candidate["target_key"] in reserved_targets or state_difference_count(first, candidate) < 2:
                continue
            different_distance = token_hamming_distance(first["context_token_ids"], candidate["context_token_ids"])
            if abs(different_distance - same_distance) > 2:
                continue
            opposing = find_opposing_candidates(tokenizer, first, candidate)
            if opposing is not None:
                chosen_b, chosen_candidates = candidate, opposing
                break
        if chosen_b is None or chosen_candidates is None:
            exclusions["no_matched_different_position"] += 1
            continue
        triplet = {
            "triplet_id": f"transposition-{len(triplets):04d}-{stable_hash(first['position_id'], same['position_id'], chosen_b['position_id'])[:8]}",
            "a_position_id": first["position_id"],
            "aprime_position_id": same["position_id"],
            "b_position_id": chosen_b["position_id"],
            "same_position_token_hamming": same_distance,
            "different_position_token_hamming": token_hamming_distance(first["context_token_ids"], chosen_b["context_token_ids"]),
            **chosen_candidates,
        }
        triplets.append(triplet)
        for row in (first, same, chosen_b):
            positions[str(row["position_id"])] = dict(row)
            used.add(str(row["game_id"]))
        reserved_targets.add(str(first["target_key"]))
        reserved_targets.add(str(chosen_b["target_key"]))
    if len(triplets) < target_count:
        exclusions["transposition_shortfall"] = target_count - len(triplets)
    return triplets, positions, exclusions


def manifest_hash(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("manifest_sha256", None)
    return sha256_json(copy)


def validate_manifest_hash(payload: Mapping[str, Any]) -> None:
    if payload.get("manifest_sha256") != manifest_hash(payload):
        raise ProtocolError("manifest SHA256 does not match its contents")


def build_manifest(
    config: StudyConfig,
    archive_path: Path,
    archive_sha256: str,
    model_id: str,
    model_revision: str,
    tokenizer_snapshot: Path,
    tokenizer_files: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    maximum_games: int | None,
) -> dict[str, Any]:
    pools, source_exclusions, scanned, valid_scanned, resource_fixture = build_candidate_pools(
        archive_path, tokenizer, maximum_games, config.seed, int(config.model["max_tokens"])
    )
    reserved_targets: set[str] = set()
    all_positions: dict[str, dict[str, Any]] = {}
    exclusions = Counter(source_exclusions)

    pilot_pairs, pilot_positions, counts = select_pairs(
        "pilot", pools.get("pilot", []), int(config.counts["pilot_pairs"]), tokenizer, reserved_targets
    )
    all_positions.update(pilot_positions)
    exclusions.update(counts)

    training = select_training_positions(
        pools.get("train", []), int(config.counts["train_states"]), reserved_targets
    )
    all_positions.update({row["position_id"]: row for row in training})

    development_pairs, development_positions, counts = select_pairs(
        "development",
        pools.get("development", []),
        int(config.counts["development_pairs"]),
        tokenizer,
        reserved_targets,
    )
    all_positions.update(development_positions)
    exclusions.update(counts)

    test_used_games: set[str] = set()
    transpositions, transposition_positions, counts = select_transposition_triplets(
        pools.get("test", []),
        int(config.counts["transposition_triplets"]),
        tokenizer,
        reserved_targets,
        test_used_games,
    )
    all_positions.update(transposition_positions)
    exclusions.update(counts)

    test_pairs, test_positions, counts = select_pairs(
        "test",
        pools.get("test", []),
        int(config.counts["test_pairs"]),
        tokenizer,
        reserved_targets,
        test_used_games,
        want_continuations=int(config.counts["continuation_pairs"]),
    )
    all_positions.update(test_positions)
    exclusions.update(counts)

    retrieved_at = utc_now()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": retrieved_at,
        "source": {
            "url": config.source["url"],
            "checksum_url": config.source["checksum_url"],
            "archive_sha256": archive_sha256,
            "publisher_sha256": config.source["expected_sha256"],
            "retrieved_at": retrieved_at,
            "archive_bytes": archive_path.stat().st_size,
            "archive_games_examined": scanned,
            "valid_games_scanned": valid_scanned,
            "scan_limit": maximum_games,
            "expanded_beyond_initial_scan": maximum_games is None,
        },
        "model": {
            "model_id": model_id,
            "resolved_revision": model_revision,
            "tokenizer_snapshot": str(tokenizer_snapshot),
            "tokenizer_files": list(tokenizer_files),
        },
        "representation": dict(config.raw["representation"]),
        "label_contract": {
            "square_order": list(SQUARE_NAMES),
            "piece_classes": list(PIECE_CLASSES),
            "head_names": list(HEAD_NAMES),
            "head_class_counts": list(HEAD_CLASS_COUNTS),
            "en_passant": "0 means none; square index plus one otherwise; FEN raw en-passant semantics",
        },
        "positions": {key: all_positions[key] for key in sorted(all_positions)},
        "train_position_ids": [row["position_id"] for row in training],
        "pilot_pairs": pilot_pairs,
        "development_pairs": development_pairs,
        "test_pairs": test_pairs,
        "transposition_triplets": transpositions,
        "resource_fixture_position": resource_fixture["position_id"],
        "resource_fixture": resource_fixture,
        "exclusions": dict(sorted(exclusions.items())),
        "actual_counts": {
            "train_states": len(training),
            "pilot_pairs": len(pilot_pairs),
            "development_pairs": len(development_pairs),
            "test_pairs": len(test_pairs),
            "continuation_pairs": sum("continuation" in pair for pair in test_pairs),
            "transposition_triplets": len(transpositions),
        },
    }
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


# ---------------------------------------------------------------------------
# Immutable hybrid-cache representation
# ---------------------------------------------------------------------------


def clone_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
        return value
    raise ProtocolError(f"unsupported cache metadata type: {type(value).__name__}")


def state_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {key: state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(state_to_device(item, device) for item in value)
    return clone_state_value(value)


def iter_state_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_state_tensors(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_state_tensors(item, child)


def metadata_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return {"torch_dtype": str(value)}
    if isinstance(value, torch.device):
        return {"torch_device": str(value)}
    if isinstance(value, dict):
        return {str(key): metadata_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [metadata_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": tensor_sha256(value),
        }
    raise ProtocolError(f"unsupported cache value in metadata: {type(value).__name__}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    update_digest_with_tensor(digest, tensor)
    return digest.hexdigest()


def update_digest_with_tensor(digest: Any, tensor: torch.Tensor) -> None:
    """Feed exact tensor storage bytes to hashlib without Python byte iteration."""
    contiguous = tensor.detach().cpu().contiguous().clone()
    try:
        array = contiguous.numpy()
        digest.update(memoryview(array).cast("B"))
    except (RuntimeError, TypeError):
        # Compatibility path for a broken or dtype-incompatible NumPy bridge.
        digest.update(bytes(contiguous.untyped_storage()))


@dataclass(frozen=True)
class LayerSnapshot:
    index: int
    kind: str
    class_name: str
    state: Mapping[str, Any]


@dataclass(frozen=True)
class CacheSnapshot:
    sequence_length: int
    next_position: int
    attention_mask_length: int
    layer_types: tuple[str, ...]
    layers: tuple[LayerSnapshot, ...]
    rope_deltas: torch.Tensor | None = None


def clone_snapshot(snapshot: CacheSnapshot) -> CacheSnapshot:
    return CacheSnapshot(
        sequence_length=int(snapshot.sequence_length),
        next_position=int(snapshot.next_position),
        attention_mask_length=int(snapshot.attention_mask_length),
        layer_types=tuple(snapshot.layer_types),
        layers=tuple(
            LayerSnapshot(layer.index, layer.kind, layer.class_name, clone_state_value(layer.state))
            for layer in snapshot.layers
        ),
        rope_deltas=None if snapshot.rope_deltas is None else snapshot.rope_deltas.detach().clone(),
    )


def validate_snapshot(snapshot: CacheSnapshot, check_finite: bool = True) -> dict[str, Any]:
    if snapshot.sequence_length <= 0:
        raise ProtocolError("cache snapshot has no processed tokens")
    if snapshot.next_position != snapshot.sequence_length:
        raise ProtocolError("cache next position differs from sequence length")
    if snapshot.attention_mask_length != snapshot.sequence_length:
        raise ProtocolError("cache attention-mask length differs from sequence length")
    if len(snapshot.layers) != len(snapshot.layer_types):
        raise ProtocolError("cache layer count differs from the model layer map")
    schemas: list[dict[str, Any]] = []
    total_bytes = 0
    recurrent_bytes = 0
    attention_bytes = 0
    for expected_index, (layer_type, layer) in enumerate(zip(snapshot.layer_types, snapshot.layers)):
        if layer.index != expected_index or layer.kind != layer_type:
            raise ProtocolError("cache layer index/type mapping is inconsistent")
        names = {name for name, _ in iter_state_tensors(layer.state)}
        if layer_type == "linear_attention":
            if not any(name.startswith("conv_states.") for name in names):
                raise ProtocolError(f"linear layer {expected_index} is missing convolution state")
            if not any(name.startswith("recurrent_states.") for name in names):
                raise ProtocolError(f"linear layer {expected_index} is missing recurrent state")
            required_metadata = {
                "is_conv_states_initialized",
                "is_recurrent_states_initialized",
                "has_previous_state",
                "conv_kernel_size",
            }
            if not required_metadata.issubset(layer.state):
                raise ProtocolError(f"linear layer {expected_index} is missing initialization metadata")
        elif layer_type == "full_attention":
            if "keys" not in layer.state or "values" not in layer.state:
                raise ProtocolError(f"attention layer {expected_index} is missing key/value tensors")
        else:
            raise ProtocolError(f"unexpected Qwen layer type: {layer_type}")
        layer_bytes = 0
        tensors: list[dict[str, Any]] = []
        for name, tensor in iter_state_tensors(layer.state):
            if check_finite and not torch.isfinite(tensor).all():
                raise ProtocolError(f"non-finite cache tensor at layer {expected_index}:{name}")
            size = tensor.numel() * tensor.element_size()
            layer_bytes += size
            tensors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": size})
        total_bytes += layer_bytes
        if layer_type == "linear_attention":
            recurrent_bytes += layer_bytes
        else:
            attention_bytes += layer_bytes
        schemas.append(
            {
                "index": expected_index,
                "kind": layer_type,
                "class_name": layer.class_name,
                "bytes": layer_bytes,
                "tensors": tensors,
            }
        )
    return {
        "sequence_length": snapshot.sequence_length,
        "layer_count": len(snapshot.layers),
        "total_bytes": total_bytes,
        "recurrent_bytes": recurrent_bytes,
        "attention_bytes": attention_bytes,
        "layers": schemas,
    }


def snapshot_digest(snapshot: CacheSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "sequence_length": snapshot.sequence_length,
                "next_position": snapshot.next_position,
                "attention_mask_length": snapshot.attention_mask_length,
                "layer_types": snapshot.layer_types,
            }
        ).encode("utf-8")
    )
    for layer in snapshot.layers:
        digest.update(f"{layer.index}:{layer.kind}:{layer.class_name}".encode("utf-8"))
        tensor_names = {name for name, _ in iter_state_tensors(layer.state)}
        scalar_state = {key: value for key, value in layer.state.items() if not any(name == key or name.startswith(f"{key}.") for name in tensor_names)}
        digest.update(canonical_json(metadata_value(scalar_state)).encode("utf-8"))
        for name, tensor in iter_state_tensors(layer.state):
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(canonical_json(list(tensor.shape)).encode("utf-8"))
            update_digest_with_tensor(digest, tensor)
    if snapshot.rope_deltas is not None:
        update_digest_with_tensor(digest, snapshot.rope_deltas)
    return digest.hexdigest()


def snapshot_storage_pointers(snapshot: CacheSnapshot) -> set[tuple[str, int]]:
    pointers: set[tuple[str, int]] = set()
    for layer in snapshot.layers:
        for name, tensor in iter_state_tensors(layer.state):
            pointers.add((f"{layer.index}:{name}", int(tensor.untyped_storage().data_ptr())))
    return pointers


def assert_storage_independent(first: CacheSnapshot, second: CacheSnapshot) -> None:
    first_map = dict(snapshot_storage_pointers(first))
    second_map = dict(snapshot_storage_pointers(second))
    if first_map.keys() != second_map.keys():
        raise ProtocolError("cache clones do not contain identical tensor paths")
    shared = [name for name in first_map if first_map[name] == second_map[name]]
    if shared:
        raise ProtocolError(f"cache clones share mutable tensor storage: {shared[:3]}")


def assemble_snapshot(recurrent_donor: CacheSnapshot, kv_donor: CacheSnapshot) -> CacheSnapshot:
    if recurrent_donor.layer_types != kv_donor.layer_types:
        raise ProtocolError("cannot assemble snapshots from different layer maps")
    if recurrent_donor.sequence_length != kv_donor.sequence_length:
        raise ProtocolError("cache transplantation requires equal token sequence lengths")
    layers: list[LayerSnapshot] = []
    for kind, recurrent_layer, kv_layer in zip(
        recurrent_donor.layer_types, recurrent_donor.layers, kv_donor.layers
    ):
        source = recurrent_layer if kind == "linear_attention" else kv_layer
        layers.append(LayerSnapshot(source.index, source.kind, source.class_name, clone_state_value(source.state)))
    result = CacheSnapshot(
        sequence_length=recurrent_donor.sequence_length,
        next_position=recurrent_donor.sequence_length,
        attention_mask_length=recurrent_donor.sequence_length,
        layer_types=recurrent_donor.layer_types,
        layers=tuple(layers),
        rope_deltas=None,
    )
    validate_snapshot(result, check_finite=False)
    return result


def zero_channels(snapshot: CacheSnapshot, recurrent: bool = False, attention: bool = False) -> CacheSnapshot:
    if not recurrent and not attention:
        return clone_snapshot(snapshot)
    layers: list[LayerSnapshot] = []
    for layer in snapshot.layers:
        should_zero = (recurrent and layer.kind == "linear_attention") or (
            attention and layer.kind == "full_attention"
        )
        state = clone_state_value(layer.state)
        if should_zero:
            for _, tensor in iter_state_tensors(state):
                tensor.zero_()
        layers.append(LayerSnapshot(layer.index, layer.kind, layer.class_name, state))
    result = CacheSnapshot(
        snapshot.sequence_length,
        snapshot.next_position,
        snapshot.attention_mask_length,
        snapshot.layer_types,
        tuple(layers),
        None if snapshot.rope_deltas is None else snapshot.rope_deltas.detach().clone(),
    )
    validate_snapshot(result, check_finite=False)
    return result


def save_snapshot_fixture(path: Path, snapshot: CacheSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        torch.save(snapshot, handle)
    os.replace(temporary, path)


def load_snapshot_fixture(path: Path) -> CacheSnapshot:
    try:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        snapshot = torch.load(path, map_location="cpu")
    if not isinstance(snapshot, CacheSnapshot):
        raise ProtocolError("serialized cache fixture has the wrong type")
    validate_snapshot(snapshot)
    return snapshot


def full_parameter_digest(module: torch.nn.Module, chunk_elements: int = 2_000_000) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters(), key=lambda pair: pair[0]):
        value = parameter.detach().contiguous().view(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(canonical_json(list(parameter.shape)).encode("utf-8"))
        for start in range(0, value.numel(), chunk_elements):
            update_digest_with_tensor(digest, value[start : start + chunk_elements])
    return digest.hexdigest()


def model_weight_files(snapshot: Path) -> list[dict[str, Any]]:
    patterns = ("*.safetensors", "*.bin", "*.json")
    paths = sorted({path for pattern in patterns for path in snapshot.glob(pattern) if path.is_file()})
    return [
        {"path": str(path.relative_to(snapshot)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    ]


class FrozenQwenBackend:
    """Text-only access to the language stack of the pinned multimodal Qwen checkpoint."""

    def __init__(
        self,
        config: StudyConfig,
        manifest: Mapping[str, Any],
        allow_remote_downloads: bool = False,
    ) -> None:
        if importlib.metadata.version("transformers") != "5.17.0":
            raise ProtocolError("the real backend requires transformers==5.17.0")
        if not torch.cuda.is_available():
            raise ProtocolError("the frozen protocol requires a CUDA GPU")
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForMultimodalLM, AutoTokenizer, DynamicCache
        except ImportError as exc:
            raise ProtocolError("the pinned Qwen backend dependencies are unavailable") from exc

        self._DynamicCache = DynamicCache
        self.device = torch.device(str(config.model["device"]))
        self.model_id = str(manifest["model"]["model_id"])
        self.revision = str(manifest["model"]["resolved_revision"])
        self.snapshot_path = Path(
            snapshot_download(
                self.model_id,
                revision=self.revision,
                local_files_only=not allow_remote_downloads,
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.snapshot_path, use_fast=True, local_files_only=True
        )
        load_kwargs = {
            "dtype": torch.float16,
            "attn_implementation": str(config.model["attn_implementation"]),
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        self.full_model = AutoModelForMultimodalLM.from_pretrained(self.snapshot_path, **load_kwargs)
        self.full_model.eval()
        self.full_model.requires_grad_(False)
        if not hasattr(self.full_model, "model") or not hasattr(self.full_model.model, "language_model"):
            raise ProtocolError("loaded checkpoint does not expose the expected Qwen3.5 language_model")
        self.language_model = self.full_model.model.language_model
        self.lm_head = self.full_model.lm_head
        self.language_model.to(self.device)
        self.lm_head.to(self.device)
        self.language_model.eval()
        self.lm_head.eval()
        self.text_config = self.language_model.config
        self.layer_types = tuple(str(value) for value in self.text_config.layer_types)
        if len(self.layer_types) != int(self.text_config.num_hidden_layers):
            raise ProtocolError("text config layer map is inconsistent")
        if set(self.layer_types) != {"linear_attention", "full_attention"}:
            raise ProtocolError(f"unexpected text layer types: {sorted(set(self.layer_types))}")
        self.hidden_size = int(self.text_config.hidden_size)
        self.max_tokens = int(config.model["max_tokens"])
        self.weight_files = model_weight_files(self.snapshot_path)

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "resolved_revision": self.revision,
            "snapshot_path": str(self.snapshot_path),
            "model_class": type(self.full_model).__name__,
            "language_model_class": type(self.language_model).__name__,
            "hidden_size": self.hidden_size,
            "layer_types": list(self.layer_types),
            "layer_type_counts": dict(Counter(self.layer_types)),
            "dtype": str(next(self.language_model.parameters()).dtype),
            "device": str(next(self.language_model.parameters()).device),
            "attention_backend": getattr(self.full_model.config, "_attn_implementation", None),
            "weight_files": self.weight_files,
        }

    def _position_ids(self, start: int, length: int) -> torch.Tensor:
        return torch.arange(start, start + length, dtype=torch.long, device=self.device).view(1, -1)

    def _snapshot_cache(
        self, cache: Any, sequence_length: int, check_finite: bool = True
    ) -> CacheSnapshot:
        layers: list[LayerSnapshot] = []
        if len(cache.layers) != len(self.layer_types):
            raise ProtocolError("returned cache has the wrong number of layers")
        for index, (kind, cache_layer) in enumerate(zip(self.layer_types, cache.layers)):
            state = {key: clone_state_value(value) for key, value in cache_layer.__dict__.items()}
            layers.append(LayerSnapshot(index, kind, type(cache_layer).__name__, state))
        snapshot = CacheSnapshot(
            sequence_length=sequence_length,
            next_position=sequence_length,
            attention_mask_length=sequence_length,
            layer_types=self.layer_types,
            layers=tuple(layers),
            rope_deltas=None,
        )
        validate_snapshot(snapshot, check_finite=check_finite)
        return snapshot

    def _restore_cache(self, snapshot: CacheSnapshot) -> Any:
        validate_snapshot(snapshot, check_finite=False)
        if snapshot.layer_types != self.layer_types:
            raise ProtocolError("snapshot layer map differs from the loaded model")
        cache = self._DynamicCache(config=self.text_config)
        if len(cache.layers) != len(snapshot.layers):
            raise ProtocolError("new DynamicCache has the wrong number of layers")
        for cache_layer, saved_layer in zip(cache.layers, snapshot.layers):
            if type(cache_layer).__name__ != saved_layer.class_name:
                raise ProtocolError("cache implementation class changed since snapshot creation")
            for key, value in saved_layer.state.items():
                setattr(cache_layer, key, state_to_device(value, self.device))
        if int(cache.get_seq_length()) != snapshot.sequence_length:
            raise ProtocolError("restored cache sequence length differs from snapshot metadata")
        return cache

    def prefill(self, context_ids: Sequence[int]) -> CacheSnapshot:
        ids = [int(value) for value in context_ids]
        if not ids or len(ids) > self.max_tokens:
            raise ProtocolError("prefill length is empty or exceeds the frozen token ceiling")
        cache = self._DynamicCache(config=self.text_config)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.language_model(
                input_ids=input_ids,
                position_ids=self._position_ids(0, len(ids)),
                past_key_values=cache,
                use_cache=True,
            )
        if not torch.isfinite(output.last_hidden_state).all():
            raise ProtocolError("prefill produced non-finite hidden states")
        return self._snapshot_cache(output.past_key_values, len(ids))

    def _consume_tokens(
        self,
        snapshot: CacheSnapshot,
        token_ids: Sequence[int],
    ) -> tuple[CacheSnapshot, torch.Tensor]:
        if not token_ids:
            raise ProtocolError("cached continuation requires at least one token")
        cache = self._restore_cache(snapshot)
        last_hidden: torch.Tensor | None = None
        position = snapshot.next_position
        with torch.inference_mode():
            for token in token_ids:
                input_ids = torch.tensor([[int(token)]], dtype=torch.long, device=self.device)
                output = self.language_model(
                    input_ids=input_ids,
                    position_ids=self._position_ids(position, 1),
                    past_key_values=cache,
                    use_cache=True,
                )
                last_hidden = output.last_hidden_state[:, -1, :]
                position += 1
        if last_hidden is None or not torch.isfinite(last_hidden).all():
            raise ProtocolError("cached continuation produced no finite hidden vector")
        return self._snapshot_cache(cache, position, check_finite=False), last_hidden

    def advance(self, snapshot: CacheSnapshot, appended_move_ids: Sequence[int]) -> CacheSnapshot:
        advanced, _ = self._consume_tokens(snapshot, appended_move_ids)
        return advanced

    def read_feature(self, snapshot: CacheSnapshot, query_ids: Sequence[int]) -> np.ndarray:
        _, hidden = self._consume_tokens(snapshot, query_ids)
        return hidden[0].float().cpu().numpy()

    def score_candidate(
        self,
        snapshot: CacheSnapshot,
        query_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> dict[str, Any]:
        targets = [int(value) for value in candidate_ids]
        if not targets:
            raise ProtocolError("candidate continuation is empty")
        state, hidden = self._consume_tokens(snapshot, query_ids)
        token_logprobs: list[float] = []
        argmax_ids: list[int] = []
        for index, target in enumerate(targets):
            with torch.inference_mode():
                logits = self.lm_head(hidden).float()
                logprob = torch.log_softmax(logits, dim=-1)[0, target]
                predicted = int(torch.argmax(logits, dim=-1).item())
            if not torch.isfinite(logprob):
                raise ProtocolError("candidate scoring produced a non-finite log-probability")
            token_logprobs.append(float(logprob.item()))
            argmax_ids.append(predicted)
            if index + 1 < len(targets):
                state, hidden = self._consume_tokens(state, [target])
        return {
            "token_ids": targets,
            "token_logprobs": token_logprobs,
            "argmax_ids": argmax_ids,
            "sum_logprob": float(sum(token_logprobs)),
            "mean_logprob": float(np.mean(token_logprobs)),
        }

    def fresh_feature(self, context_ids: Sequence[int], query_ids: Sequence[int]) -> np.ndarray:
        ids = [int(value) for value in (*context_ids, *query_ids)]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.language_model(
                input_ids=input_ids,
                position_ids=self._position_ids(0, len(ids)),
                use_cache=False,
            )
        hidden = output.last_hidden_state[0, -1].float()
        if not torch.isfinite(hidden).all():
            raise ProtocolError("fresh forward produced a non-finite feature")
        return hidden.cpu().numpy()

    def fresh_score_candidate(
        self,
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> dict[str, Any]:
        prefix = [int(value) for value in (*context_ids, *query_ids)]
        targets = [int(value) for value in candidate_ids]
        if not prefix or not targets:
            raise ProtocolError("fresh scoring requires nonempty prefix and candidate")
        fed = prefix + targets[:-1]
        input_ids = torch.tensor([fed], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.language_model(
                input_ids=input_ids,
                position_ids=self._position_ids(0, len(fed)),
                use_cache=False,
            )
            start = len(prefix) - 1
            selected = output.last_hidden_state[:, start : start + len(targets), :]
            logits = self.lm_head(selected).float()
            logprobs = torch.log_softmax(logits, dim=-1)
        values = [float(logprobs[0, index, target].item()) for index, target in enumerate(targets)]
        argmax_ids = [int(torch.argmax(logits[0, index]).item()) for index in range(len(targets))]
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("fresh candidate scoring produced non-finite values")
        return {
            "token_ids": targets,
            "token_logprobs": values,
            "argmax_ids": argmax_ids,
            "sum_logprob": float(sum(values)),
            "mean_logprob": float(np.mean(values)),
        }

    def reference_last_logits(self, all_ids: Sequence[int]) -> torch.Tensor:
        ids = [int(value) for value in all_ids]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        if hasattr(self.full_model.model, "rope_deltas"):
            self.full_model.model.rope_deltas = None
        with torch.inference_mode():
            output = self.full_model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        return output.logits[0, -1].float()

    def language_last_logits(self, all_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [int(value) for value in all_ids]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.language_model(
                input_ids=input_ids,
                position_ids=self._position_ids(0, len(ids)),
                use_cache=False,
            )
            hidden = output.last_hidden_state[0, -1]
            logits = self.lm_head(hidden).float()
        return hidden.float(), logits

    def input_embedding_features(self, position: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ids = torch.tensor([position["context_token_ids"]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            embeddings = self.language_model.embed_tokens(ids)[0].float()
        move_indices = token_span_for_moves(self.tokenizer, str(position["context_text"]))
        last_indices = token_span_for_last_move(
            self.tokenizer, str(position["context_text"]), str(position["moves"][-1])
        )
        return (
            embeddings[move_indices].mean(0).cpu().numpy(),
            embeddings[last_indices].mean(0).cpu().numpy(),
        )


def load_backend(
    config: StudyConfig,
    manifest: Mapping[str, Any],
    allow_remote_downloads: bool = False,
) -> tuple[FrozenQwenBackend, dict[str, Any]]:
    backend = FrozenQwenBackend(config, manifest, allow_remote_downloads=allow_remote_downloads)
    return backend, backend.runtime_manifest()


# ---------------------------------------------------------------------------
# Preparation and deterministic data audit
# ---------------------------------------------------------------------------


def manifest_path_for_args(args: argparse.Namespace) -> Path:
    return resolve_path(getattr(args, "manifest", DEFAULT_MANIFEST_PATH))


def load_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError("manifest root must be an object")
    validate_manifest_hash(payload)
    return payload


def resolve_manifest_tokenizer(manifest: Mapping[str, Any], allow_remote_downloads: bool) -> Any:
    snapshot = Path(str(manifest["model"].get("tokenizer_snapshot", "")))
    if snapshot.exists():
        return load_tokenizer(snapshot)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ProtocolError("huggingface-hub is required to recover the frozen tokenizer") from exc
    snapshot = Path(
        snapshot_download(
            str(manifest["model"]["model_id"]),
            revision=str(manifest["model"]["resolved_revision"]),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "chat_template*",
            ],
            local_files_only=not allow_remote_downloads,
        )
    )
    return load_tokenizer(snapshot)


def verify_publisher_checksum(checksum_path: Path, archive_name: str, expected: str) -> None:
    matches = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[-1].lstrip("*") == archive_name:
            matches.append(fields[0])
    if matches != [expected]:
        raise ProtocolError(
            f"publisher checksum entry for {archive_name} did not uniquely equal the frozen checksum"
        )


def validate_tokenizer_file_hashes(manifest: Mapping[str, Any], tokenizer: Any) -> None:
    root = Path(str(getattr(tokenizer, "name_or_path", "")))
    if not root.exists():
        raise ProtocolError("loaded tokenizer does not expose its immutable snapshot directory")
    for record in manifest["model"]["tokenizer_files"]:
        path = root / str(record["path"])
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            raise ProtocolError(f"frozen tokenizer file is missing or has a different size: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise ProtocolError(f"frozen tokenizer file hash changed: {record['path']}")


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def validate_data_manifest(
    config: StudyConfig,
    manifest: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    validate_manifest_hash(manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("study_id") != STUDY_ID:
        raise ProtocolError("manifest schema/study identifier is wrong")
    model_id = str(manifest["model"]["model_id"])
    if model_id not in (config.model["primary_id"], config.model["fallback_id"]):
        raise ProtocolError("manifest model is not a frozen primary/fallback checkpoint")
    if len(str(manifest["model"]["resolved_revision"])) != 40:
        raise ProtocolError("manifest does not contain an immutable model revision")
    if str(manifest["model"]["resolved_revision"]) != str(config.model["revisions"][model_id]):
        raise ProtocolError("manifest model revision differs from the model-specific frozen revision")
    if manifest["source"]["archive_sha256"] != config.source["expected_sha256"]:
        raise ProtocolError("manifest archive checksum differs from the frozen source")
    validate_tokenizer_file_hashes(manifest, tokenizer)

    positions = manifest.get("positions", {})
    if not isinstance(positions, dict) or not positions:
        raise ProtocolError("manifest has no selected positions")
    seen_prefixes: set[tuple[str, ...]] = set()
    seen_games: set[str] = set()
    target_to_positions: dict[str, list[str]] = defaultdict(list)
    token_lengths: list[int] = []
    for position_id, row in positions.items():
        if position_id != row.get("position_id"):
            raise ProtocolError("position dictionary key differs from position_id")
        moves = tuple(str(value) for value in row["moves"])
        if moves in seen_prefixes:
            raise ProtocolError("a selected move prefix is repeated")
        seen_prefixes.add(moves)
        if row["game_id"] in seen_games:
            raise ProtocolError("a selected game is reused across evaluation/training positions")
        seen_games.add(str(row["game_id"]))
        if str(row.get("source_game_hash", ""))[:24] != str(row["game_id"]):
            raise ProtocolError("position game ID differs from its full-game hash")
        bucket = int(row.get("partition_bucket", -1))
        expected_split = "pilot" if bucket <= 9 else "train" if bucket <= 59 else "development" if bucket <= 74 else "test"
        if bucket < 0 or bucket > 99 or expected_split != row["split"]:
            raise ProtocolError("position partition does not match its frozen full-game hash bucket")
        board = strict_board_from_moves(moves)
        if six_field_fen(board) != row["fen"] or board_targets(board) != row["targets"]:
            raise ProtocolError(f"verifier replay disagrees with saved labels for {position_id}")
        if target_key(row["targets"]) != row["target_key"]:
            raise ProtocolError(f"target key disagrees with labels for {position_id}")
        if sorted(move.uci() for move in board.legal_moves) != row["legal_moves"]:
            raise ProtocolError(f"legal move set disagrees with replay for {position_id}")
        context_ids, query_ids, all_ids = tokenize_context_query(tokenizer, str(row["context_text"]))
        if context_ids != row["context_token_ids"] or query_ids != row["query_token_ids"]:
            raise ProtocolError(f"tokenizer revision disagrees with position {position_id}")
        if all_ids != row["context_query_token_ids"]:
            raise ProtocolError(f"context/query token IDs disagree for {position_id}")
        if len(all_ids) != int(row["total_token_length"]):
            raise ProtocolError(f"saved token length is wrong for {position_id}")
        if len(all_ids) + 16 > int(config.model["max_tokens"]):
            raise ProtocolError(f"selected position {position_id} exceeds the frozen safe token budget")
        target_to_positions[str(row["target_key"])].append(str(position_id))
        token_lengths.append(len(all_ids))

    intentional_duplicate_sets = {
        frozenset((triplet["a_position_id"], triplet["aprime_position_id"]))
        for triplet in manifest.get("transposition_triplets", [])
    }
    for target, ids in target_to_positions.items():
        if len(ids) > 1 and frozenset(ids) not in intentional_duplicate_sets:
            raise ProtocolError(f"selected target state collision outside a transposition triplet: {target}")

    train_ids = [str(value) for value in manifest.get("train_position_ids", [])]
    if len(train_ids) != int(config.counts["train_states"]) or len(set(train_ids)) != len(train_ids):
        raise ProtocolError("training position count/uniqueness differs from the frozen plan")
    if any(positions[position_id]["split"] != "train" for position_id in train_ids):
        raise ProtocolError("training IDs include a non-training partition")

    continuation_target_keys: set[str] = set()
    pair_counts: dict[str, int] = {}
    for split, key in (
        ("pilot", "pilot_pairs"),
        ("development", "development_pairs"),
        ("test", "test_pairs"),
    ):
        pairs = list(manifest.get(key, []))
        pair_counts[key] = len(pairs)
        for pair in pairs:
            a = positions[pair["a_position_id"]]
            b = positions[pair["b_position_id"]]
            if a["split"] != split or b["split"] != split:
                raise ProtocolError(f"{pair['pair_id']} crosses data partitions")
            if pair_match_key(a) != pair_match_key(b):
                raise ProtocolError(f"{pair['pair_id']} violates the exact matching contract")
            if int(pair["changed_square_count"]) < 2 or state_difference_count(a, b) != int(pair["changed_square_count"]):
                raise ProtocolError(f"{pair['pair_id']} has an invalid board-state contrast")
            if pair["a_move"] not in a["legal_moves"] or pair["a_move"] in b["legal_moves"]:
                raise ProtocolError(f"{pair['pair_id']} A candidate is not A-exclusive")
            if pair["b_move"] not in b["legal_moves"] or pair["b_move"] in a["legal_moves"]:
                raise ProtocolError(f"{pair['pair_id']} B candidate is not B-exclusive")
            for move, saved_ids in (
                (pair["a_move"], pair["a_candidate_ids"]),
                (pair["b_move"], pair["b_candidate_ids"]),
            ):
                ids_a = candidate_continuation_ids(tokenizer, a["context_text"], move)
                ids_b = candidate_continuation_ids(tokenizer, b["context_text"], move)
                if ids_a != ids_b or ids_a != saved_ids:
                    raise ProtocolError(f"{pair['pair_id']} candidate token IDs are not context invariant")
            if len(pair["a_candidate_ids"]) != len(pair["b_candidate_ids"]):
                raise ProtocolError(f"{pair['pair_id']} candidate continuation lengths differ")
            if "continuation" in pair:
                continuation = pair["continuation"]
                suffix = list(continuation["moves"])
                if len(suffix) != 4:
                    raise ProtocolError(f"{pair['pair_id']} continuation does not contain four plies")
                replayed_a = {h: advance_position(a, suffix[:h], tokenizer) for h in (1, 2, 4)}
                replayed_b = {h: advance_position(b, suffix[:h], tokenizer) for h in (1, 2, 4)}
                for horizon in (1, 2, 4):
                    saved = continuation["horizons"][str(horizon)]
                    for donor, expected in (("a", replayed_a[horizon]), ("b", replayed_b[horizon])):
                        for field_name in ("moves", "advance_token_ids", "fen", "targets", "target_key", "legal_moves"):
                            if saved[donor][field_name] != expected[field_name]:
                                raise ProtocolError(
                                    f"{pair['pair_id']} continuation {horizon}/{donor} has invalid {field_name}"
                                )
                        key_value = str(saved[donor]["target_key"])
                        if key_value in target_to_positions or key_value in continuation_target_keys:
                            raise ProtocolError("continuation target state collides with another selected target")
                        continuation_target_keys.add(key_value)
                    candidates = saved["candidates"]
                    advanced_a = saved["a"]
                    advanced_b = saved["b"]
                    if candidates["a_move"] not in advanced_a["legal_moves"] or candidates["a_move"] in advanced_b["legal_moves"]:
                        raise ProtocolError("continuation A candidate is not donor-exclusive")
                    if candidates["b_move"] not in advanced_b["legal_moves"] or candidates["b_move"] in advanced_a["legal_moves"]:
                        raise ProtocolError("continuation B candidate is not donor-exclusive")

    transpositions = list(manifest.get("transposition_triplets", []))
    for triplet in transpositions:
        a = positions[triplet["a_position_id"]]
        aprime = positions[triplet["aprime_position_id"]]
        b = positions[triplet["b_position_id"]]
        if not (a["split"] == aprime["split"] == b["split"] == "test"):
            raise ProtocolError("transposition triplet is not fully in the test partition")
        if a["fen"] != aprime["fen"] or a["moves"] == aprime["moves"]:
            raise ProtocolError("transposition A/A' does not contain distinct histories with the same six-field FEN")
        if a["cutoff_ply"] != aprime["cutoff_ply"] or a["context_token_length"] != aprime["context_token_length"]:
            raise ProtocolError("transposition A/A' differs in ply or token length")
        if pair_match_key(a) != pair_match_key(b) or state_difference_count(a, b) < 2:
            raise ProtocolError("transposition B is not an exact matched different-position control")
        same_distance = token_hamming_distance(a["context_token_ids"], aprime["context_token_ids"])
        different_distance = token_hamming_distance(a["context_token_ids"], b["context_token_ids"])
        if same_distance != triplet["same_position_token_hamming"] or different_distance != triplet["different_position_token_hamming"]:
            raise ProtocolError("transposition token distances disagree with the manifest")
        if abs(same_distance - different_distance) > 2:
            raise ProtocolError("transposition token-distance matching exceeds the frozen tolerance")
        for move, saved_ids in (
            (triplet["a_move"], triplet["a_candidate_ids"]),
            (triplet["b_move"], triplet["b_candidate_ids"]),
        ):
            contextual = [candidate_continuation_ids(tokenizer, row["context_text"], move) for row in (a, aprime, b)]
            if any(ids != saved_ids for ids in contextual):
                raise ProtocolError("transposition candidate token IDs differ across contexts")

    minimums = {"development_pairs": 25, "test_pairs": 50}
    for key, minimum in minimums.items():
        if pair_counts[key] < minimum:
            raise ProtocolError(f"{key} has only {pair_counts[key]} examples; minimum is {minimum}")

    fixture = manifest["resource_fixture"]
    fixture_board = strict_board_from_moves(fixture["moves"])
    if fixture["resource_candidate_move"] not in [move.uci() for move in fixture_board.legal_moves]:
        raise ProtocolError("resource fixture candidate is not legal")
    fixture_ids = candidate_continuation_ids(
        tokenizer, fixture["context_text"], fixture["resource_candidate_move"]
    )
    if fixture_ids != fixture["resource_candidate_ids"]:
        raise ProtocolError("resource fixture candidate tokenization changed")
    if int(fixture["planned_workload_tokens"]) > int(config.model["max_tokens"]):
        raise ProtocolError("resource fixture exceeds the maximum planned workload")

    selected_rows = list(positions.values())
    audit = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "validated_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "position_count": len(positions),
        "unique_game_count": len(seen_games),
        "train_states": len(train_ids),
        **pair_counts,
        "continuation_pairs": sum("continuation" in pair for pair in manifest["test_pairs"]),
        "transposition_triplets": len(transpositions),
        "transposition_status": "available" if len(transpositions) >= 20 else "unavailable_or_underpowered",
        "ply_distribution": _distribution(row["cutoff_ply"] for row in selected_rows),
        "side_to_move_distribution": _distribution(row["side_to_move"] for row in selected_rows),
        "material_distribution": _distribution(row["piece_count"] for row in selected_rows),
        "token_length": {
            "minimum": min(token_lengths),
            "median": float(np.median(token_lengths)),
            "maximum": max(token_lengths),
        },
        "changed_square_distribution": _distribution(
            pair["changed_square_count"]
            for key in ("pilot_pairs", "development_pairs", "test_pairs")
            for pair in manifest[key]
        ),
        "older_state_subgroup": {
            key: sum(bool(pair["older_state_subgroup"]) for pair in manifest[key])
            for key in ("pilot_pairs", "development_pairs", "test_pairs")
        },
        "shared_final_two_moves": {
            key: sum(bool(pair["shared_final_two_moves"]) for pair in manifest[key])
            for key in ("pilot_pairs", "development_pairs", "test_pairs")
        },
        "resource_fixture_tokens": int(fixture["planned_workload_tokens"]),
        "exclusions": dict(manifest.get("exclusions", {})),
        "all_checks_passed": True,
    }
    return audit


def cmd_prepare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    manifest_path = manifest_path_for_args(args)
    existing = require_new_or_resume(manifest_path, args.resume)
    if existing is not None:
        manifest = load_manifest(manifest_path)
        requested_model = str(
            config.model["fallback_id"] if args.use_fallback else config.model["primary_id"]
        )
        if str(manifest["model"]["model_id"]) != requested_model:
            raise ProtocolError(
                "the existing manifest belongs to a different primary/fallback model; "
                "preserve it and pass a new --manifest path for the requested model"
            )
        tokenizer = resolve_manifest_tokenizer(manifest, args.allow_remote_downloads)
        audit = validate_data_manifest(config, manifest, tokenizer)
        atomic_write_json(run_dir / "manifest.json", manifest)
        atomic_write_json(run_dir / "data_audit.json", audit)
        atomic_write_json(run_dir / "environment.json", environment_payload())
        log_time(run_dir, "prepare", started, "reused packaged frozen manifest")
        print(
            json.dumps(
                {
                    "reused": str(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "audit": audit,
                },
                indent=2,
            )
        )
        return

    if args.use_fallback:
        if not args.fallback_reason or not Path(args.fallback_reason).exists():
            raise ProtocolError("fallback prepare requires --fallback-reason pointing to a preserved 4B resource failure")
        model_id = str(config.model["fallback_id"])
    else:
        model_id = str(config.model["primary_id"])
    revision, tokenizer_snapshot, tokenizer_files = resolve_hf_snapshot(
        model_id, str(config.model["revisions"][model_id]), args.allow_remote_downloads
    )
    tokenizer = load_tokenizer(tokenizer_snapshot)

    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_dir / Path(str(config.source["url"])).name
    checksum_path = source_dir / "sha256sums.txt"
    if not archive_path.exists():
        if not args.allow_remote_downloads:
            raise ProtocolError("Lichess archive is absent; prepare needs --allow-remote-downloads")
        download_file(str(config.source["url"]), archive_path)
    if not checksum_path.exists():
        if not args.allow_remote_downloads:
            raise ProtocolError("publisher checksum list is absent; prepare needs --allow-remote-downloads")
        download_file(str(config.source["checksum_url"]), checksum_path)
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != config.source["expected_sha256"]:
        raise ProtocolError("downloaded Lichess archive does not match the frozen SHA256")
    verify_publisher_checksum(checksum_path, archive_path.name, str(config.source["expected_sha256"]))

    scan_limit = int(config.source["initial_scan_games"])
    try:
        manifest = build_manifest(
            config,
            archive_path,
            archive_sha256,
            model_id,
            revision,
            tokenizer_snapshot,
            tokenizer_files,
            tokenizer,
            scan_limit,
        )
        counts = manifest["actual_counts"]
        required = {
            "pilot_pairs": int(config.counts["pilot_pairs"]),
            "train_states": int(config.counts["train_states"]),
            "development_pairs": int(config.counts["development_pairs"]),
            "test_pairs": int(config.counts["test_pairs"]),
            "continuation_pairs": int(config.counts["continuation_pairs"]),
            "transposition_triplets": int(config.counts["transposition_triplets"]),
        }
        if any(int(counts[key]) < value for key, value in required.items()):
            raise ProtocolError("initial fixed scan did not fill structural quotas")
    except ProtocolError as first_error:
        with (run_dir / "decisions.md").open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n- {utc_now()}: Initial {scan_limit}-game scan could not fill the frozen quotas "
                f"({first_error}). Expanded only to the remainder of the same January 2013 archive.\n"
            )
        manifest = build_manifest(
            config,
            archive_path,
            archive_sha256,
            model_id,
            revision,
            tokenizer_snapshot,
            tokenizer_files,
            tokenizer,
            None,
        )
    audit = validate_data_manifest(config, manifest, tokenizer)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "data_audit.json", audit)
    atomic_write_json(run_dir / "environment.json", environment_payload())
    if args.use_fallback:
        shutil.copy2(args.fallback_reason, run_dir / "four_b_resource_failure.json")
    log_time(run_dir, "prepare", started, f"manifest={manifest['manifest_sha256']}")
    print(json.dumps({"manifest": str(manifest_path), "audit": audit}, indent=2))


def cmd_validate_data(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    manifest = load_manifest(manifest_path_for_args(args))
    tokenizer = resolve_manifest_tokenizer(manifest, args.allow_remote_downloads)
    audit = validate_data_manifest(config, manifest, tokenizer)
    existing = require_new_or_resume(run_dir / "data_audit.json", args.resume)
    if existing is not None and existing.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ProtocolError("resumed data audit belongs to a different manifest")
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "data_audit.json", audit)
    if not (run_dir / "environment.json").exists():
        atomic_write_json(run_dir / "environment.json", environment_payload())
    log_time(run_dir, "validate-data", started)
    print(json.dumps(audit, indent=2))


# ---------------------------------------------------------------------------
# Engineering preflight
# ---------------------------------------------------------------------------


def relative_l2(first: np.ndarray | torch.Tensor, second: np.ndarray | torch.Tensor) -> float:
    left = np.asarray(first.detach().cpu() if isinstance(first, torch.Tensor) else first, dtype=np.float64)
    right = np.asarray(second.detach().cpu() if isinstance(second, torch.Tensor) else second, dtype=np.float64)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12))


def max_abs_difference(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        return math.inf
    return max((abs(float(a) - float(b)) for a, b in zip(first, second)), default=0.0)


def pair_position_rows(manifest: Mapping[str, Any], pair: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = manifest["positions"]
    return dict(positions[pair["a_position_id"]]), dict(positions[pair["b_position_id"]])


def preflight_fixture_scores(
    backend: FrozenQwenBackend,
    position: Mapping[str, Any],
    candidate_ids: Sequence[int],
) -> dict[str, Any]:
    snapshot = backend.prefill(position["context_token_ids"])
    digest_before = snapshot_digest(snapshot)
    cached_feature = backend.read_feature(snapshot, position["query_token_ids"])
    cached_score = backend.score_candidate(snapshot, position["query_token_ids"], candidate_ids)
    digest_after = snapshot_digest(snapshot)
    fresh_feature = backend.fresh_feature(position["context_token_ids"], position["query_token_ids"])
    fresh_score = backend.fresh_score_candidate(
        position["context_token_ids"], position["query_token_ids"], candidate_ids
    )
    return {
        "position_id": position["position_id"],
        "candidate_ids": list(candidate_ids),
        "snapshot_digest_before": digest_before,
        "snapshot_digest_after": digest_after,
        "input_snapshot_unchanged": digest_before == digest_after,
        "hidden_relative_l2": relative_l2(cached_feature, fresh_feature),
        "cached": cached_score,
        "fresh": fresh_score,
        "argmax_matches": [
            int(left == right) for left, right in zip(cached_score["argmax_ids"], fresh_score["argmax_ids"])
        ],
        "absolute_logprob_differences": [
            abs(float(left) - float(right))
            for left, right in zip(cached_score["token_logprobs"], fresh_score["token_logprobs"])
        ],
    }


def _score_pair_in_order(
    backend: FrozenQwenBackend,
    manifest: Mapping[str, Any],
    pair: Mapping[str, Any],
    donor_order: Sequence[str],
    condition_order: Sequence[str],
    candidate_order: Sequence[str] = ("a", "b"),
) -> dict[str, float]:
    a, b = pair_position_rows(manifest, pair)
    snapshots: dict[str, CacheSnapshot] = {}
    for donor in donor_order:
        row = a if donor == "A" else b
        snapshots[donor] = backend.prefill(row["context_token_ids"])
    conditions = {
        "AA": lambda: assemble_snapshot(snapshots["A"], snapshots["A"]),
        "BB": lambda: assemble_snapshot(snapshots["B"], snapshots["B"]),
        "BA": lambda: assemble_snapshot(snapshots["B"], snapshots["A"]),
        "AB": lambda: assemble_snapshot(snapshots["A"], snapshots["B"]),
    }
    out: dict[str, float] = {}
    query_ids = a["query_token_ids"]
    for condition in condition_order:
        snapshot = conditions[condition]()
        candidate_map = {"a": pair["a_candidate_ids"], "b": pair["b_candidate_ids"]}
        for name in candidate_order:
            candidate_ids = candidate_map[name]
            result = backend.score_candidate(snapshot, query_ids, candidate_ids)
            out[f"{condition}:{name}"] = float(result["sum_logprob"])
    return out


def cmd_preflight(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    output_path = run_dir / "preflight.json"
    reused = require_new_or_resume(output_path, args.resume)
    if reused is not None:
        print(json.dumps(reused, indent=2))
        return
    manifest = load_manifest(manifest_path_for_args(args))
    tokenizer = resolve_manifest_tokenizer(manifest, args.allow_remote_downloads)
    data_audit = validate_data_manifest(config, manifest, tokenizer)
    atomic_write_json(run_dir / "data_audit.json", data_audit)
    backend, runtime = load_backend(config, manifest, args.allow_remote_downloads)
    torch.cuda.reset_peak_memory_stats(backend.device)
    torch.cuda.synchronize(backend.device)
    model_digest_before = full_parameter_digest(backend.full_model)
    parameters_frozen = all(not parameter.requires_grad for parameter in backend.full_model.parameters())
    no_existing_gradients = all(parameter.grad is None for parameter in backend.full_model.parameters())
    evaluation_mode = not backend.full_model.training and not backend.language_model.training and not backend.lm_head.training

    pilot_pairs = list(manifest["pilot_pairs"])
    if not pilot_pairs:
        raise ProtocolError("preflight requires at least one frozen pilot pair")
    first_pair = pilot_pairs[0]
    first_a, first_b = pair_position_rows(manifest, first_pair)
    source_snapshot = backend.prefill(first_a["context_token_ids"])
    source_schema = validate_snapshot(source_snapshot)
    copied_snapshot = clone_snapshot(source_snapshot)
    assert_storage_independent(source_snapshot, copied_snapshot)
    copy_digest_equal = snapshot_digest(source_snapshot) == snapshot_digest(copied_snapshot)
    identity = assemble_snapshot(source_snapshot, source_snapshot)
    identity_digest_equal = snapshot_digest(source_snapshot) == snapshot_digest(identity)

    fixture_path = run_dir / "cache_restore_fixture.pt"
    save_snapshot_fixture(fixture_path, source_snapshot)
    restored_cpu = load_snapshot_fixture(fixture_path)
    serialize_digest_equal = snapshot_digest(source_snapshot) == snapshot_digest(restored_cpu)
    repeated_first = backend.score_candidate(
        source_snapshot, first_a["query_token_ids"], first_pair["a_candidate_ids"]
    )
    repeated_copy = backend.score_candidate(
        copied_snapshot, first_a["query_token_ids"], first_pair["a_candidate_ids"]
    )
    repeated_restored = backend.score_candidate(
        restored_cpu, first_a["query_token_ids"], first_pair["a_candidate_ids"]
    )
    repeat_max_error = max(
        max_abs_difference(repeated_first["token_logprobs"], repeated_copy["token_logprobs"]),
        max_abs_difference(repeated_first["token_logprobs"], repeated_restored["token_logprobs"]),
    )

    language_hidden, language_logits = backend.language_last_logits(first_a["context_query_token_ids"])
    reference_logits = backend.reference_last_logits(first_a["context_query_token_ids"])
    output_head_error = relative_l2(language_logits, reference_logits)
    output_head_argmax_equal = int(torch.argmax(language_logits).item()) == int(torch.argmax(reference_logits).item())
    finite_logits = bool(torch.isfinite(language_logits).all() and torch.isfinite(reference_logits).all())
    finite_hidden = bool(torch.isfinite(language_hidden).all())

    fixture_results: list[dict[str, Any]] = []
    for pair in pilot_pairs[: min(4, len(pilot_pairs))]:
        a, b = pair_position_rows(manifest, pair)
        fixture_results.append(preflight_fixture_scores(backend, a, pair["a_candidate_ids"]))
        fixture_results.append(preflight_fixture_scores(backend, b, pair["b_candidate_ids"]))
    argmax_values = [value for row in fixture_results for value in row["argmax_matches"]]
    hidden_errors = [float(row["hidden_relative_l2"]) for row in fixture_results]
    logprob_errors = [value for row in fixture_results for value in row["absolute_logprob_differences"]]
    cached_fresh = {
        "fixture_count": len(fixture_results),
        "next_token_argmax_agreement": float(np.mean(argmax_values)) if argmax_values else math.nan,
        "hidden_relative_l2_mean": float(np.mean(hidden_errors)),
        "hidden_relative_l2_max": float(np.max(hidden_errors)),
        "candidate_logprob_absolute_difference_mean": float(np.mean(logprob_errors)),
        "candidate_logprob_absolute_difference_max": float(np.max(logprob_errors)),
        "fixtures": fixture_results,
    }

    forward_order = _score_pair_in_order(
        backend, manifest, first_pair, ("A", "B"), ("AA", "BB", "BA", "AB")
    )
    reverse_order = _score_pair_in_order(
        backend, manifest, first_pair, ("B", "A"), ("AB", "BA", "BB", "AA"), ("b", "a")
    )
    order_max_error = max(abs(forward_order[key] - reverse_order[key]) for key in forward_order)

    shared_legal = sorted(set(first_a["legal_moves"]) & set(first_b["legal_moves"]))
    matched_advance = None
    for move in shared_legal:
        a_ids = advance_suffix_ids(backend.tokenizer, first_a["moves"], move)
        b_ids = advance_suffix_ids(backend.tokenizer, first_b["moves"], move)
        if a_ids == b_ids:
            matched_advance = (move, a_ids)
            break
    if matched_advance is None:
        raise ProtocolError(
            "pilot pair has no common legal move with identical contextual token IDs for the sequence-length check"
        )
    advance_move, advance_ids = matched_advance
    b_source_snapshot = backend.prefill(first_b["context_token_ids"])
    if b_source_snapshot.sequence_length != source_snapshot.sequence_length:
        raise ProtocolError("pilot donor snapshots differ in sequence length")
    crossed_snapshot = assemble_snapshot(source_snapshot, b_source_snapshot)
    advanced_snapshots = {
        "A": backend.advance(source_snapshot, advance_ids),
        "B": backend.advance(b_source_snapshot, advance_ids),
        "AB": backend.advance(crossed_snapshot, advance_ids),
    }
    expected_advanced_length = source_snapshot.sequence_length + len(advance_ids)
    continuation_length_pass = all(
        snapshot.sequence_length == expected_advanced_length
        and snapshot.next_position == expected_advanced_length
        and snapshot.attention_mask_length == expected_advanced_length
        for snapshot in advanced_snapshots.values()
    )

    fixture = manifest["resource_fixture"]
    resource_started = time.monotonic()
    resource_snapshot = backend.prefill(fixture["context_token_ids"])
    resource_clone = clone_snapshot(resource_snapshot)
    assert_storage_independent(resource_snapshot, resource_clone)
    _ = assemble_snapshot(resource_snapshot, resource_clone)
    resource_score = backend.score_candidate(
        resource_snapshot, fixture["query_token_ids"], fixture["resource_candidate_ids"]
    )
    torch.cuda.synchronize(backend.device)
    resource_seconds = time.monotonic() - resource_started
    peak_allocated = int(torch.cuda.max_memory_allocated(backend.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(backend.device))

    model_digest_after = full_parameter_digest(backend.full_model)
    no_gradients_after = all(parameter.grad is None for parameter in backend.full_model.parameters())
    expected_hidden = config.model.get("expected_hidden_sizes", {}).get(manifest["model"]["model_id"])
    expected_layers = config.model.get("expected_layer_counts", {}).get(manifest["model"]["model_id"])
    architecture_pass = (
        (expected_hidden is None or int(expected_hidden) == backend.hidden_size)
        and (expected_layers is None or int(expected_layers) == len(backend.layer_types))
        and set(backend.layer_types) == {"linear_attention", "full_attention"}
    )
    gates = {
        "frozen_eval_finite": parameters_frozen
        and no_existing_gradients
        and no_gradients_after
        and evaluation_mode
        and finite_logits
        and finite_hidden,
        "architecture": architecture_pass,
        "output_head_reproduction": output_head_error <= 1e-6 and output_head_argmax_equal,
        "snapshot_copy_identity_restore": copy_digest_equal
        and identity_digest_equal
        and serialize_digest_equal
        and repeat_max_error <= 1e-5,
        "fresh_cached_equivalence": cached_fresh["next_token_argmax_agreement"] >= 0.95
        and cached_fresh["hidden_relative_l2_max"] <= 0.01
        and cached_fresh["candidate_logprob_absolute_difference_mean"] <= 0.05,
        "cache_poisoning_and_order": order_max_error <= 1e-5 and continuation_length_pass,
        "token_and_data_contracts": bool(data_audit["all_checks_passed"]),
        "resource_workload": int(fixture["planned_workload_tokens"]) <= int(config.model["max_tokens"])
        and math.isfinite(float(resource_score["sum_logprob"])),
        "model_immutability": model_digest_before == model_digest_after,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_manifest": runtime,
        "model_parameter_digest_before": model_digest_before,
        "model_parameter_digest_after": model_digest_after,
        "source_cache_schema": source_schema,
        "copy_digest_equal": copy_digest_equal,
        "identity_reassembly_digest_equal": identity_digest_equal,
        "serialize_restore_digest_equal": serialize_digest_equal,
        "repeat_max_logprob_error_nats": repeat_max_error,
        "output_head_relative_l2": output_head_error,
        "output_head_argmax_equal": output_head_argmax_equal,
        "fresh_cached": cached_fresh,
        "order_check": {
            "forward": forward_order,
            "reverse": reverse_order,
            "maximum_absolute_error_nats": order_max_error,
            "sequence_length_check_move": advance_move,
            "sequence_length_check_token_ids": advance_ids,
            "sequence_length_check_conditions": sorted(advanced_snapshots),
            "post_continuation_sequence_length_pass": continuation_length_pass,
        },
        "resource_test": {
            "position_id": fixture["position_id"],
            "planned_workload_tokens": fixture["planned_workload_tokens"],
            "runtime_seconds": resource_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "candidate_score": resource_score,
            "cache_schema": validate_snapshot(resource_snapshot),
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    atomic_write_json(output_path, payload)
    try:
        fixture_path.unlink()
    except FileNotFoundError:
        pass
    atomic_write_json(run_dir / "environment.json", environment_payload())
    log_time(run_dir, "preflight", started, f"passed={payload['all_gates_passed']}")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Feature extraction and linear readouts
# ---------------------------------------------------------------------------


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def feature_ids_for_split(manifest: Mapping[str, Any], split: str) -> list[str]:
    if split == "train":
        return [str(value) for value in manifest["train_position_ids"]]
    if split != "development":
        raise ProtocolError("features may only be extracted for train/development before freeze")
    return sorted(
        {
            str(pair[key])
            for pair in manifest["development_pairs"]
            for key in ("a_position_id", "b_position_id")
        }
    )


def extract_feature_split(
    backend: FrozenQwenBackend,
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, np.ndarray]:
    identifiers = feature_ids_for_split(manifest, split)
    model_features: list[np.ndarray] = []
    orderless_features: list[np.ndarray] = []
    last_move_features: list[np.ndarray] = []
    targets: list[list[int]] = []
    for index, position_id in enumerate(identifiers, 1):
        row = manifest["positions"][position_id]
        snapshot = backend.prefill(row["context_token_ids"])
        model_features.append(backend.read_feature(snapshot, row["query_token_ids"]))
        orderless, last_move = backend.input_embedding_features(row)
        orderless_features.append(orderless)
        last_move_features.append(last_move)
        targets.append([int(value) for value in row["targets"]])
        if index % 25 == 0:
            print(f"[{split}] extracted {index}/{len(identifiers)}", flush=True)
    return {
        "position_ids": np.asarray(identifiers, dtype="U64"),
        "model_features": np.asarray(model_features, dtype=np.float32),
        "orderless_features": np.asarray(orderless_features, dtype=np.float32),
        "last_move_features": np.asarray(last_move_features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.int16),
    }


def validate_feature_file(
    path: Path,
    expected_ids: Sequence[str],
    hidden_size: int | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"position_ids", "model_features", "orderless_features", "last_move_features", "targets"}
        if set(data.files) != required:
            raise ProtocolError(f"feature file has wrong arrays: {path}")
        ids = [str(value) for value in data["position_ids"].tolist()]
        if ids != list(expected_ids) or len(ids) != len(set(ids)):
            raise ProtocolError(f"feature IDs are incomplete, reordered, or duplicated: {path}")
        n = len(ids)
        targets = data["targets"]
        feature_arrays = [data["model_features"], data["orderless_features"], data["last_move_features"]]
        if targets.shape != (n, 70):
            raise ProtocolError(f"target shape is wrong in {path}: {targets.shape}")
        if any(array.ndim != 2 or array.shape[0] != n for array in feature_arrays):
            raise ProtocolError(f"feature matrix shape is wrong in {path}")
        if hidden_size is not None and any(array.shape[1] != hidden_size for array in feature_arrays):
            raise ProtocolError(f"feature width differs from the loaded hidden size in {path}")
        if any(not np.isfinite(array).all() for array in feature_arrays):
            raise ProtocolError(f"non-finite feature found in {path}")
        return {
            "sha256": sha256_file(path),
            "rows": n,
            "model_feature_shape": list(data["model_features"].shape),
            "orderless_feature_shape": list(data["orderless_features"].shape),
            "last_move_feature_shape": list(data["last_move_features"].shape),
            "target_shape": list(targets.shape),
        }


def cmd_extract_features(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    preflight = read_json(run_dir / "preflight.json")
    if not preflight.get("all_gates_passed"):
        raise ProtocolError("feature extraction refuses because real-model Gate A did not pass")
    manifest = load_manifest(manifest_path_for_args(args))
    if preflight.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ProtocolError("preflight and manifest hashes differ")
    features_dir = run_dir / "features"
    train_path = features_dir / "train.npz"
    development_path = features_dir / "development.npz"
    existing = train_path.exists() or development_path.exists()
    if existing and not args.resume:
        raise ProtocolError("feature artifacts already exist; pass --resume to verify and reuse")
    if existing and not (train_path.exists() and development_path.exists()):
        raise ProtocolError("only one feature split exists; preserve the run and restart in a new directory")
    if existing:
        train_meta = validate_feature_file(train_path, feature_ids_for_split(manifest, "train"))
        development_meta = validate_feature_file(
            development_path, feature_ids_for_split(manifest, "development")
        )
        print(json.dumps({"reused": True, "train": train_meta, "development": development_meta}, indent=2))
        return

    backend, runtime = load_backend(config, manifest, args.allow_remote_downloads)
    if runtime["resolved_revision"] != preflight["runtime_manifest"]["resolved_revision"]:
        raise ProtocolError("loaded model revision differs from the passed preflight")
    parameter_digest_before = full_parameter_digest(backend.full_model)
    if parameter_digest_before != preflight["model_parameter_digest_after"]:
        raise ProtocolError("loaded model parameters differ from the preflight")
    train_arrays = extract_feature_split(backend, manifest, "train")
    development_arrays = extract_feature_split(backend, manifest, "development")
    atomic_savez(train_path, **train_arrays)
    atomic_savez(development_path, **development_arrays)
    parameter_digest_after = full_parameter_digest(backend.full_model)
    if parameter_digest_before != parameter_digest_after:
        raise ProtocolError("frozen model changed during feature extraction")
    train_meta = validate_feature_file(
        train_path, feature_ids_for_split(manifest, "train"), backend.hidden_size
    )
    development_meta = validate_feature_file(
        development_path, feature_ids_for_split(manifest, "development"), backend.hidden_size
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_manifest": runtime,
        "parameter_digest_before": parameter_digest_before,
        "parameter_digest_after": parameter_digest_after,
        "feature_contract": "final normalized language-model activation at the last fixed query token",
        "orderless_contract": "mean frozen input embeddings over contextual move-history token spans",
        "last_move_contract": "mean frozen input embeddings over contextual final-UCI-move token spans",
        "train": train_meta,
        "development": development_meta,
    }
    atomic_write_json(features_dir / "metadata.json", metadata)
    log_time(run_dir, "extract-features", started)
    print(json.dumps(metadata, indent=2))


def targets_to_one_hot(targets: np.ndarray) -> np.ndarray:
    if targets.ndim != 2 or targets.shape[1] != 70:
        raise ProtocolError("targets must have shape [rows, 70]")
    output = np.zeros((targets.shape[0], HEAD_OFFSETS[-1]), dtype=np.float64)
    for head, classes in enumerate(HEAD_CLASS_COUNTS):
        values = targets[:, head].astype(np.int64)
        if np.any(values < 0) or np.any(values >= classes):
            raise ProtocolError(f"target class outside range for head {head}")
        output[np.arange(targets.shape[0]), HEAD_OFFSETS[head] + values] = 1.0
    return output


@dataclass(frozen=True)
class RidgeReadout:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    ridge_lambda: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        standardized[:, self.feature_scale == 1.0e30] = 0.0
        return standardized @ self.weights + self.target_mean

    def predict(self, features: np.ndarray) -> np.ndarray:
        return decode_head_scores(self.scores(features))


def fit_ridge_readout(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    ridge_lambda: float,
    std_floor: float,
) -> RidgeReadout:
    x = np.asarray(train_features, dtype=np.float64)
    y = targets_to_one_hot(np.asarray(train_targets))
    feature_mean = x.mean(axis=0)
    raw_scale = x.std(axis=0)
    low_variance = raw_scale < std_floor
    feature_scale = raw_scale.copy()
    feature_scale[low_variance] = 1.0e30
    z = (x - feature_mean) / feature_scale
    z[:, low_variance] = 0.0
    target_mean = y.mean(axis=0)
    centered_y = y - target_mean
    gram = z @ z.T
    system = gram + float(ridge_lambda) * np.eye(z.shape[0], dtype=np.float64)
    try:
        dual = np.linalg.solve(system, centered_y)
    except np.linalg.LinAlgError as exc:
        raise ProtocolError(f"ridge solve failed for lambda={ridge_lambda}") from exc
    weights = z.T @ dual
    if not all(np.isfinite(value).all() for value in (feature_mean, feature_scale, target_mean, weights)):
        raise ProtocolError("ridge fitting produced non-finite parameters")
    return RidgeReadout(feature_mean, feature_scale, target_mean, weights, float(ridge_lambda))


def decode_head_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != HEAD_OFFSETS[-1]:
        raise ProtocolError("readout scores have the wrong width")
    predictions = np.zeros((values.shape[0], 70), dtype=np.int16)
    for head in range(70):
        predictions[:, head] = np.argmax(
            values[:, HEAD_OFFSETS[head] : HEAD_OFFSETS[head + 1]], axis=1
        )
    return predictions


def initial_board_targets() -> np.ndarray:
    return np.asarray(board_targets(import_chess().Board()), dtype=np.int16)


def row_accuracy_on_changed_squares(prediction: np.ndarray, target: np.ndarray) -> float:
    changed = target[:64] != initial_board_targets()[:64]
    if not np.any(changed):
        return math.nan
    return float(np.mean(prediction[:64][changed] == target[:64][changed]))


def macro_recall_supported(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, int]]:
    true = targets[:, :64].reshape(-1)
    predicted = predictions[:, :64].reshape(-1)
    recalls: list[float] = []
    support: dict[str, int] = {}
    for class_id, name in enumerate(PIECE_CLASSES):
        mask = true == class_id
        support[name] = int(mask.sum())
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == class_id)))
    return float(np.mean(recalls)) if recalls else math.nan, support


def probe_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape or predictions.shape[1] != 70:
        raise ProtocolError("prediction/target shape mismatch")
    initial = initial_board_targets()
    changed = targets[:, :64] != initial[None, :64]
    occupied = targets[:, :64] != PIECE_TO_INDEX["empty"]
    macro, support = macro_recall_supported(predictions, targets)
    rule_metrics = {}
    for head in range(64, 70):
        rule_metrics[HEAD_NAMES[head]] = {
            "accuracy": float(np.mean(predictions[:, head] == targets[:, head])),
            "support": _distribution(targets[:, head].tolist()),
        }
    return {
        "rows": int(targets.shape[0]),
        "square_accuracy": float(np.mean(predictions[:, :64] == targets[:, :64])),
        "macro_recall_supported_piece_classes": macro,
        "piece_class_support": support,
        "occupied_square_accuracy": float(np.mean(predictions[:, :64][occupied] == targets[:, :64][occupied]))
        if occupied.any()
        else None,
        "changed_from_initial_square_accuracy": float(
            np.mean(predictions[:, :64][changed] == targets[:, :64][changed])
        )
        if changed.any()
        else None,
        "exact_64_square_match": float(np.mean(np.all(predictions[:, :64] == targets[:, :64], axis=1))),
        "rule_heads": rule_metrics,
        "piece_and_rule_state_exact_match": float(np.mean(np.all(predictions == targets, axis=1))),
    }


def choose_readout(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    development_features: np.ndarray,
    development_targets: np.ndarray,
    ridge_grid: Sequence[float],
    std_floor: float,
) -> tuple[RidgeReadout, list[dict[str, Any]]]:
    candidates: list[tuple[float, float, float, RidgeReadout, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    for ridge_lambda in ridge_grid:
        readout = fit_ridge_readout(
            train_features, train_targets, float(ridge_lambda), float(std_floor)
        )
        predictions = readout.predict(development_features)
        metrics = probe_metrics(predictions, development_targets)
        changed = float(metrics["changed_from_initial_square_accuracy"])
        full = float(metrics["square_accuracy"])
        record = {"ridge_lambda": float(ridge_lambda), "development_metrics": metrics}
        records.append(record)
        candidates.append((changed, full, float(ridge_lambda), readout, record))
    # The last key is larger lambda, exactly matching the frozen tie-break.
    selected = max(candidates, key=lambda value: (value[0], value[1], value[2]))
    return selected[3], records


def bootstrap_mean_interval(
    values: Sequence[float],
    replicates: int = 10000,
    seed: int = 41,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "n_clusters": int(array.size),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def save_readout_arrays(prefix: str, readout: RidgeReadout, arrays: MutableMapping[str, Any]) -> None:
    arrays[f"{prefix}feature_mean"] = readout.feature_mean.astype(np.float64)
    arrays[f"{prefix}feature_scale"] = readout.feature_scale.astype(np.float64)
    arrays[f"{prefix}target_mean"] = readout.target_mean.astype(np.float64)
    arrays[f"{prefix}weights"] = readout.weights.astype(np.float64)
    arrays[f"{prefix}ridge_lambda"] = np.asarray([readout.ridge_lambda], dtype=np.float64)


def load_readout(path: Path, prefix: str = "") -> RidgeReadout:
    with np.load(path, allow_pickle=False) as data:
        return RidgeReadout(
            feature_mean=data[f"{prefix}feature_mean"].astype(np.float64),
            feature_scale=data[f"{prefix}feature_scale"].astype(np.float64),
            target_mean=data[f"{prefix}target_mean"].astype(np.float64),
            weights=data[f"{prefix}weights"].astype(np.float64),
            ridge_lambda=float(data[f"{prefix}ridge_lambda"][0]),
        )


def development_pair_values(
    manifest: Mapping[str, Any],
    ids: Sequence[str],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[float]:
    id_to_index = {str(position_id): index for index, position_id in enumerate(ids)}
    values: list[float] = []
    for pair in manifest["development_pairs"]:
        scores = []
        for key in ("a_position_id", "b_position_id"):
            index = id_to_index[str(pair[key])]
            scores.append(row_accuracy_on_changed_squares(predictions[index], targets[index]))
        finite = [value for value in scores if math.isfinite(value)]
        values.append(float(np.mean(finite)) if finite else math.nan)
    return values


def cmd_fit_probe(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    output_path = run_dir / "probe_config.json"
    main_weights_path = run_dir / "probe_weights.npz"
    baseline_weights_path = run_dir / "baseline_probe_weights.npz"
    if any(path.exists() for path in (output_path, main_weights_path, baseline_weights_path)):
        if not args.resume or not all(path.exists() for path in (output_path, main_weights_path, baseline_weights_path)):
            raise ProtocolError("probe artifacts already/partially exist; use --resume only for a complete set")
        payload = read_json(output_path)
        if payload["probe_weights_sha256"] != sha256_file(main_weights_path):
            raise ProtocolError("resumed main probe hash mismatch")
        if payload["baseline_probe_weights_sha256"] != sha256_file(baseline_weights_path):
            raise ProtocolError("resumed baseline probe hash mismatch")
        print(json.dumps(payload, indent=2))
        return
    manifest = load_manifest(manifest_path_for_args(args))
    train_path = run_dir / "features" / "train.npz"
    development_path = run_dir / "features" / "development.npz"
    validate_feature_file(train_path, feature_ids_for_split(manifest, "train"))
    validate_feature_file(development_path, feature_ids_for_split(manifest, "development"))
    with np.load(train_path, allow_pickle=False) as train_data, np.load(
        development_path, allow_pickle=False
    ) as development_data:
        train = {key: train_data[key].copy() for key in train_data.files}
        development = {key: development_data[key].copy() for key in development_data.files}
    ridge_grid = [float(value) for value in config.raw["probe"]["ridge_grid"]]
    std_floor = float(config.raw["probe"]["feature_std_floor"])
    fitted: dict[str, RidgeReadout] = {}
    searches: dict[str, Any] = {}
    for name, feature_key in (
        ("main", "model_features"),
        ("orderless", "orderless_features"),
        ("last_move", "last_move_features"),
    ):
        fitted[name], searches[name] = choose_readout(
            train[feature_key],
            train["targets"],
            development[feature_key],
            development["targets"],
            ridge_grid,
            std_floor,
        )
    permutation = np.random.default_rng(config.seed).permutation(train["targets"].shape[0])
    fitted["permutation"], searches["permutation"] = choose_readout(
        train["model_features"],
        train["targets"][permutation],
        development["model_features"],
        development["targets"],
        ridge_grid,
        std_floor,
    )
    frequency_prediction = np.asarray(
        [
            Counter(train["targets"][:, head].tolist()).most_common(1)[0][0]
            for head in range(70)
        ],
        dtype=np.int16,
    )
    initial_prediction = initial_board_targets()
    development_predictions: dict[str, np.ndarray] = {
        "main": fitted["main"].predict(development["model_features"]),
        "orderless": fitted["orderless"].predict(development["orderless_features"]),
        "last_move": fitted["last_move"].predict(development["last_move_features"]),
        "permutation": fitted["permutation"].predict(development["model_features"]),
        "training_frequency": np.repeat(frequency_prediction[None, :], development["targets"].shape[0], axis=0),
        "initial_board": np.repeat(initial_prediction[None, :], development["targets"].shape[0], axis=0),
    }
    metrics = {
        name: probe_metrics(prediction, development["targets"])
        for name, prediction in development_predictions.items()
    }
    ids = [str(value) for value in development["position_ids"].tolist()]
    pair_values = {
        name: development_pair_values(
            manifest, ids, prediction, development["targets"]
        )
        for name, prediction in development_predictions.items()
    }
    baseline_names = [
        "initial_board",
        "training_frequency",
        "orderless",
        "last_move",
        "permutation",
    ]
    strongest_baseline = max(
        baseline_names,
        key=lambda name: float(np.nanmean(np.asarray(pair_values[name], dtype=np.float64))),
    )
    improvements = [
        main - baseline
        for main, baseline in zip(pair_values["main"], pair_values[strongest_baseline])
    ]
    improvement_interval = bootstrap_mean_interval(
        improvements, int(config.raw["bootstrap_replicates"]), int(config.raw["bootstrap_seed"])
    )
    gate_c = (
        improvement_interval["estimate"] is not None
        and float(improvement_interval["estimate"]) >= float(config.gates["probe_minimum_improvement"])
        and float(improvement_interval["lower"]) > 0.0
    )

    main_arrays: dict[str, Any] = {
        "head_class_counts": np.asarray(HEAD_CLASS_COUNTS, dtype=np.int16),
        "head_offsets": np.asarray(HEAD_OFFSETS, dtype=np.int32),
    }
    save_readout_arrays("", fitted["main"], main_arrays)
    atomic_savez(main_weights_path, **main_arrays)
    baseline_arrays: dict[str, Any] = {
        "training_frequency_prediction": frequency_prediction,
        "initial_board_prediction": initial_prediction,
        "training_row_permutation": permutation.astype(np.int32),
    }
    for name in ("orderless", "last_move", "permutation"):
        save_readout_arrays(f"{name}_", fitted[name], baseline_arrays)
    atomic_savez(baseline_weights_path, **baseline_arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "train_features_sha256": sha256_file(train_path),
        "development_features_sha256": sha256_file(development_path),
        "feature_contract": "final normalized language-model activation at the last fixed query token",
        "target_contract": {
            "head_names": list(HEAD_NAMES),
            "head_class_counts": list(HEAD_CLASS_COUNTS),
            "total_scores": HEAD_OFFSETS[-1],
            "scores_are_probabilities": False,
        },
        "objective": "sum squared error plus lambda times squared weight norm; intercept unregularized",
        "ridge_grid": ridge_grid,
        "standardization": {
            "statistics": "training rows only",
            "std_floor": std_floor,
            "low_variance_coordinates_are_zero": True,
        },
        "selected_lambdas": {name: readout.ridge_lambda for name, readout in fitted.items()},
        "development_searches": searches,
        "development_metrics": metrics,
        "development_pair_changed_square_values": pair_values,
        "strongest_cheap_baseline": strongest_baseline,
        "main_minus_strongest_baseline_interval": improvement_interval,
        "gate_c_probe_passed": gate_c,
        "unseen_training_classes": {
            HEAD_NAMES[head]: [
                class_id
                for class_id in range(HEAD_CLASS_COUNTS[head])
                if class_id not in set(train["targets"][:, head].tolist())
            ]
            for head in range(70)
        },
        "development_labels_used_for_selection": len(ids),
        "training_labels_used_for_fit": int(train["targets"].shape[0]),
        "probe_weights_sha256": sha256_file(main_weights_path),
        "baseline_probe_weights_sha256": sha256_file(baseline_weights_path),
    }
    atomic_write_json(output_path, payload)
    log_time(run_dir, "fit-probe", started, f"gate_c={gate_c}")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Behavioral competence pilot and shared condition evaluation
# ---------------------------------------------------------------------------


def preference_credit(value: float, positive_is_success: bool) -> float:
    if value == 0.0:
        return 0.5
    return float((value > 0.0) == positive_is_success)


def condition_ids_for_pair() -> tuple[str, ...]:
    return (
        "AA",
        "BB",
        "BA",
        "AB",
        "A_R_zero_K",
        "A_zero_R_K",
        "A_zero_both",
        "B_R_zero_K",
        "B_zero_R_K",
        "B_zero_both",
    )


@dataclass(frozen=True)
class BaselineReadouts:
    orderless: RidgeReadout
    last_move: RidgeReadout
    permutation: RidgeReadout
    training_frequency: np.ndarray
    initial_board: np.ndarray


def load_baseline_readouts(path: Path) -> BaselineReadouts:
    with np.load(path, allow_pickle=False) as data:
        def read(prefix: str) -> RidgeReadout:
            return RidgeReadout(
                data[f"{prefix}_feature_mean"].astype(np.float64),
                data[f"{prefix}_feature_scale"].astype(np.float64),
                data[f"{prefix}_target_mean"].astype(np.float64),
                data[f"{prefix}_weights"].astype(np.float64),
                float(data[f"{prefix}_ridge_lambda"][0]),
            )

        return BaselineReadouts(
            orderless=read("orderless"),
            last_move=read("last_move"),
            permutation=read("permutation"),
            training_frequency=data["training_frequency_prediction"].astype(np.int16),
            initial_board=data["initial_board_prediction"].astype(np.int16),
        )


def evaluate_position_baselines(
    backend: FrozenQwenBackend,
    baselines: BaselineReadouts,
    position: Mapping[str, Any],
    snapshot: CacheSnapshot,
) -> dict[str, Any]:
    orderless_feature, last_move_feature = backend.input_embedding_features(position)
    model_feature = backend.read_feature(snapshot, position["query_token_ids"])
    output: dict[str, Any] = {
        "initial_board": {"prediction": baselines.initial_board.astype(int).tolist()},
        "training_frequency": {"prediction": baselines.training_frequency.astype(int).tolist()},
    }
    for name, readout, feature in (
        ("orderless", baselines.orderless, orderless_feature),
        ("last_move", baselines.last_move, last_move_feature),
        ("permutation", baselines.permutation, model_feature),
    ):
        scores = readout.scores(feature[None, :])[0]
        output[name] = {
            "prediction": decode_head_scores(scores[None, :])[0].astype(int).tolist(),
            "scores": scores.astype(float).tolist(),
        }
    return output


def make_condition_snapshot(
    condition_id: str,
    a_snapshot: CacheSnapshot,
    b_snapshot: CacheSnapshot,
) -> CacheSnapshot:
    if condition_id == "AA":
        return assemble_snapshot(a_snapshot, a_snapshot)
    if condition_id == "BB":
        return assemble_snapshot(b_snapshot, b_snapshot)
    if condition_id == "BA":
        return assemble_snapshot(b_snapshot, a_snapshot)
    if condition_id == "AB":
        return assemble_snapshot(a_snapshot, b_snapshot)
    if condition_id == "A_R_zero_K":
        return zero_channels(a_snapshot, attention=True)
    if condition_id == "A_zero_R_K":
        return zero_channels(a_snapshot, recurrent=True)
    if condition_id == "A_zero_both":
        return zero_channels(a_snapshot, recurrent=True, attention=True)
    if condition_id == "B_R_zero_K":
        return zero_channels(b_snapshot, attention=True)
    if condition_id == "B_zero_R_K":
        return zero_channels(b_snapshot, recurrent=True)
    if condition_id == "B_zero_both":
        return zero_channels(b_snapshot, recurrent=True, attention=True)
    raise ProtocolError(f"unknown intervention condition: {condition_id}")


def score_condition(
    backend: FrozenQwenBackend,
    readout: RidgeReadout,
    snapshot: CacheSnapshot,
    query_ids: Sequence[int],
    a_candidate_ids: Sequence[int],
    b_candidate_ids: Sequence[int],
    save_scores: bool = True,
) -> dict[str, Any]:
    feature = backend.read_feature(snapshot, query_ids)
    scores = readout.scores(feature[None, :])[0]
    prediction = decode_head_scores(scores[None, :])[0]
    a_score = backend.score_candidate(snapshot, query_ids, a_candidate_ids)
    b_score = backend.score_candidate(snapshot, query_ids, b_candidate_ids)
    result = {
        "feature_norm": float(np.linalg.norm(feature)),
        "probe_prediction": prediction.astype(int).tolist(),
        "a_candidate": a_score,
        "b_candidate": b_score,
        "d_b_minus_a": float(b_score["sum_logprob"] - a_score["sum_logprob"]),
    }
    if save_scores:
        result["probe_scores"] = scores.astype(float).tolist()
    return result


def factorial_effects(condition_results: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    d = {condition: float(condition_results[condition]["d_b_minus_a"]) for condition in CONDITION_IDS}
    recurrent_effect = 0.5 * ((d["BA"] - d["AA"]) + (d["BB"] - d["AB"]))
    kv_effect = 0.5 * ((d["AB"] - d["AA"]) + (d["BB"] - d["BA"]))
    interaction = d["BB"] - d["BA"] - d["AB"] + d["AA"]
    full_difference = d["BB"] - d["AA"]
    return {
        "recurrent_effect": recurrent_effect,
        "kv_effect": kv_effect,
        "interaction": interaction,
        "full_difference": full_difference,
        "channel_contrast": kv_effect - recurrent_effect,
        "recurrent_effect_at_KA": d["BA"] - d["AA"],
        "recurrent_effect_at_KB": d["BB"] - d["AB"],
        "kv_effect_at_RA": d["AB"] - d["AA"],
        "kv_effect_at_RB": d["BB"] - d["BA"],
    }


def evaluate_pair_conditions(
    backend: FrozenQwenBackend,
    readout: RidgeReadout,
    manifest: Mapping[str, Any],
    pair: Mapping[str, Any],
    save_scores: bool = True,
    baselines: BaselineReadouts | None = None,
) -> dict[str, Any]:
    a, b = pair_position_rows(manifest, pair)
    a_snapshot = backend.prefill(a["context_token_ids"])
    b_snapshot = backend.prefill(b["context_token_ids"])
    a_digest = snapshot_digest(a_snapshot)
    b_digest = snapshot_digest(b_snapshot)
    baseline_results = None
    if baselines is not None:
        baseline_results = {
            "A": evaluate_position_baselines(backend, baselines, a, a_snapshot),
            "B": evaluate_position_baselines(backend, baselines, b, b_snapshot),
        }
    order = list(condition_ids_for_pair())
    random.Random(int(stable_hash(11, pair["pair_id"], "condition-order")[:16], 16)).shuffle(order)
    results: dict[str, Any] = {}
    for condition_id in order:
        condition_snapshot = make_condition_snapshot(condition_id, a_snapshot, b_snapshot)
        results[condition_id] = score_condition(
            backend,
            readout,
            condition_snapshot,
            a["query_token_ids"],
            pair["a_candidate_ids"],
            pair["b_candidate_ids"],
            save_scores=save_scores,
        )
        del condition_snapshot
    if snapshot_digest(a_snapshot) != a_digest or snapshot_digest(b_snapshot) != b_digest:
        raise ProtocolError("pair evaluation mutated a source donor snapshot")
    effects = factorial_effects(results)
    return {
        "pair_id": pair["pair_id"],
        "split": pair["split"],
        "a_position": a,
        "b_position": b,
        "candidates": {
            "a_move": pair["a_move"],
            "b_move": pair["b_move"],
            "a_candidate_ids": pair["a_candidate_ids"],
            "b_candidate_ids": pair["b_candidate_ids"],
        },
        "condition_order": order,
        "conditions": results,
        "effects": effects,
        "baselines": baseline_results,
        "source_snapshot_digests": {"A": a_digest, "B": b_digest},
        "complete": True,
    }


def evaluate_full_behavior_pair(
    backend: FrozenQwenBackend,
    manifest: Mapping[str, Any],
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    a, b = pair_position_rows(manifest, pair)
    rows = {}
    for donor, position, candidate_name in (
        ("A", a, "a_candidate_ids"),
        ("B", b, "b_candidate_ids"),
    ):
        snapshot = backend.prefill(position["context_token_ids"])
        digest = snapshot_digest(snapshot)
        a_score = backend.score_candidate(snapshot, position["query_token_ids"], pair["a_candidate_ids"])
        b_score = backend.score_candidate(snapshot, position["query_token_ids"], pair["b_candidate_ids"])
        if snapshot_digest(snapshot) != digest:
            raise ProtocolError("development behavior scoring mutated its source snapshot")
        difference = float(b_score["sum_logprob"] - a_score["sum_logprob"])
        rows[donor] = {
            "position_id": position["position_id"],
            "a_candidate": a_score,
            "b_candidate": b_score,
            "d_b_minus_a": difference,
            "success": preference_credit(difference, positive_is_success=(donor == "B")),
            "intended_candidate": candidate_name,
        }
    return {
        "pair_id": pair["pair_id"],
        "A": rows["A"],
        "B": rows["B"],
        "balanced_accuracy": 0.5 * (rows["A"]["success"] + rows["B"]["success"]),
        "both_correct": bool(rows["A"]["success"] == 1.0 and rows["B"]["success"] == 1.0),
    }


def cmd_pilot(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, _, run_dir = stage_context(args)
    output_path = run_dir / "pilot.json"
    reused = require_new_or_resume(output_path, args.resume)
    if reused is not None:
        print(json.dumps(reused, indent=2))
        return
    preflight = read_json(run_dir / "preflight.json")
    if not preflight.get("all_gates_passed"):
        raise ProtocolError("pilot refuses because real-model Gate A did not pass")
    probe_config = read_json(run_dir / "probe_config.json")
    manifest = load_manifest(manifest_path_for_args(args))
    if any(
        payload.get("manifest_sha256") != manifest["manifest_sha256"]
        for payload in (preflight, probe_config)
    ):
        raise ProtocolError("preflight/probe artifacts do not match the current manifest")
    readout = load_readout(run_dir / "probe_weights.npz")
    backend, runtime = load_backend(config, manifest, args.allow_remote_downloads)
    parameter_digest_before = full_parameter_digest(backend.full_model)
    if parameter_digest_before != preflight["model_parameter_digest_after"]:
        raise ProtocolError("pilot model parameters differ from the preflight")

    development_records = []
    for index, pair in enumerate(manifest["development_pairs"], 1):
        development_records.append(evaluate_full_behavior_pair(backend, manifest, pair))
        if index % 10 == 0:
            print(f"[pilot] development behavior {index}/{len(manifest['development_pairs'])}", flush=True)
    balanced_values = [float(row["balanced_accuracy"]) for row in development_records]
    behavior_interval = bootstrap_mean_interval(
        balanced_values, int(config.raw["bootstrap_replicates"]), int(config.raw["bootstrap_seed"])
    )
    gate_b = (
        behavior_interval["estimate"] is not None
        and float(behavior_interval["estimate"]) >= float(config.gates["behavior_minimum_balanced_accuracy"])
        and float(behavior_interval["lower"]) > float(config.gates["behavior_interval_lower_bound"])
    )

    timing_records = []
    timing_count = min(int(config.raw["runtime"]["timing_pilot_pairs"]), len(manifest["pilot_pairs"]))
    for pair in manifest["pilot_pairs"][:timing_count]:
        pair_started = time.monotonic()
        result = evaluate_pair_conditions(backend, readout, manifest, pair, save_scores=False)
        torch.cuda.synchronize(backend.device)
        timing_records.append(
            {
                "pair_id": pair["pair_id"],
                "seconds": time.monotonic() - pair_started,
                "effects": result["effects"],
            }
        )
    seconds_per_pair = float(np.mean([row["seconds"] for row in timing_records]))
    full_units = int(config.counts["test_pairs"]) + 3 * int(config.counts["continuation_pairs"]) + 1.5 * int(
        config.counts["transposition_triplets"]
    )
    compact_units = int(config.raw["compact_counts"]["test_pairs"]) + 3 * int(
        config.raw["compact_counts"]["continuation_pairs"]
    ) + 1.5 * int(config.raw["compact_counts"]["transposition_triplets"])
    parameter_digest_after = full_parameter_digest(backend.full_model)
    if parameter_digest_before != parameter_digest_after:
        raise ProtocolError("frozen model changed during the behavioral pilot")
    gate_c = bool(probe_config["gate_c_probe_passed"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_manifest": runtime,
        "model_parameter_digest_before": parameter_digest_before,
        "model_parameter_digest_after": parameter_digest_after,
        "development_behavior": {
            "records": development_records,
            "balanced_accuracy_interval": behavior_interval,
            "both_full_donors_correct_fraction": float(np.mean([row["both_correct"] for row in development_records])),
        },
        "gate_b_behavior_passed": gate_b,
        "gate_c_probe_passed": gate_c,
        "probe_gate_evidence": probe_config["main_minus_strongest_baseline_interval"],
        "strongest_cheap_baseline": probe_config["strongest_cheap_baseline"],
        "stop_decision": "proceed_behavioral_assay"
        if gate_b
        else "stop_before_test_behavioral_assay",
        "probe_interpretation": "informative_for_planned_board_analysis" if gate_c else "linear_readout_gate_failed",
        "timing": {
            "records": timing_records,
            "seconds_per_actual_pair_mean": seconds_per_pair,
            "projection_method": "primary pairs + 3 incremental continuation horizons + 1.5 units per transposition triplet",
            "full_schedule_projected_seconds": seconds_per_pair * full_units,
            "compact_schedule_projected_seconds": seconds_per_pair * compact_units,
            "full_schedule_units": full_units,
            "compact_schedule_units": compact_units,
        },
    }
    atomic_write_json(output_path, payload)
    log_time(run_dir, "pilot", started, f"gate_b={gate_b};gate_c={gate_c}")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Protocol freeze and resume contract
# ---------------------------------------------------------------------------


def freeze_hash(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("freeze_sha256", None)
    return sha256_json(copy)


def validate_freeze_hash(payload: Mapping[str, Any]) -> None:
    if payload.get("freeze_sha256") != freeze_hash(payload):
        raise ProtocolError("freeze artifact hash does not match its contents")


def code_and_protocol_hashes(config_path: Path, manifest_path: Path, run_dir: Path) -> dict[str, str]:
    paths = {
        "config": config_path,
        "manifest": manifest_path,
        "script": resolve_path("qwen35_chess_memory.py"),
        "requirements": resolve_path("requirements_qwen35_chess_memory.txt"),
        "plan": resolve_path(PLAN_PATH),
        "probe_config": run_dir / "probe_config.json",
        "probe_weights": run_dir / "probe_weights.npz",
        "baseline_probe_weights": run_dir / "baseline_probe_weights.npz",
        "preflight": run_dir / "preflight.json",
        "pilot": run_dir / "pilot.json",
        "train_features": run_dir / "features" / "train.npz",
        "development_features": run_dir / "features" / "development.npz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ProtocolError(f"cannot freeze because required files are absent: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def schedule_ids(manifest: Mapping[str, Any], schedule: str, config: StudyConfig) -> dict[str, list[str]]:
    counts = config.counts if schedule == "full" else config.raw["compact_counts"]
    # Manifest order is already the prospective stable-hash proposal order.
    ordered_pairs = list(manifest["test_pairs"])
    continuation_pool = [pair for pair in ordered_pairs if "continuation" in pair]
    continuations = continuation_pool[: int(counts["continuation_pairs"])]
    selected_ids = {str(pair["pair_id"]) for pair in continuations}
    for pair in ordered_pairs:
        if len(selected_ids) >= int(counts["test_pairs"]):
            break
        selected_ids.add(str(pair["pair_id"]))
    primary = [pair for pair in ordered_pairs if str(pair["pair_id"]) in selected_ids]
    transpositions = list(manifest["transposition_triplets"])[: int(counts["transposition_triplets"])]
    return {
        "primary_pair_ids": [str(pair["pair_id"]) for pair in primary],
        "continuation_pair_ids": [str(pair["pair_id"]) for pair in continuations],
        "transposition_triplet_ids": [str(triplet["triplet_id"]) for triplet in transpositions],
    }


def cmd_freeze(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, config_path, run_dir = stage_context(args)
    output_path = run_dir / "freeze.json"
    reused = require_new_or_resume(output_path, args.resume)
    if reused is not None:
        validate_freeze_hash(reused)
        print(json.dumps(reused, indent=2))
        return
    if args.remaining_seconds <= 0:
        raise ProtocolError("freeze requires a positive measured remaining-time budget")
    if any((run_dir / "results" / name).exists() for name in ("primary.jsonl", "transpositions.jsonl", "continuations.jsonl")):
        raise ProtocolError("cannot create a prospective freeze after test result files exist")
    manifest_path = manifest_path_for_args(args)
    manifest = load_manifest(manifest_path)
    preflight = read_json(run_dir / "preflight.json")
    pilot = read_json(run_dir / "pilot.json")
    probe_config = read_json(run_dir / "probe_config.json")
    if not preflight.get("all_gates_passed"):
        raise ProtocolError("freeze refuses because Gate A did not pass")
    if not pilot.get("gate_b_behavior_passed"):
        raise ProtocolError("freeze refuses because Gate B did not pass")
    if any(
        payload.get("manifest_sha256") != manifest["manifest_sha256"]
        for payload in (preflight, pilot, probe_config)
    ):
        raise ProtocolError("preflight, pilot, probe, and manifest are not from one frozen dataset")
    timing = pilot["timing"]
    full_projection = float(timing["full_schedule_projected_seconds"])
    compact_projection = float(timing["compact_schedule_projected_seconds"])
    usable = float(args.remaining_seconds) * float(config.raw["runtime"]["schedule_safety_fraction"])
    if full_projection <= usable:
        schedule = "full"
    elif compact_projection <= usable:
        schedule = "compact"
    else:
        raise ProtocolError(
            "even the compact schedule exceeds the measured remaining-time budget; report the pilot instead"
        )
    identifiers = schedule_ids(manifest, schedule, config)
    required_primary = int(
        config.counts["test_pairs"] if schedule == "full" else config.raw["compact_counts"]["test_pairs"]
    )
    if len(identifiers["primary_pair_ids"]) < required_primary:
        raise ProtocolError("manifest cannot supply the selected primary schedule")
    if len(manifest["development_pairs"]) < 25 or len(manifest["test_pairs"]) < 50:
        raise ProtocolError("ordinary-pair assay coverage is below the preregistered minimum")
    availability = {
        "continuation": "available"
        if len(identifiers["continuation_pair_ids"]) >= 16
        else "underpowered",
        "transposition": "available"
        if len(identifiers["transposition_triplet_ids"]) >= 20
        else "unavailable_or_underpowered",
    }
    file_hashes = code_and_protocol_hashes(config_path, manifest_path, run_dir)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "schedule": schedule,
        "selection_basis": "measured runtime projection only",
        "remaining_seconds_declared": float(args.remaining_seconds),
        "schedule_safety_fraction": float(config.raw["runtime"]["schedule_safety_fraction"]),
        "usable_seconds": usable,
        "full_projected_seconds": full_projection,
        "compact_projected_seconds": compact_projection,
        "manifest_sha256": manifest["manifest_sha256"],
        "model": manifest["model"],
        "model_parameter_digest": preflight["model_parameter_digest_after"],
        "model_weight_files": preflight["runtime_manifest"]["weight_files"],
        "tokenizer_files": manifest["model"]["tokenizer_files"],
        "file_hashes": file_hashes,
        "conditions": list(condition_ids_for_pair()),
        "condition_notation": "first donor is recurrent; second donor is full-attention KV",
        "continuation_horizons": list(HORIZONS),
        "selected_ids": identifiers,
        "actual_selected_counts": {key: len(value) for key, value in identifiers.items()},
        "control_availability": availability,
        "gate_a_passed": True,
        "gate_b_passed": True,
        "gate_c_passed": bool(pilot["gate_c_probe_passed"]),
        "strongest_cheap_baseline": pilot["strongest_cheap_baseline"],
        "probe_claim_status": "planned_interpretation_available"
        if pilot["gate_c_probe_passed"]
        else "readout_gate_failed_behavior_only",
    }
    payload["freeze_sha256"] = freeze_hash(payload)
    atomic_write_json(output_path, payload)
    log_time(run_dir, "freeze", started, f"schedule={schedule}")
    print(json.dumps(payload, indent=2))


def load_and_validate_freeze(
    config_path: Path,
    manifest_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    freeze = read_json(run_dir / "freeze.json")
    validate_freeze_hash(freeze)
    current = code_and_protocol_hashes(config_path, manifest_path, run_dir)
    if current != freeze["file_hashes"]:
        changed = sorted(name for name in current if current.get(name) != freeze["file_hashes"].get(name))
        raise ProtocolError(f"frozen source/probe files changed after unsealing: {changed}")
    manifest = load_manifest(manifest_path)
    if manifest["manifest_sha256"] != freeze["manifest_sha256"]:
        raise ProtocolError("frozen manifest hash changed")
    return freeze


# ---------------------------------------------------------------------------
# Held-out evaluation with append-only cluster journals
# ---------------------------------------------------------------------------


def result_record_hash(record: Mapping[str, Any]) -> str:
    copy = dict(record)
    copy.pop("record_sha256", None)
    return sha256_json(copy)


def journal_state(path: Path, marker_dir: Path, id_field: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(id_field, ""))
        if not identifier or identifier in completed:
            raise ProtocolError(f"journal has a missing or duplicated {id_field}: {path}")
        if not row.get("complete") or row.get("record_sha256") != result_record_hash(row):
            raise ProtocolError(f"journal row is incomplete or corrupt: {path}:{identifier}")
        completed[identifier] = row
    marker_dir.mkdir(parents=True, exist_ok=True)
    markers = {path.stem: path for path in marker_dir.glob("*.json")}
    unknown_markers = sorted(set(markers) - set(completed))
    if unknown_markers:
        raise ProtocolError(f"completion markers exist without journal rows: {unknown_markers[:3]}")
    for identifier, row in completed.items():
        marker_path = marker_dir / f"{identifier}.json"
        expected = {id_field: identifier, "record_sha256": row["record_sha256"]}
        if marker_path.exists():
            marker = read_json(marker_path)
            if marker != expected:
                raise ProtocolError(f"completion marker hash mismatch for {identifier}")
        else:
            # A crash after fsyncing the complete journal row but before the
            # atomic marker is safely recoverable without recomputation.
            atomic_write_json(marker_path, expected)
    return completed


def append_complete_result(
    path: Path,
    marker_dir: Path,
    id_field: str,
    record: MutableMapping[str, Any],
) -> None:
    identifier = str(record[id_field])
    if (marker_dir / f"{identifier}.json").exists():
        raise ProtocolError(f"refusing to duplicate completed result {identifier}")
    record["complete"] = True
    record["record_sha256"] = result_record_hash(record)
    append_jsonl(path, record)
    atomic_write_json(
        marker_dir / f"{identifier}.json",
        {id_field: identifier, "record_sha256": record["record_sha256"]},
    )


def evaluate_transposition_triplet(
    backend: FrozenQwenBackend,
    readout: RidgeReadout,
    manifest: Mapping[str, Any],
    triplet: Mapping[str, Any],
) -> dict[str, Any]:
    positions = manifest["positions"]
    a = positions[triplet["a_position_id"]]
    aprime = positions[triplet["aprime_position_id"]]
    b = positions[triplet["b_position_id"]]
    snapshots = {
        "A": backend.prefill(a["context_token_ids"]),
        "P": backend.prefill(aprime["context_token_ids"]),
        "B": backend.prefill(b["context_token_ids"]),
    }
    digests = {name: snapshot_digest(snapshot) for name, snapshot in snapshots.items()}
    factories: dict[str, Callable[[], CacheSnapshot]] = {
        "AA": lambda: assemble_snapshot(snapshots["A"], snapshots["A"]),
        "PP": lambda: assemble_snapshot(snapshots["P"], snapshots["P"]),
        "BB": lambda: assemble_snapshot(snapshots["B"], snapshots["B"]),
        "PA": lambda: assemble_snapshot(snapshots["P"], snapshots["A"]),
        "AP": lambda: assemble_snapshot(snapshots["A"], snapshots["P"]),
        "BA": lambda: assemble_snapshot(snapshots["B"], snapshots["A"]),
        "AB": lambda: assemble_snapshot(snapshots["A"], snapshots["B"]),
    }
    order = list(factories)
    random.Random(int(stable_hash(11, triplet["triplet_id"], "condition-order")[:16], 16)).shuffle(order)
    results: dict[str, Any] = {}
    for condition in order:
        snapshot = factories[condition]()
        results[condition] = score_condition(
            backend,
            readout,
            snapshot,
            a["query_token_ids"],
            triplet["a_candidate_ids"],
            triplet["b_candidate_ids"],
            save_scores=True,
        )
        del snapshot
    if any(snapshot_digest(snapshots[name]) != digest for name, digest in digests.items()):
        raise ProtocolError("transposition evaluation mutated a source snapshot")
    factorial_input = {condition: results[condition] for condition in CONDITION_IDS}
    return {
        "triplet_id": triplet["triplet_id"],
        "a_position": a,
        "aprime_position": aprime,
        "b_position": b,
        "matching": {
            "same_position_token_hamming": triplet["same_position_token_hamming"],
            "different_position_token_hamming": triplet["different_position_token_hamming"],
        },
        "candidates": {
            "a_move": triplet["a_move"],
            "b_move": triplet["b_move"],
            "a_candidate_ids": triplet["a_candidate_ids"],
            "b_candidate_ids": triplet["b_candidate_ids"],
        },
        "condition_order": order,
        "conditions": results,
        "different_position_effects": factorial_effects(factorial_input),
        "source_snapshot_digests": digests,
    }


def continuation_condition_states(
    a_snapshot: CacheSnapshot,
    b_snapshot: CacheSnapshot,
) -> dict[str, CacheSnapshot]:
    return {
        condition: make_condition_snapshot(condition, a_snapshot, b_snapshot)
        for condition in condition_ids_for_pair()
    }


def evaluate_continuation_pair(
    backend: FrozenQwenBackend,
    readout: RidgeReadout,
    manifest: Mapping[str, Any],
    pair: Mapping[str, Any],
    primary_record: Mapping[str, Any],
) -> dict[str, Any]:
    if "continuation" not in pair:
        raise ProtocolError("continuation evaluation received an ineligible pair")
    a, b = pair_position_rows(manifest, pair)
    a_snapshot = backend.prefill(a["context_token_ids"])
    b_snapshot = backend.prefill(b["context_token_ids"])
    source_digests = {"A": snapshot_digest(a_snapshot), "B": snapshot_digest(b_snapshot)}
    trajectories = continuation_condition_states(a_snapshot, b_snapshot)
    horizon_records: dict[str, Any] = {
        "0": {
            "a_position": a,
            "b_position": b,
            "candidates": primary_record["candidates"],
            "conditions": primary_record["conditions"],
            "effects": primary_record["effects"],
            "source": "identical frozen primary record",
        }
    }
    previous_cumulative: list[int] = []
    for horizon in (1, 2, 4):
        saved = pair["continuation"]["horizons"][str(horizon)]
        cumulative = [int(value) for value in saved["a"]["advance_token_ids"]]
        if cumulative[: len(previous_cumulative)] != previous_cumulative:
            raise ProtocolError("continuation horizon token IDs are not cumulatively nested")
        increment = cumulative[len(previous_cumulative) :]
        if not increment:
            raise ProtocolError("continuation horizon has no newly appended token IDs")
        trajectories = {
            condition: backend.advance(snapshot, increment)
            for condition, snapshot in trajectories.items()
        }
        expected_length = a_snapshot.sequence_length + len(cumulative)
        if any(snapshot.sequence_length != expected_length for snapshot in trajectories.values()):
            raise ProtocolError("continued trajectory has an incorrect cache sequence length")
        order = list(condition_ids_for_pair())
        random.Random(
            int(stable_hash(11, pair["pair_id"], horizon, "condition-order")[:16], 16)
        ).shuffle(order)
        results: dict[str, Any] = {}
        for condition in order:
            results[condition] = score_condition(
                backend,
                readout,
                trajectories[condition],
                saved["a"]["query_token_ids"],
                saved["candidates"]["a_candidate_ids"],
                saved["candidates"]["b_candidate_ids"],
                save_scores=True,
            )
        horizon_records[str(horizon)] = {
            "a_position": saved["a"],
            "b_position": saved["b"],
            "candidates": saved["candidates"],
            "condition_order": order,
            "conditions": results,
            "effects": factorial_effects(results),
            "cumulative_advance_token_ids": cumulative,
            "incremental_advance_token_ids": increment,
        }
        previous_cumulative = cumulative
    if snapshot_digest(a_snapshot) != source_digests["A"] or snapshot_digest(b_snapshot) != source_digests["B"]:
        raise ProtocolError("continuation evaluation mutated an original donor snapshot")
    return {
        "pair_id": pair["pair_id"],
        "continuation_source": pair["continuation"]["source"],
        "shared_suffix_moves": pair["continuation"]["moves"],
        "primary_record_sha256": primary_record["record_sha256"],
        "source_snapshot_digests": source_digests,
        "horizons": horizon_records,
    }


def result_maps(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_map = {str(pair["pair_id"]): pair for pair in manifest["test_pairs"]}
    triplet_map = {
        str(triplet["triplet_id"]): triplet for triplet in manifest["transposition_triplets"]
    }
    return pair_map, triplet_map


def cmd_evaluate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, config_path, run_dir = stage_context(args)
    manifest_path = manifest_path_for_args(args)
    manifest = load_manifest(manifest_path)
    freeze = load_and_validate_freeze(config_path, manifest_path, run_dir)
    pair_map, triplet_map = result_maps(manifest)
    results_dir = run_dir / "results"
    primary_path = results_dir / "primary.jsonl"
    transposition_path = results_dir / "transpositions.jsonl"
    continuation_path = results_dir / "continuations.jsonl"
    primary_completed = journal_state(primary_path, results_dir / "complete" / "primary", "pair_id")
    transposition_completed = journal_state(
        transposition_path, results_dir / "complete" / "transpositions", "triplet_id"
    )
    continuation_completed = journal_state(
        continuation_path, results_dir / "complete" / "continuations", "pair_id"
    )
    if (primary_completed or transposition_completed or continuation_completed) and not args.resume:
        raise ProtocolError("held-out journal already contains results; pass --resume to continue exact frozen IDs")

    backend, runtime = load_backend(config, manifest, args.allow_remote_downloads)
    if runtime["resolved_revision"] != freeze["model"]["resolved_revision"]:
        raise ProtocolError("evaluation loaded a different model revision than the freeze")
    if runtime["weight_files"] != freeze["model_weight_files"]:
        raise ProtocolError("evaluation model files differ from the freeze")
    parameter_digest_before = full_parameter_digest(backend.full_model)
    if parameter_digest_before != freeze["model_parameter_digest"]:
        raise ProtocolError("evaluation model parameter digest differs from the freeze")
    readout = load_readout(run_dir / "probe_weights.npz")
    baseline_readouts = load_baseline_readouts(run_dir / "baseline_probe_weights.npz")

    selected = freeze["selected_ids"]
    for index, pair_id in enumerate(selected["primary_pair_ids"], 1):
        if pair_id in primary_completed:
            continue
        pair = pair_map[pair_id]
        row = evaluate_pair_conditions(
            backend,
            readout,
            manifest,
            pair,
            save_scores=True,
            baselines=baseline_readouts,
        )
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "freeze_sha256": freeze["freeze_sha256"],
                "model_revision": runtime["resolved_revision"],
            }
        )
        append_complete_result(
            primary_path, results_dir / "complete" / "primary", "pair_id", row
        )
        primary_completed[pair_id] = row
        print(f"[evaluate] primary {index}/{len(selected['primary_pair_ids'])}: {pair_id}", flush=True)

    for index, triplet_id in enumerate(selected["transposition_triplet_ids"], 1):
        if triplet_id in transposition_completed:
            continue
        triplet = triplet_map[triplet_id]
        row = evaluate_transposition_triplet(backend, readout, manifest, triplet)
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "freeze_sha256": freeze["freeze_sha256"],
                "model_revision": runtime["resolved_revision"],
            }
        )
        append_complete_result(
            transposition_path,
            results_dir / "complete" / "transpositions",
            "triplet_id",
            row,
        )
        transposition_completed[triplet_id] = row
        print(
            f"[evaluate] transposition {index}/{len(selected['transposition_triplet_ids'])}: {triplet_id}",
            flush=True,
        )

    for index, pair_id in enumerate(selected["continuation_pair_ids"], 1):
        if pair_id in continuation_completed:
            continue
        if pair_id not in primary_completed:
            raise ProtocolError("continuation pair is missing its exact primary horizon-zero record")
        pair = pair_map[pair_id]
        row = evaluate_continuation_pair(
            backend, readout, manifest, pair, primary_completed[pair_id]
        )
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "freeze_sha256": freeze["freeze_sha256"],
                "model_revision": runtime["resolved_revision"],
            }
        )
        append_complete_result(
            continuation_path,
            results_dir / "complete" / "continuations",
            "pair_id",
            row,
        )
        continuation_completed[pair_id] = row
        print(
            f"[evaluate] continuation {index}/{len(selected['continuation_pair_ids'])}: {pair_id}",
            flush=True,
        )

    parameter_digest_after = full_parameter_digest(backend.full_model)
    if parameter_digest_before != parameter_digest_after:
        raise ProtocolError("frozen model changed during held-out evaluation")
    evaluation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "freeze_sha256": freeze["freeze_sha256"],
        "runtime_manifest": runtime,
        "model_parameter_digest_before": parameter_digest_before,
        "model_parameter_digest_after": parameter_digest_after,
        "journal_hashes": {
            "primary": sha256_file(primary_path),
            "transpositions": sha256_file(transposition_path),
            "continuations": sha256_file(continuation_path),
        },
        "completed_counts": {
            "primary": len(primary_completed),
            "transpositions": len(transposition_completed),
            "continuations": len(continuation_completed),
        },
    }
    atomic_write_json(results_dir / "evaluation_manifest.json", evaluation_manifest)
    log_time(run_dir, "evaluate", started)
    print(json.dumps(evaluation_manifest, indent=2))


# ---------------------------------------------------------------------------
# Saved-output analysis and figures
# ---------------------------------------------------------------------------


def condition_prediction(record: Mapping[str, Any], condition: str) -> np.ndarray:
    return np.asarray(record["conditions"][condition]["probe_prediction"], dtype=np.int16)


def square_accuracy(prediction: np.ndarray, target: Sequence[int], mask: np.ndarray | None = None) -> float:
    gold = np.asarray(target, dtype=np.int16)[:64]
    predicted = np.asarray(prediction, dtype=np.int16)[:64]
    selected = np.ones(64, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not selected.any():
        return math.nan
    return float(np.mean(predicted[selected] == gold[selected]))


def exact_state_accuracy(prediction: np.ndarray, target: Sequence[int]) -> float:
    return float(np.all(np.asarray(prediction, dtype=np.int16) == np.asarray(target, dtype=np.int16)))


def changed_from_initial_mask(target: Sequence[int]) -> np.ndarray:
    return np.asarray(target, dtype=np.int16)[:64] != initial_board_targets()[:64]


def behavioral_credit_for_reference(condition: Mapping[str, Any], reference: str) -> float:
    difference = float(condition["d_b_minus_a"])
    return preference_credit(difference, positive_is_success=(reference == "B"))


def interval_for(
    values: Sequence[float], config: StudyConfig
) -> dict[str, Any]:
    return bootstrap_mean_interval(
        values,
        int(config.raw["bootstrap_replicates"]),
        int(config.raw["bootstrap_seed"]),
    )


def paired_condition_metrics(
    primary_rows: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> dict[str, Any]:
    groups = {
        "full": (("AA", "A"), ("BB", "B")),
        "attention_content_zeroed": (("A_R_zero_K", "A"), ("B_R_zero_K", "B")),
        "recurrent_content_zeroed": (("A_zero_R_K", "A"), ("B_zero_R_K", "B")),
        "both_contents_zeroed": (("A_zero_both", "A"), ("B_zero_both", "B")),
    }
    output: dict[str, Any] = {}
    for group, condition_references in groups.items():
        behavior_values: list[float] = []
        changed_values: list[float] = []
        square_values: list[float] = []
        exact_board_values: list[float] = []
        exact_state_values: list[float] = []
        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for row in primary_rows:
            per_behavior = []
            per_changed = []
            per_square = []
            per_exact_board = []
            per_exact_state = []
            for condition, reference in condition_references:
                position = row["a_position"] if reference == "A" else row["b_position"]
                prediction = condition_prediction(row, condition)
                target = np.asarray(position["targets"], dtype=np.int16)
                per_behavior.append(behavioral_credit_for_reference(row["conditions"][condition], reference))
                per_changed.append(square_accuracy(prediction, target, changed_from_initial_mask(target)))
                per_square.append(square_accuracy(prediction, target))
                per_exact_board.append(float(np.all(prediction[:64] == target[:64])))
                per_exact_state.append(float(np.all(prediction == target)))
                all_predictions.append(prediction)
                all_targets.append(target)
            behavior_values.append(float(np.mean(per_behavior)))
            changed_values.append(float(np.nanmean(per_changed)))
            square_values.append(float(np.mean(per_square)))
            exact_board_values.append(float(np.mean(per_exact_board)))
            exact_state_values.append(float(np.mean(per_exact_state)))
        combined_metrics = probe_metrics(np.stack(all_predictions), np.stack(all_targets))
        output[group] = {
            "balanced_legal_candidate_accuracy": interval_for(behavior_values, config),
            "changed_square_probe_accuracy": interval_for(changed_values, config),
            "square_probe_accuracy": interval_for(square_values, config),
            "exact_board_probe_accuracy": interval_for(exact_board_values, config),
            "piece_and_rule_exact_probe_accuracy": interval_for(exact_state_values, config),
            "all_readout_metrics": combined_metrics,
            "cluster_values": {
                "balanced_legal_candidate_accuracy": behavior_values,
                "changed_square_probe_accuracy": changed_values,
                "square_probe_accuracy": square_values,
                "exact_board_probe_accuracy": exact_board_values,
                "piece_and_rule_exact_probe_accuracy": exact_state_values,
            },
        }
    return output


def head_class_score(scores: Sequence[float], head: int, class_id: int) -> float:
    return float(scores[HEAD_OFFSETS[head] + int(class_id)])


def donor_following_for_condition(
    row: Mapping[str, Any], condition: str
) -> dict[str, float]:
    a = np.asarray(row["a_position"]["targets"], dtype=np.int16)
    b = np.asarray(row["b_position"]["targets"], dtype=np.int16)
    prediction = condition_prediction(row, condition)
    differing = a[:64] != b[:64]
    if not differing.any():
        return {"a": math.nan, "b": math.nan, "neither": math.nan, "b_minus_a_score": math.nan}
    predicted = prediction[:64][differing]
    a_labels = a[:64][differing]
    b_labels = b[:64][differing]
    a_fraction = float(np.mean(predicted == a_labels))
    b_fraction = float(np.mean(predicted == b_labels))
    neither = float(np.mean((predicted != a_labels) & (predicted != b_labels)))
    scores = row["conditions"][condition]["probe_scores"]
    differences = [
        head_class_score(scores, head, int(b[head])) - head_class_score(scores, head, int(a[head]))
        for head in np.flatnonzero(differing)
    ]
    return {
        "a": a_fraction,
        "b": b_fraction,
        "neither": neither,
        "b_minus_a_score": float(np.mean(differences)),
    }


def analyze_saved_baselines(
    rows: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> dict[str, Any]:
    names = ("initial_board", "training_frequency", "orderless", "last_move", "permutation")
    output: dict[str, Any] = {}
    for name in names:
        pair_changed: list[float] = []
        pair_square: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for row in rows:
            if row.get("baselines") is None:
                raise ProtocolError("primary result is missing held-out cheap-baseline predictions")
            changed_values = []
            square_values = []
            for donor in ("A", "B"):
                target = np.asarray(
                    row["a_position" if donor == "A" else "b_position"]["targets"], dtype=np.int16
                )
                prediction = np.asarray(row["baselines"][donor][name]["prediction"], dtype=np.int16)
                changed_values.append(square_accuracy(prediction, target, changed_from_initial_mask(target)))
                square_values.append(square_accuracy(prediction, target))
                predictions.append(prediction)
                targets.append(target)
            pair_changed.append(float(np.nanmean(changed_values)))
            pair_square.append(float(np.mean(square_values)))
        output[name] = {
            "changed_square_accuracy": interval_for(pair_changed, config),
            "square_accuracy": interval_for(pair_square, config),
            "all_readout_metrics": probe_metrics(np.stack(predictions), np.stack(targets)),
            "cluster_values": {
                "changed_square_accuracy": pair_changed,
                "square_accuracy": pair_square,
            },
        }
    return output


def analyze_primary(
    rows: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> dict[str, Any]:
    effect_names = (
        "recurrent_effect",
        "kv_effect",
        "interaction",
        "full_difference",
        "channel_contrast",
        "recurrent_effect_at_KA",
        "recurrent_effect_at_KB",
        "kv_effect_at_RA",
        "kv_effect_at_RB",
    )
    effects = {
        name: interval_for([float(row["effects"][name]) for row in rows], config)
        for name in effect_names
    }
    donor_following: dict[str, Any] = {}
    for condition in CONDITION_IDS:
        values = [donor_following_for_condition(row, condition) for row in rows]
        donor_following[condition] = {
            key: interval_for([value[key] for value in values], config)
            for key in ("a", "b", "neither", "b_minus_a_score")
        }
    both_correct = [
        behavioral_credit_for_reference(row["conditions"]["AA"], "A") == 1.0
        and behavioral_credit_for_reference(row["conditions"]["BB"], "B") == 1.0
        for row in rows
    ]
    conditioned = [row for row, keep in zip(rows, both_correct) if keep]
    subgroups: dict[str, Any] = {}
    for name, selected_rows in (
        ("both_full_donors_correct", conditioned),
        ("older_state", [row for row in rows if row["a_position"].get("position_id") and row.get("pair_id") and _pair_flag(row, "older_state_subgroup")]),
        ("shared_final_two_moves", [row for row in rows if _pair_flag(row, "shared_final_two_moves")]),
    ):
        subgroups[name] = {
            "coverage": len(selected_rows),
            "coverage_fraction": len(selected_rows) / len(rows) if rows else 0.0,
            "channel_contrast": interval_for(
                [float(row["effects"]["channel_contrast"]) for row in selected_rows], config
            ),
        }
    return {
        "n_pairs": len(rows),
        "primary_horizon_zero_effects_nats": effects,
        "recipient_reference_and_zeroing": paired_condition_metrics(rows, config),
        "held_out_cheap_baselines": analyze_saved_baselines(rows, config),
        "probe_donor_following_on_a_b_differing_squares": donor_following,
        "secondary_subgroups": subgroups,
        "conditioned_both_full_correct_is_secondary": True,
    }


_ANALYSIS_PAIR_FLAGS: dict[str, Mapping[str, Any]] = {}


def _pair_flag(row: Mapping[str, Any], name: str) -> bool:
    pair = _ANALYSIS_PAIR_FLAGS.get(str(row["pair_id"]), {})
    return bool(pair.get(name, False))


def analyze_transpositions(
    rows: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> dict[str, Any]:
    if not rows:
        return {"status": "unavailable", "n_triplets": 0}
    state_differences: list[float] = []
    behavior_differences: list[float] = []
    full_state_values: list[float] = []
    crossed_state_values: list[float] = []
    full_behavior_values: list[float] = []
    crossed_behavior_values: list[float] = []
    different_effects: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = np.asarray(row["a_position"]["targets"], dtype=np.int16)
        full_state = np.mean(
            [square_accuracy(condition_prediction(row, condition), target) for condition in ("AA", "PP")]
        )
        crossed_state = np.mean(
            [square_accuracy(condition_prediction(row, condition), target) for condition in ("PA", "AP")]
        )
        full_behavior = np.mean(
            [behavioral_credit_for_reference(row["conditions"][condition], "A") for condition in ("AA", "PP")]
        )
        crossed_behavior = np.mean(
            [behavioral_credit_for_reference(row["conditions"][condition], "A") for condition in ("PA", "AP")]
        )
        full_state_values.append(float(full_state))
        crossed_state_values.append(float(crossed_state))
        full_behavior_values.append(float(full_behavior))
        crossed_behavior_values.append(float(crossed_behavior))
        state_differences.append(float(crossed_state - full_state))
        behavior_differences.append(float(crossed_behavior - full_behavior))
        for name, value in row["different_position_effects"].items():
            different_effects[name].append(float(value))
    state_interval = interval_for(state_differences, config)
    behavior_interval = interval_for(behavior_differences, config)
    minimum = float(config.gates["transposition_noninferiority_margin"])
    status = "available" if len(rows) >= 20 else "unavailable_or_underpowered"
    return {
        "status": status,
        "n_triplets": len(rows),
        "same_position": {
            "full_square_accuracy": interval_for(full_state_values, config),
            "crossed_square_accuracy": interval_for(crossed_state_values, config),
            "crossed_minus_full_square_accuracy": state_interval,
            "full_legal_candidate_accuracy": interval_for(full_behavior_values, config),
            "crossed_legal_candidate_accuracy": interval_for(crossed_behavior_values, config),
            "crossed_minus_full_legal_candidate_accuracy": behavior_interval,
            "noninferiority_margin": minimum,
            "state_noninferiority_passed": state_interval["lower"] is not None
            and float(state_interval["lower"]) > minimum,
            "behavior_noninferiority_passed": behavior_interval["lower"] is not None
            and float(behavior_interval["lower"]) > minimum,
        },
        "different_position_effects_nats": {
            name: interval_for(values, config) for name, values in different_effects.items()
        },
        "token_hamming": {
            "same_position_mean": float(np.mean([row["matching"]["same_position_token_hamming"] for row in rows])),
            "different_position_mean": float(
                np.mean([row["matching"]["different_position_token_hamming"] for row in rows])
            ),
        },
    }


def suffix_touched_squares(base_moves: Sequence[str], suffix: Sequence[str]) -> set[int]:
    chess = import_chess()
    board = strict_board_from_moves(base_moves)
    touched: set[int] = set()
    for uci in suffix:
        move = chess.Move.from_uci(str(uci))
        if move not in board.legal_moves:
            raise ProtocolError("saved continuation became illegal during analysis")
        touched.update(move_touched_squares(board, move))
        board.push(move)
    return touched


def analyze_continuations(
    rows: Sequence[Mapping[str, Any]],
    config: StudyConfig,
) -> dict[str, Any]:
    if not rows:
        return {"status": "unavailable", "n_pairs": 0}
    by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        effect_values: dict[str, list[float]] = defaultdict(list)
        current_update_values: list[float] = []
        static_update_values: list[float] = []
        ba_retention: list[float] = []
        ab_retention: list[float] = []
        retention_neither_ba: list[float] = []
        retention_neither_ab: list[float] = []
        full_square_values: list[float] = []
        exact_board_values: list[float] = []
        behavior_values: list[float] = []
        for row in rows:
            h0 = row["horizons"]["0"]
            current = row["horizons"][str(horizon)]
            for name, value in current["effects"].items():
                effect_values[name].append(float(value))
            a_target = np.asarray(current["a_position"]["targets"], dtype=np.int16)
            b_target = np.asarray(current["b_position"]["targets"], dtype=np.int16)
            a_prediction = np.asarray(current["conditions"]["AA"]["probe_prediction"], dtype=np.int16)
            b_prediction = np.asarray(current["conditions"]["BB"]["probe_prediction"], dtype=np.int16)
            full_square_values.append(
                float(np.mean([square_accuracy(a_prediction, a_target), square_accuracy(b_prediction, b_target)]))
            )
            exact_board_values.append(
                float(np.mean([np.all(a_prediction[:64] == a_target[:64]), np.all(b_prediction[:64] == b_target[:64])]))
            )
            behavior_values.append(
                float(
                    np.mean(
                        [
                            behavioral_credit_for_reference(current["conditions"]["AA"], "A"),
                            behavioral_credit_for_reference(current["conditions"]["BB"], "B"),
                        ]
                    )
                )
            )
            if horizon > 0:
                a_initial_target = np.asarray(h0["a_position"]["targets"], dtype=np.int16)
                b_initial_target = np.asarray(h0["b_position"]["targets"], dtype=np.int16)
                a_changed = a_target[:64] != a_initial_target[:64]
                b_changed = b_target[:64] != b_initial_target[:64]
                current_scores = []
                static_scores = []
                if a_changed.any():
                    current_scores.append(square_accuracy(a_prediction, a_target, a_changed))
                    static_a = np.asarray(h0["conditions"]["AA"]["probe_prediction"], dtype=np.int16)
                    static_scores.append(square_accuracy(static_a, a_target, a_changed))
                if b_changed.any():
                    current_scores.append(square_accuracy(b_prediction, b_target, b_changed))
                    static_b = np.asarray(h0["conditions"]["BB"]["probe_prediction"], dtype=np.int16)
                    static_scores.append(square_accuracy(static_b, b_target, b_changed))
                current_update_values.append(float(np.mean(current_scores)) if current_scores else math.nan)
                static_update_values.append(float(np.mean(static_scores)) if static_scores else math.nan)

            differing = a_target[:64] != b_target[:64]
            suffix = row["shared_suffix_moves"][:horizon]
            touched = suffix_touched_squares(h0["a_position"]["moves"], suffix) | suffix_touched_squares(
                h0["b_position"]["moves"], suffix
            )
            untouched = differing.copy()
            if touched:
                untouched[list(touched)] = False
            if untouched.any():
                for condition, values, neither_values in (
                    ("BA", ba_retention, retention_neither_ba),
                    ("AB", ab_retention, retention_neither_ab),
                ):
                    prediction = np.asarray(current["conditions"][condition]["probe_prediction"], dtype=np.int16)[:64]
                    values.append(float(np.mean(prediction[untouched] == b_target[:64][untouched])))
                    neither_values.append(
                        float(
                            np.mean(
                                (prediction[untouched] != a_target[:64][untouched])
                                & (prediction[untouched] != b_target[:64][untouched])
                            )
                        )
                    )
        by_horizon[str(horizon)] = {
            "effects_nats": {name: interval_for(values, config) for name, values in effect_values.items()},
            "full_balanced_legal_candidate_accuracy": interval_for(behavior_values, config),
            "full_square_probe_accuracy": interval_for(full_square_values, config),
            "full_exact_board_probe_accuracy": interval_for(exact_board_values, config),
            "changed_squares_current_prediction_accuracy": interval_for(current_update_values, config),
            "changed_squares_static_horizon_zero_prediction_accuracy": interval_for(static_update_values, config),
            "untouched_differing_squares_b_agreement": {
                "BA_recurrent_from_B": interval_for(ba_retention, config),
                "AB_kv_from_B": interval_for(ab_retention, config),
            },
            "untouched_differing_squares_neither": {
                "BA_recurrent_from_B": interval_for(retention_neither_ba, config),
                "AB_kv_from_B": interval_for(retention_neither_ab, config),
            },
        }
    return {
        "status": "available" if len(rows) >= 16 else "underpowered",
        "n_pairs": len(rows),
        "horizons": by_horizon,
    }


def write_metrics_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows: list[tuple[str, Any, Any, Any, Any]] = []

    def collect(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if {"estimate", "lower", "upper", "n_clusters"}.issubset(value):
                rows.append((prefix, value["estimate"], value["lower"], value["upper"], value["n_clusters"]))
            else:
                for key, child in value.items():
                    collect(f"{prefix}.{key}" if prefix else str(key), child)

    collect("", summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "estimate", "lower_95", "upper_95", "n_clusters"))
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def interval_error(interval: Mapping[str, Any]) -> tuple[float, float, float]:
    if interval.get("estimate") is None:
        return math.nan, 0.0, 0.0
    estimate = float(interval["estimate"])
    return estimate, estimate - float(interval["lower"]), float(interval["upper"]) - estimate


def save_figure(fig: Any, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(figures_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)


def plot_memory_conditions(summary: Mapping[str, Any], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"full": "#3F454B", "attention_content_zeroed": "#D9772B", "recurrent_content_zeroed": "#2F6B9A", "both_contents_zeroed": "#B8BEC4"}
    labels = {"full": "Full cache", "attention_content_zeroed": "KV content zeroed", "recurrent_content_zeroed": "Recurrent content zeroed", "both_contents_zeroed": "Both zeroed"}
    groups = list(colors)
    metrics = summary["primary"]["recipient_reference_and_zeroing"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35, 1.35]})
    fig.patch.set_facecolor("white")
    axes[0].axis("off")
    axes[0].text(0.5, 0.95, "Boundary interventions", ha="center", va="top", fontsize=13, weight="bold")
    schematic = [
        ("AA", "R_A  +  K_A", "intact A"),
        ("BA", "R_B  +  K_A", "swap recurrence"),
        ("AB", "R_A  +  K_B", "swap KV"),
        ("zero", "0(R) / 0(K)", "content erasure"),
    ]
    for index, (name, equation, note) in enumerate(schematic):
        y = 0.76 - index * 0.19
        axes[0].text(0.06, y, name, fontsize=11, weight="bold", va="center")
        axes[0].text(0.25, y, equation, fontsize=11, family="monospace", va="center")
        axes[0].text(0.25, y - 0.07, note, fontsize=8.5, color="#60676D", va="center")
    axes[0].text(0.05, 0.04, "First cache letter = recurrent donor", fontsize=8.5, color="#60676D")
    for axis, metric_name, title, ylabel in (
        (axes[1], "balanced_legal_candidate_accuracy", "Native position-sensitive preference", "Balanced legal-candidate accuracy"),
        (axes[2], "changed_square_probe_accuracy", "Linear board readout", "Changed-square accuracy"),
    ):
        estimates, low, high = zip(*(interval_error(metrics[group][metric_name]) for group in groups))
        x = np.arange(len(groups))
        axis.bar(x, estimates, color=[colors[group] for group in groups], width=0.68)
        axis.errorbar(x, estimates, yerr=np.asarray([low, high]), fmt="none", ecolor="#202428", capsize=4, lw=1.2)
        baseline = 0.5 if metric_name.startswith("balanced") else summary["primary"]["held_out_cheap_baselines"][summary["primary"]["preregistered_strongest_cheap_baseline"]]["changed_square_accuracy"]["estimate"]
        if baseline is not None:
            axis.axhline(float(baseline), color="#777", ls="--", lw=1)
        axis.set_xticks(x, [labels[group] for group in groups], rotation=24, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.set_title(title, weight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    exact = metrics["full"]["exact_board_probe_accuracy"]["estimate"]
    axes[2].text(0.98, 0.97, f"Full exact-board: {exact:.3f}" if exact is not None else "Full exact-board: unavailable", ha="right", va="top", transform=axes[2].transAxes, fontsize=8.5)
    fig.suptitle(f"Memory-channel controls (n={summary['primary']['n_pairs']} independent pairs)", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir, "01_memory_conditions")
    plt.close(fig)


def plot_donor_following(summary: Mapping[str, Any], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    blue, orange, gray = "#2F6B9A", "#D9772B", "#3F454B"
    primary = summary["primary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    effects = primary["primary_horizon_zero_effects_nats"]
    names = ("recurrent_effect", "kv_effect", "channel_contrast")
    labels = ("Recurrent", "KV", "KV - recurrent")
    estimates, low, high = zip(*(interval_error(effects[name]) for name in names))
    x = np.arange(3)
    axes[0].bar(x, estimates, color=[orange, blue, gray])
    axes[0].errorbar(x, estimates, yerr=np.asarray([low, high]), fmt="none", ecolor="#202428", capsize=4)
    axes[0].axhline(0, color="#777", ls="--", lw=1)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Shift toward B move (log-probability nats)")
    axes[0].set_title("Factorial behavioral effects", weight="bold")

    donor = primary["probe_donor_following_on_a_b_differing_squares"]
    conditions = ("BA", "AB")
    a_values = [donor[c]["a"]["estimate"] for c in conditions]
    b_values = [donor[c]["b"]["estimate"] for c in conditions]
    neither_values = [donor[c]["neither"]["estimate"] for c in conditions]
    x = np.arange(2)
    axes[1].bar(x, a_values, color="#8FAFC6", label="A label")
    axes[1].bar(x, b_values, bottom=a_values, color="#D89A64", label="B label")
    axes[1].bar(x, neither_values, bottom=np.asarray(a_values) + np.asarray(b_values), color="#C8CDD1", label="Neither")
    axes[1].set_xticks(x, ("BA: recurrent from B", "AB: KV from B"), rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction on A/B-differing squares")
    axes[1].set_title("Probe donor following", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    trans = summary["transpositions"]
    axes[2].axhline(-0.05, color="#A33", ls="--", lw=1, label="-5 pp margin")
    if trans.get("n_triplets", 0):
        state = trans["same_position"]["crossed_minus_full_square_accuracy"]
        behavior = trans["same_position"]["crossed_minus_full_legal_candidate_accuracy"]
        estimates, low, high = zip(*(interval_error(value) for value in (state, behavior)))
        axes[2].bar((0, 1), estimates, color=[blue, orange])
        axes[2].errorbar((0, 1), estimates, yerr=np.asarray([low, high]), fmt="none", ecolor="#202428", capsize=4)
        axes[2].set_xticks((0, 1), ("Probe state", "Move preference"), rotation=15)
        axes[2].set_title(f"Same-position swaps (n={trans['n_triplets']})", weight="bold")
    else:
        axes[2].text(0.5, 0.5, "Transposition control\nunavailable", ha="center", va="center")
        axes[2].set_title("Same-position swaps", weight="bold")
    axes[2].set_ylabel("Crossed minus intact accuracy")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Which donor does the hybrid state follow? (n={primary['n_pairs']} pairs)", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir, "02_donor_following")
    plt.close(fig)


def plot_continuation(summary: Mapping[str, Any], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    blue, orange, gray = "#2F6B9A", "#D9772B", "#3F454B"
    continuation = summary["continuations"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    if not continuation.get("n_pairs"):
        for axis in axes:
            axis.axis("off")
        axes[1].text(0.5, 0.5, "Continuation control unavailable", ha="center", va="center", fontsize=14)
    else:
        horizons = list(HORIZONS)
        records = continuation["horizons"]
        for effect, color, label in (
            ("recurrent_effect", orange, "Recurrent effect"),
            ("kv_effect", blue, "KV effect"),
        ):
            intervals = [records[str(h)]["effects_nats"][effect] for h in horizons]
            values, low, high = zip(*(interval_error(value) for value in intervals))
            axes[0].plot(horizons, values, color=color, marker="o", label=label)
            axes[0].fill_between(horizons, np.asarray(values) - low, np.asarray(values) + high, color=color, alpha=0.18)
        axes[0].axhline(0, color="#777", ls="--", lw=1)
        axes[0].set_ylabel("Shift toward B move (nats)")
        axes[0].set_title("Behavioral influence", weight="bold")
        axes[0].legend(frameon=False, fontsize=8)

        update_horizons = [1, 2, 4]
        for key, color, label, style in (
            ("changed_squares_current_prediction_accuracy", gray, "Updated trajectory", "-"),
            ("changed_squares_static_horizon_zero_prediction_accuracy", "#9AA0A6", "Static h=0 prediction", "--"),
        ):
            values = [records[str(h)][key]["estimate"] for h in update_horizons]
            axes[1].plot(update_horizons, values, color=color, marker="o", ls=style, label=label)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Accuracy on squares changed since h=0")
        axes[1].set_title("State updating", weight="bold")
        axes[1].legend(frameon=False, fontsize=8)

        for key, color, label in (
            ("BA_recurrent_from_B", orange, "BA: recurrent from B"),
            ("AB_kv_from_B", blue, "AB: KV from B"),
        ):
            intervals = [records[str(h)]["untouched_differing_squares_b_agreement"][key] for h in horizons]
            values = [value["estimate"] for value in intervals]
            axes[2].plot(horizons, values, color=color, marker="o", label=label)
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("Agreement with B on untouched differing squares")
        axes[2].set_title("Donor-specific persistence", weight="bold")
        axes[2].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.set_xlabel("Additional plies")
            axis.set_xticks(horizons)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Continued state processing (n={continuation['n_pairs']} independent pairs)", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_continuation")
    plt.close(fig)


def draw_board(axis: Any, targets: Sequence[int], title: str) -> None:
    from matplotlib.patches import Rectangle

    symbols = PIECE_CLASSES
    for rank in range(8):
        for file_index in range(8):
            square = rank * 8 + file_index
            color = "#EEE7DA" if (rank + file_index) % 2 == 1 else "#90A38F"
            axis.add_patch(Rectangle((file_index, rank), 1, 1, facecolor=color, edgecolor="none"))
            symbol = symbols[int(targets[square])]
            if symbol != "empty":
                axis.text(file_index + 0.5, rank + 0.5, symbol, ha="center", va="center", fontsize=12, weight="bold")
    axis.set_xlim(0, 8)
    axis.set_ylim(0, 8)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(8) + 0.5, list("abcdefgh"), fontsize=7)
    axis.set_yticks(np.arange(8) + 0.5, [str(value) for value in range(1, 9)], fontsize=7)
    axis.tick_params(length=0)
    axis.set_title(title, fontsize=10, weight="bold")


def plot_board_examples(primary_rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    ordered = sorted(primary_rows, key=lambda row: (float(row["effects"]["channel_contrast"]), str(row["pair_id"])))
    median = ordered[(len(ordered) - 1) // 2]
    failures = [
        row
        for row in primary_rows
        if not (
            behavioral_credit_for_reference(row["conditions"]["AA"], "A") == 1.0
            and behavioral_credit_for_reference(row["conditions"]["BB"], "B") == 1.0
        )
    ]
    failure = sorted(failures, key=lambda row: str(row["pair_id"]))[0] if failures else None
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.3))
    draw_board(axes[0], median["a_position"]["targets"], "Source A")
    draw_board(axes[1], median["b_position"]["targets"], "Source B")
    draw_board(axes[2], median["conditions"]["BA"]["probe_prediction"], "Decoded BA")
    draw_board(axes[3], median["conditions"]["AB"]["probe_prediction"], "Decoded AB")
    fig.suptitle(
        f"Median channel-contrast example: {median['pair_id']}\n"
        "Selected by sorted horizon-zero channel contrast; illustrations are not population estimates",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures_dir, "04_board_examples")
    plt.close(fig)
    if failure is not None:
        fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.3))
        draw_board(axes[0], failure["a_position"]["targets"], "A verifier state")
        draw_board(axes[1], failure["conditions"]["AA"]["probe_prediction"], "Decoded full A")
        draw_board(axes[2], failure["b_position"]["targets"], "B verifier state")
        draw_board(axes[3], failure["conditions"]["BB"]["probe_prediction"], "Decoded full B")
        fig.suptitle(
            f"Full-cache behavioral failure example: {failure['pair_id']}\n"
            "Lexicographically first pair where at least one intact donor did not prefer its legal candidate",
            fontsize=12,
            weight="bold",
        )
        fig.tight_layout()
        save_figure(fig, figures_dir, "04b_failure_example")
        plt.close(fig)
    return {
        "median_pair_id": median["pair_id"],
        "selection_rule": "lower median after sorting all primary pairs by channel_contrast, then pair_id",
        "failure_pair_id": failure["pair_id"] if failure else None,
        "failure_selection_rule": "lexicographically first pair where at least one full donor did not prefer its legal candidate",
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, config_path, run_dir = stage_context(args)
    manifest_path = manifest_path_for_args(args)
    manifest = load_manifest(manifest_path)
    freeze = load_and_validate_freeze(config_path, manifest_path, run_dir)
    results_dir = run_dir / "results"
    primary_map = journal_state(results_dir / "primary.jsonl", results_dir / "complete" / "primary", "pair_id")
    transposition_map = journal_state(
        results_dir / "transpositions.jsonl", results_dir / "complete" / "transpositions", "triplet_id"
    )
    continuation_map = journal_state(
        results_dir / "continuations.jsonl", results_dir / "complete" / "continuations", "pair_id"
    )
    selected = freeze["selected_ids"]
    for key, completed in (
        ("primary_pair_ids", primary_map),
        ("transposition_triplet_ids", transposition_map),
        ("continuation_pair_ids", continuation_map),
    ):
        if set(completed) != set(selected[key]):
            missing = sorted(set(selected[key]) - set(completed))
            extra = sorted(set(completed) - set(selected[key]))
            raise ProtocolError(f"analysis refuses incomplete/unfrozen {key}; missing={missing[:3]} extra={extra[:3]}")
    primary_rows = [primary_map[pair_id] for pair_id in selected["primary_pair_ids"]]
    transposition_rows = [transposition_map[value] for value in selected["transposition_triplet_ids"]]
    continuation_rows = [continuation_map[value] for value in selected["continuation_pair_ids"]]
    global _ANALYSIS_PAIR_FLAGS
    _ANALYSIS_PAIR_FLAGS = {str(pair["pair_id"]): pair for pair in manifest["test_pairs"]}
    primary_summary = analyze_primary(primary_rows, config)
    primary_summary["preregistered_strongest_cheap_baseline"] = freeze["strongest_cheap_baseline"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": utc_now(),
        "freeze_sha256": freeze["freeze_sha256"],
        "schedule": freeze["schedule"],
        "primary": primary_summary,
        "transpositions": analyze_transpositions(transposition_rows, config),
        "continuations": analyze_continuations(continuation_rows, config),
        "claim_boundaries": [
            "Cache swaps measure boundary-carried memory influence; both channel types remain active afterward.",
            "Linear recoverability is accessibility evidence, not proof that the decoded features mediate behavior.",
            "Crossed caches have no unique gold board and are reported against A, B, and neither.",
            "Matched real-game contrasts are a diagnostic distribution, not all human chess.",
            "Piece-and-rule state excludes repetition and complete draw-rule history.",
        ],
    }
    figures_dir = run_dir / "figures"
    plot_memory_conditions(summary, figures_dir)
    plot_donor_following(summary, figures_dir)
    plot_continuation(summary, figures_dir)
    summary["board_examples"] = plot_board_examples(primary_rows, figures_dir)
    summary["figure_hashes"] = {
        path.name: sha256_file(path) for path in sorted(figures_dir.iterdir()) if path.is_file()
    }
    atomic_write_json(run_dir / "summary.json", summary)
    write_metrics_csv(run_dir / "metrics.csv", summary)
    log_time(run_dir, "analyze", started)
    print(json.dumps({"summary": str(run_dir / 'summary.json'), "figures": summary["figure_hashes"]}, indent=2))


# ---------------------------------------------------------------------------
# Independent artifact verification
# ---------------------------------------------------------------------------


def verify_candidate_result(result: Mapping[str, Any], expected_ids: Sequence[int]) -> None:
    if [int(value) for value in result["token_ids"]] != [int(value) for value in expected_ids]:
        raise ProtocolError("saved candidate token IDs differ from the frozen manifest")
    values = [float(value) for value in result["token_logprobs"]]
    if len(values) != len(expected_ids) or not all(math.isfinite(value) for value in values):
        raise ProtocolError("candidate log-probability vector is incomplete or non-finite")
    if abs(sum(values) - float(result["sum_logprob"])) > 1e-8:
        raise ProtocolError("saved candidate sum does not equal all token log-probabilities")
    if abs(float(np.mean(values)) - float(result["mean_logprob"])) > 1e-8:
        raise ProtocolError("saved candidate mean is inconsistent")


def verify_condition_results(
    conditions: Mapping[str, Any],
    expected_conditions: Iterable[str],
    a_ids: Sequence[int],
    b_ids: Sequence[int],
) -> None:
    if set(conditions) != set(expected_conditions):
        raise ProtocolError("saved result has a missing or extra intervention condition")
    for condition in conditions.values():
        verify_candidate_result(condition["a_candidate"], a_ids)
        verify_candidate_result(condition["b_candidate"], b_ids)
        d = float(condition["b_candidate"]["sum_logprob"]) - float(
            condition["a_candidate"]["sum_logprob"]
        )
        if abs(d - float(condition["d_b_minus_a"])) > 1e-8:
            raise ProtocolError("saved B-minus-A candidate score is inconsistent")
        prediction = condition["probe_prediction"]
        scores = condition.get("probe_scores")
        if len(prediction) != 70 or (scores is not None and len(scores) != HEAD_OFFSETS[-1]):
            raise ProtocolError("saved probe prediction/scores are incomplete")
        if scores is not None:
            decoded = decode_head_scores(np.asarray(scores, dtype=np.float64)[None, :])[0]
            if decoded.tolist() != [int(value) for value in prediction]:
                raise ProtocolError("saved probe labels do not decode from the saved scores")


def verify_result_artifacts(
    config: StudyConfig,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    results_dir = run_dir / "results"
    primary = journal_state(results_dir / "primary.jsonl", results_dir / "complete" / "primary", "pair_id")
    transpositions = journal_state(
        results_dir / "transpositions.jsonl", results_dir / "complete" / "transpositions", "triplet_id"
    )
    continuations = journal_state(
        results_dir / "continuations.jsonl", results_dir / "complete" / "continuations", "pair_id"
    )
    selected = freeze["selected_ids"]
    expected_sets = {
        "primary": set(selected["primary_pair_ids"]),
        "transpositions": set(selected["transposition_triplet_ids"]),
        "continuations": set(selected["continuation_pair_ids"]),
    }
    actual_sets = {
        "primary": set(primary),
        "transpositions": set(transpositions),
        "continuations": set(continuations),
    }
    if actual_sets != expected_sets:
        raise ProtocolError("held-out result coverage differs from the exact frozen cluster IDs")
    pair_map, triplet_map = result_maps(manifest)
    for pair_id, row in primary.items():
        pair = pair_map[pair_id]
        if row["freeze_sha256"] != freeze["freeze_sha256"]:
            raise ProtocolError("primary row belongs to a different freeze")
        verify_condition_results(
            row["conditions"],
            condition_ids_for_pair(),
            pair["a_candidate_ids"],
            pair["b_candidate_ids"],
        )
        recomputed = factorial_effects(row["conditions"])
        if any(abs(recomputed[key] - float(row["effects"][key])) > 1e-8 for key in recomputed):
            raise ProtocolError("primary factorial effect was not computed from saved candidate scores")
        if row.get("baselines") is None or set(row["baselines"]) != {"A", "B"}:
            raise ProtocolError("primary row is missing held-out baseline predictions")
        for donor in ("A", "B"):
            if set(row["baselines"][donor]) != {
                "initial_board",
                "training_frequency",
                "orderless",
                "last_move",
                "permutation",
            }:
                raise ProtocolError("primary baseline family differs from the frozen protocol")
            for baseline in row["baselines"][donor].values():
                if len(baseline["prediction"]) != 70:
                    raise ProtocolError("primary baseline prediction is incomplete")
    for triplet_id, row in transpositions.items():
        triplet = triplet_map[triplet_id]
        verify_condition_results(
            row["conditions"],
            ("AA", "PP", "BB", "PA", "AP", "BA", "AB"),
            triplet["a_candidate_ids"],
            triplet["b_candidate_ids"],
        )
        recomputed = factorial_effects(row["conditions"])
        if any(
            abs(recomputed[key] - float(row["different_position_effects"][key])) > 1e-8
            for key in recomputed
        ):
            raise ProtocolError("transposition factorial effect is inconsistent")
    for pair_id, row in continuations.items():
        pair = pair_map[pair_id]
        if row["primary_record_sha256"] != primary[pair_id]["record_sha256"]:
            raise ProtocolError("continuation horizon zero does not point to the exact primary record")
        previous: list[int] = []
        for horizon in HORIZONS:
            saved = row["horizons"][str(horizon)]
            candidates = pair if horizon == 0 else pair["continuation"]["horizons"][str(horizon)]["candidates"]
            verify_condition_results(
                saved["conditions"],
                condition_ids_for_pair(),
                candidates["a_candidate_ids"],
                candidates["b_candidate_ids"],
            )
            recomputed = factorial_effects(saved["conditions"])
            if any(abs(recomputed[key] - float(saved["effects"][key])) > 1e-8 for key in recomputed):
                raise ProtocolError("continuation factorial effect is inconsistent")
            if horizon > 0:
                cumulative = [int(value) for value in saved["cumulative_advance_token_ids"]]
                incremental = [int(value) for value in saved["incremental_advance_token_ids"]]
                if cumulative[: len(previous)] != previous or cumulative[len(previous) :] != incremental:
                    raise ProtocolError("continuation incremental token journal is inconsistent")
                previous = cumulative
    evaluation_manifest = read_json(results_dir / "evaluation_manifest.json")
    expected_hashes = {
        "primary": sha256_file(results_dir / "primary.jsonl"),
        "transpositions": sha256_file(results_dir / "transpositions.jsonl"),
        "continuations": sha256_file(results_dir / "continuations.jsonl"),
    }
    if evaluation_manifest["journal_hashes"] != expected_hashes:
        raise ProtocolError("evaluation manifest journal hashes no longer match")
    if evaluation_manifest["model_parameter_digest_before"] != evaluation_manifest["model_parameter_digest_after"]:
        raise ProtocolError("evaluation model immutability digest failed")
    if evaluation_manifest["model_parameter_digest_before"] != freeze["model_parameter_digest"]:
        raise ProtocolError("evaluation parameter digest differs from the freeze")
    return {
        "completed_counts": {key: len(value) for key, value in actual_sets.items()},
        "journal_hashes": expected_hashes,
        "all_saved_candidate_scores_recomputed": True,
        "all_factorial_effects_recomputed": True,
        "all_completion_markers_verified": True,
    }


def cmd_verify(args: argparse.Namespace) -> None:
    started = time.monotonic()
    config, config_path, run_dir = stage_context(args)
    output_path = run_dir / "verification.json"
    if output_path.exists() and not args.resume:
        raise ProtocolError("verification artifact exists; pass --resume to recompute it")
    manifest_path = manifest_path_for_args(args)
    manifest = load_manifest(manifest_path)
    tokenizer = resolve_manifest_tokenizer(manifest, args.allow_remote_downloads)
    data_audit = validate_data_manifest(config, manifest, tokenizer)
    freeze = load_and_validate_freeze(config_path, manifest_path, run_dir)
    preflight = read_json(run_dir / "preflight.json")
    pilot = read_json(run_dir / "pilot.json")
    probe_config = read_json(run_dir / "probe_config.json")
    if not preflight.get("all_gates_passed") or not pilot.get("gate_b_behavior_passed"):
        raise ProtocolError("verification found a failed mandatory engineering/behavior gate")
    result_audit = verify_result_artifacts(config, manifest, freeze, run_dir)
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise ProtocolError("analysis artifacts are absent")
    summary = read_json(summary_path)
    if summary.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ProtocolError("summary belongs to a different freeze")
    figure_dir = run_dir / "figures"
    required_stems = ("01_memory_conditions", "02_donor_following", "03_continuation", "04_board_examples")
    missing_figures = [
        f"{stem}.{suffix}"
        for stem in required_stems
        for suffix in ("svg", "pdf", "png")
        if not (figure_dir / f"{stem}.{suffix}").exists()
    ]
    if missing_figures:
        raise ProtocolError(f"required exported figure formats are missing: {missing_figures}")
    current_figure_hashes = {
        path.name: sha256_file(path) for path in sorted(figure_dir.iterdir()) if path.is_file()
    }
    if current_figure_hashes != summary.get("figure_hashes"):
        raise ProtocolError("saved figure hashes differ from the analyzed summary")
    required_artifacts = [
        "environment.json",
        "config.json",
        "manifest.json",
        "data_audit.json",
        "preflight.json",
        "probe_config.json",
        "probe_weights.npz",
        "baseline_probe_weights.npz",
        "pilot.json",
        "freeze.json",
        "features/train.npz",
        "features/development.npz",
        "results/primary.jsonl",
        "results/transpositions.jsonl",
        "results/continuations.jsonl",
        "summary.json",
        "metrics.csv",
        "time_log.csv",
        "decisions.md",
    ]
    missing = [value for value in required_artifacts if not (run_dir / value).exists()]
    if missing:
        raise ProtocolError(f"required run artifacts are missing: {missing}")
    availability = {
        "primary": "complete",
        "probe": "informative" if probe_config["gate_c_probe_passed"] else "linear_readout_gate_failed",
        "transposition": summary["transpositions"]["status"],
        "continuation": summary["continuations"]["status"],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "verified_at": utc_now(),
        "freeze_sha256": freeze["freeze_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "data_audit": {"all_checks_passed": data_audit["all_checks_passed"], "sha256": sha256_file(run_dir / "data_audit.json")},
        "gates": {
            "gate_a_engineering": bool(preflight["all_gates_passed"]),
            "gate_b_behavior": bool(pilot["gate_b_behavior_passed"]),
            "gate_c_probe": bool(probe_config["gate_c_probe_passed"]),
        },
        "result_audit": result_audit,
        "section_status": availability,
        "artifact_hashes": {
            value: sha256_file(run_dir / value) for value in required_artifacts
        },
        "figure_hashes": current_figure_hashes,
        "all_mandatory_checks_passed": True,
        "claim_ceiling": "causal influence of boundary-carried cache channels in this frozen Qwen/UCI assay; not sole storage, pure state, optimal chess, or probe-mediated mechanism",
    }
    atomic_write_json(output_path, payload)
    log_time(run_dir, "verify", started)
    print(json.dumps(payload, indent=2))


def add_common_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-remote-downloads",
        action="store_true",
        help="Allow only the frozen Lichess/Hugging Face resources to be downloaded.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Where Does Qwen3.5 Keep Its World? Frozen chess-memory study"
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "prepare": cmd_prepare,
        "validate-data": cmd_validate_data,
        "preflight": cmd_preflight,
        "extract-features": cmd_extract_features,
        "fit-probe": cmd_fit_probe,
        "pilot": cmd_pilot,
        "freeze": cmd_freeze,
        "evaluate": cmd_evaluate,
        "analyze": cmd_analyze,
        "verify": cmd_verify,
    }
    for name, handler in handlers.items():
        stage_parser = subparsers.add_parser(name)
        add_common_stage_arguments(stage_parser)
        stage_parser.set_defaults(handler=handler)
        if name == "prepare":
            stage_parser.add_argument("--use-fallback", action="store_true")
            stage_parser.add_argument(
                "--fallback-reason",
                help="Path to the preserved JSON artifact documenting a 4B resource preflight failure.",
            )
        if name == "freeze":
            stage_parser.add_argument(
                "--remaining-seconds",
                required=True,
                type=float,
                help="Actual remaining compute/wall-clock budget used for the preregistered runtime-only schedule choice.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ProtocolError as exc:
        print(f"PROTOCOL ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
