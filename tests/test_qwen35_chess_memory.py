from __future__ import annotations

import copy
import json
import numpy as np
import pytest
import torch

import qwen35_chess_memory as study


class CharacterTokenizer:
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


def make_position(moves, full_suffix, game_name, split, tokenizer):
    full = list(moves) + list(full_suffix)
    return study.position_payload(full, len(moves), game_name, 1, split, tokenizer)


def mock_snapshot() -> study.CacheSnapshot:
    linear_state = {
        "number_of_states": 1,
        "conv_states": {0: torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)},
        "recurrent_states": {0: torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)},
        "is_conv_states_initialized": {0: True},
        "is_recurrent_states_initialized": {0: True},
        "has_previous_state": {0: True},
        "conv_kernel_size": {0: 4},
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "record_past": False,
    }
    attention_state = {
        "keys": torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        "values": torch.arange(18, 36, dtype=torch.float32).reshape(1, 2, 3, 3),
        "is_initialized": True,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    snapshot = study.CacheSnapshot(
        sequence_length=3,
        next_position=3,
        attention_mask_length=3,
        layer_types=("linear_attention", "full_attention"),
        layers=(
            study.LayerSnapshot(0, "linear_attention", "MockLinear", linear_state),
            study.LayerSnapshot(1, "full_attention", "MockAttention", attention_state),
        ),
        rope_deltas=None,
    )
    study.validate_snapshot(snapshot)
    return snapshot


def test_uci_replay_square_order_and_class_round_trip():
    board = study.strict_board_from_moves(["e2e4", "c7c5", "g1f3"])
    labels = study.piece_labels(board)
    assert study.SQUARE_NAMES[0] == "a1"
    assert study.SQUARE_NAMES[63] == "h8"
    assert labels[study.import_chess().E4] == study.PIECE_TO_INDEX["P"]
    assert labels[study.import_chess().C5] == study.PIECE_TO_INDEX["p"]
    assert labels[study.import_chess().F3] == study.PIECE_TO_INDEX["N"]
    assert len(study.board_targets(board)) == 70


def test_castling_en_passant_promotion_side_and_rights_fixtures():
    chess = study.import_chess()
    castled = study.strict_board_from_moves(
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1"]
    )
    assert castled.piece_at(chess.G1).symbol() == "K"
    assert castled.piece_at(chess.F1).symbol() == "R"
    assert not castled.has_kingside_castling_rights(chess.WHITE)
    assert castled.turn == chess.BLACK

    before_ep = study.strict_board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5"])
    targets = study.board_targets(before_ep)
    assert targets[-1] == chess.D6 + 1
    after_ep = study.strict_board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"])
    assert after_ep.piece_at(chess.D5) is None
    assert after_ep.piece_at(chess.D6).symbol() == "P"
    assert study.board_targets(after_ep)[-1] == 0

    raw_ep_without_capture = study.strict_board_from_moves(["e2e4"])
    assert raw_ep_without_capture.fen(en_passant="legal").split()[3] == "-"
    assert raw_ep_without_capture.fen(en_passant="fen").split()[3] == "e3"

    promoted = study.strict_board_from_moves(
        ["a2a4", "h7h5", "a4a5", "h5h4", "a5a6", "h4h3", "a6b7", "h3g2", "b7a8q"]
    )
    assert promoted.piece_at(chess.A8).symbol() == "Q"


def test_illegal_null_and_nonstandard_start_are_rejected(tmp_path):
    with pytest.raises(study.ProtocolError, match="illegal/null"):
        study.strict_board_from_moves(["e2e5"])
    with pytest.raises(study.ProtocolError, match="illegal/null"):
        study.strict_board_from_moves(["0000"])

    zstandard = pytest.importorskip("zstandard")
    pgn = """[Event \"fixture\"]
[Variant \"Standard\"]
[SetUp \"1\"]
[FEN \"8/8/8/8/8/8/8/K6k w - - 0 1\"]

1. Ka2 Kh2 2. Ka3 Kh3 3. Ka4 Kh4 4. Ka5 Kh5 5. Ka6 Kh6 6. Ka7 Kh7 7. Ka8 Kh8 8. Kb8 Kg8 *
"""
    archive = tmp_path / "fixture.pgn.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(pgn.encode()))
    rows = list(study.iter_lichess_games(archive, maximum_games=1))
    assert rows[0]["error"] == "nonstandard_start_or_variant"


def test_split_determinism_target_exclusion_and_intentional_transposition(tokenizer):
    full = ["g1f3", "g8f6", "b1c3", "b8c6", "e2e4", "e7e5"]
    assert study.partition_for_game(full, 11) == study.partition_for_game(list(full), 11)
    transposed = ["b1c3", "b8c6", "g1f3", "g8f6"]
    original = ["g1f3", "g8f6", "b1c3", "b8c6"]
    board_a = study.strict_board_from_moves(original)
    board_p = study.strict_board_from_moves(transposed)
    assert original != transposed
    assert study.six_field_fen(board_a) == study.six_field_fen(board_p)
    assert study.target_key(study.board_targets(board_a)) == study.target_key(study.board_targets(board_p))
    assert study.stable_hash(" ".join(full)) == study.stable_hash(" ".join(list(full)))


def test_opposing_candidates_are_exclusive_and_context_length_matched(tokenizer):
    a = make_position(["e2e4", "e7e5"], ["g1f3"] * 4, "a" * 24, "pilot", tokenizer)
    b = make_position(["d2d4", "d7d5"], ["g1f3"] * 4, "b" * 24, "pilot", tokenizer)
    result = study.find_opposing_candidates(tokenizer, a, b)
    assert result is not None
    assert result["a_move"] in a["legal_moves"] and result["a_move"] not in b["legal_moves"]
    assert result["b_move"] in b["legal_moves"] and result["b_move"] not in a["legal_moves"]
    assert len(result["a_candidate_ids"]) == len(result["b_candidate_ids"])
    assert study.candidate_continuation_ids(tokenizer, a["context_text"], result["a_move"]) == result["a_candidate_ids"]
    assert study.candidate_continuation_ids(tokenizer, b["context_text"], result["a_move"]) == result["a_candidate_ids"]


def test_continuation_rejects_a_move_legal_for_only_one_donor(tokenizer):
    a = make_position(["e2e4", "e7e5"], ["d2d4"] * 4, "a" * 24, "test", tokenizer)
    b = make_position(["d2d4", "d7d5"], ["g1f3"] * 4, "b" * 24, "test", tokenizer)
    advanced = study.advance_position(a, ["d2d4"], tokenizer)
    assert advanced["position_id"] == study.stable_hash("advanced", a["position_id"], "d2d4")[:24]
    assert advanced["cutoff_ply"] == a["cutoff_ply"] + 1
    with pytest.raises(study.ProtocolError, match="illegal/null"):
        study.advance_position(b, ["d2d4"], tokenizer)


def test_same_position_contract_uses_six_fen_fields_and_token_distance(tokenizer):
    first = ["g1f3", "g8f6", "b1c3", "b8c6"]
    same = ["b1c3", "b8c6", "g1f3", "g8f6"]
    assert study.six_field_fen(study.strict_board_from_moves(first)) == study.six_field_fen(
        study.strict_board_from_moves(same)
    )
    returned = ["g1f3", "g8f6", "f3g1", "f6g8"]
    assert study.piece_labels(study.strict_board_from_moves(returned)) == study.piece_labels(
        study.import_chess().Board()
    )
    assert study.six_field_fen(study.strict_board_from_moves(returned)) != study.six_field_fen(
        study.import_chess().Board()
    )
    text_a = study.encode_ids(tokenizer, study.render_context(first))
    text_p = study.encode_ids(tokenizer, study.render_context(same))
    assert len(text_a) == len(text_p)
    assert study.token_hamming_distance(text_a, text_p) > 0
    assert abs(7 - 9) <= 2 and abs(7 - 10) > 2


def test_mock_snapshot_copies_recurrent_conv_and_kv_without_aliasing():
    source = mock_snapshot()
    clone = study.clone_snapshot(source)
    study.assert_storage_independent(source, clone)
    assert study.snapshot_digest(source) == study.snapshot_digest(clone)
    assembled = study.assemble_snapshot(source, clone)
    assert study.snapshot_digest(source) == study.snapshot_digest(assembled)
    assert study.validate_snapshot(source)["recurrent_bytes"] > 0
    assert study.validate_snapshot(source)["attention_bytes"] > 0


def test_zeroing_preserves_metadata_and_only_removes_selected_contents():
    source = mock_snapshot()
    source_digest = study.snapshot_digest(source)
    recurrent_zero = study.zero_channels(source, recurrent=True)
    attention_zero = study.zero_channels(source, attention=True)
    both_zero = study.zero_channels(source, recurrent=True, attention=True)
    for changed in (recurrent_zero, attention_zero, both_zero):
        assert changed.sequence_length == source.sequence_length
        assert changed.next_position == source.next_position
        assert changed.attention_mask_length == source.attention_mask_length
    assert all(torch.count_nonzero(tensor) == 0 for _, tensor in study.iter_state_tensors(recurrent_zero.layers[0].state))
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(study.iter_state_tensors(source.layers[1].state), study.iter_state_tensors(recurrent_zero.layers[1].state)))
    assert all(torch.count_nonzero(tensor) == 0 for _, tensor in study.iter_state_tensors(attention_zero.layers[1].state))
    assert all(torch.count_nonzero(tensor) == 0 for layer in both_zero.layers for _, tensor in study.iter_state_tensors(layer.state))
    assert study.snapshot_digest(source) == source_digest


def test_scoring_sums_candidate_and_delimiter_tokens_but_not_query():
    class LookupHead(torch.nn.Module):
        def forward(self, hidden):
            return hidden * torch.arange(10, dtype=torch.float32).view(1, -1)

    backend = study.FrozenQwenBackend.__new__(study.FrozenQwenBackend)
    backend.lm_head = LookupHead()
    backend._consume_tokens = lambda snapshot, ids: (snapshot, torch.tensor([[float(ids[-1])]]))
    result = backend.score_candidate(None, [8, 3], [2, 4, 5])
    expected = []
    for hidden_value, target in ((3.0, 2), (2.0, 4), (4.0, 5)):
        logits = hidden_value * torch.arange(10, dtype=torch.float32)
        expected.append(float(torch.log_softmax(logits, dim=-1)[target]))
    assert result["token_ids"] == [2, 4, 5]
    assert len(result["token_logprobs"]) == 3
    assert result["token_logprobs"] == pytest.approx(expected)
    assert result["sum_logprob"] == pytest.approx(sum(expected))


def test_scoring_branch_cannot_contaminate_another_or_trajectory():
    source = mock_snapshot()
    digest = study.snapshot_digest(source)
    first = study.clone_snapshot(source)
    second = study.clone_snapshot(source)
    next(iter(study.iter_state_tensors(first.layers[0].state)))[1].add_(100)
    assert study.snapshot_digest(second) == digest
    assert study.snapshot_digest(source) == digest
    assert study.snapshot_digest(first) != digest


def test_probe_uses_train_statistics_reconstructs_linear_fixture_and_permutation_is_train_only():
    rows = 12
    targets = np.zeros((rows, 70), dtype=np.int16)
    for head, classes in enumerate(study.HEAD_CLASS_COUNTS):
        targets[:, head] = np.arange(rows) % min(classes, rows)
    features = study.targets_to_one_hot(targets)
    readout = study.fit_ridge_readout(features, targets, ridge_lambda=1e-6, std_floor=1e-6)
    assert np.array_equal(readout.predict(features), targets)
    assert np.allclose(readout.feature_mean, features.mean(axis=0))
    shifted_development = features + 1000.0
    _ = readout.predict(shifted_development)
    assert np.allclose(readout.feature_mean, features.mean(axis=0))
    permutation = np.random.default_rng(11).permutation(rows)
    assert sorted(permutation.tolist()) == list(range(rows))
    assert shifted_development.shape[0] not in permutation


def test_factorial_effect_signs_and_neither_stays_in_donor_denominator():
    conditions = {
        "AA": {"d_b_minus_a": -2.0},
        "BA": {"d_b_minus_a": 0.0},
        "AB": {"d_b_minus_a": -1.0},
        "BB": {"d_b_minus_a": 2.0},
    }
    effects = study.factorial_effects(conditions)
    assert effects["recurrent_effect"] == pytest.approx(2.5)
    assert effects["kv_effect"] == pytest.approx(1.5)
    assert effects["channel_contrast"] == pytest.approx(-1.0)
    row = {
        "a_position": {"targets": [0] * 70},
        "b_position": {"targets": [1] * 64 + [0] * 6},
        "conditions": {
            "BA": {
                "probe_prediction": [2] * 64 + [0] * 6,
                "probe_scores": [0.0] * study.HEAD_OFFSETS[-1],
            }
        },
    }
    following = study.donor_following_for_condition(row, "BA")
    assert following["a"] == 0.0
    assert following["b"] == 0.0
    assert following["neither"] == 1.0
    assert following["a"] + following["b"] + following["neither"] == 1.0


def test_bootstrap_resamples_cluster_values_and_empty_is_unavailable():
    empty = study.bootstrap_mean_interval([], replicates=100, seed=41)
    assert empty == {"estimate": None, "lower": None, "upper": None, "n_clusters": 0}
    values = [0.0, 1.0, 0.5]
    first = study.bootstrap_mean_interval(values, replicates=500, seed=41)
    second = study.bootstrap_mean_interval(values, replicates=500, seed=41)
    assert first == second
    assert first["n_clusters"] == 3
    assert first["estimate"] == pytest.approx(0.5)


def test_freeze_tampering_and_resume_duplicate_are_rejected(tmp_path):
    base = {
        "model_revision": "a" * 40,
        "tokenizer": "tok",
        "script": "script",
        "data": "data",
        "conditions": ["AA", "BB", "BA", "AB"],
        "probe": "probe",
    }
    frozen = copy.deepcopy(base)
    frozen["freeze_sha256"] = study.freeze_hash(frozen)
    study.validate_freeze_hash(frozen)
    for key in base:
        changed = copy.deepcopy(frozen)
        changed[key] = "changed" if not isinstance(changed[key], list) else ["changed"]
        with pytest.raises(study.ProtocolError, match="freeze artifact hash"):
            study.validate_freeze_hash(changed)

    journal = tmp_path / "results.jsonl"
    marker_dir = tmp_path / "markers"
    record = {"pair_id": "pair-1", "value": 7}
    study.append_complete_result(journal, marker_dir, "pair_id", record)
    recovered = study.journal_state(journal, marker_dir, "pair_id")
    assert recovered["pair-1"]["value"] == 7
    with pytest.raises(study.ProtocolError, match="duplicate"):
        study.append_complete_result(journal, marker_dir, "pair_id", {"pair_id": "pair-1"})
    lines = journal.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["record_sha256"] == recovered["pair-1"]["record_sha256"]
