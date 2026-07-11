# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Real-kernel, mocked-model tests for PEAGLEDraftWorker.draft_forward (the V2
port). Only self.draft_runner.forward(...) is mocked -- everything else
(fused_parallel_draft_input, the confidence-masking kernel, the ForwardBatch
repeat_interleave expansion, and the topk=1 chain fast-path assembly) runs
for real on whatever device is available. Follows the same
object.__new__(...)-plus-manual-wiring pattern as
test_eagle_worker_v2_topk1_fastpath.py, since PEAGLEDraftWorker needs no
different a test harness than vanilla EagleDraftWorker does.

No model checkpoint is required.
"""

import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.p_eagle_worker import PEAGLEDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_worker(
    num_steps: int,
    bs_cap: int = 8,
    enable_dsl: bool = False,
    dsl_threshold: float = 2.0,
    hidden_dim: int = 32,
    vocab_size: int = 128,
):
    """Construct a PEAGLEDraftWorker without running __init__ (no model
    load), wiring exactly the attributes draft_forward touches."""
    worker = object.__new__(PEAGLEDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_steps + 1
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=bs_cap)),
        max_running_requests=bs_cap,
        speculative_use_rejection_sampling=False,
    )
    worker._rebuild_topk1_chain_buffers()

    worker.enable_dsl = enable_dsl
    worker.dsl_threshold = dsl_threshold
    worker.hot_token_id = None
    worker._mask_token_id = 2
    worker._h_shared = torch.zeros(hidden_dim, dtype=torch.float32, device=DEVICE)

    fake_embed = torch.randn(vocab_size, hidden_dim, device=DEVICE)
    worker._embed_table_override = fake_embed
    # _embed_table is a @property on the real class; patch the instance's
    # class lookup by monkeypatching the bound method target directly.
    type(worker)._embed_table = property(lambda self: self._embed_table_override)

    worker.draft_attn_backend = SimpleNamespace(
        attn_backends=[SimpleNamespace()],
        init_forward_metadata=lambda fb: None,
    )

    return worker


def _mark_forward_metadata_ready(forward_batch, replan_equivalent=False):
    forward_batch.forward_metadata_ready = True
    forward_batch.forward_metadata_planned_bs = forward_batch.batch_size
    forward_batch.forward_metadata_planned_num_tokens = (
        forward_batch.input_ids.shape[0]
        if forward_batch.input_ids is not None
        else 0
    )
    forward_batch.forward_metadata_replan_equivalent = replan_equivalent


def _make_forward_batch(worker, bs: int, hidden_dim: int, vocab_size: int, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden_states = torch.randn(
        bs, hidden_dim, device=DEVICE, generator=g if DEVICE == "cuda" else None
    )
    spec_info = EagleDraftInput(
        hidden_states=hidden_states,
        bonus_tokens=torch.randint(
            0, vocab_size, (bs,), device=DEVICE, dtype=torch.int32
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=spec_info,
        batch_size=bs,
        input_ids=torch.randint(0, vocab_size, (bs,), device=DEVICE, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, device=DEVICE, dtype=torch.int64),
        seq_lens=torch.full((bs,), 10, device=DEVICE, dtype=torch.int64),
        positions=torch.full((bs,), 10, device=DEVICE, dtype=torch.int64),
        forward_metadata_ready=False,
        forward_metadata_planned_bs=None,
        forward_metadata_planned_num_tokens=None,
        forward_metadata_replan_equivalent=False,
        out_cache_loc=torch.arange(
            bs * worker.speculative_num_steps, device=DEVICE, dtype=torch.int64
        ),
    )
    forward_batch.mark_forward_metadata_ready = (
        lambda replan_equivalent=False: _mark_forward_metadata_ready(
            forward_batch, replan_equivalent
        )
    )
    return forward_batch


class TestPEagleV2WorkerDraftForward(CustomTestCase):
    def test_output_shape_contract(self):
        bs, K, hidden_dim, vocab_size = 3, 4, 32, 128
        worker = _make_worker(num_steps=K, hidden_dim=hidden_dim, vocab_size=vocab_size)
        forward_batch = _make_forward_batch(worker, bs, hidden_dim, vocab_size)

        captured_logits_batch_size = {}

        def fake_forward(fb, skip_attn_backend_init=False):
            # This is the one thing draft_forward cannot exercise for real:
            # a real draft model's forward pass. Everything upstream of this
            # call (the K-expansion of fb) and downstream of it (sampling +
            # tuple assembly) is real.
            captured_logits_batch_size["bs_times_k"] = fb.input_ids.shape[0]
            logits = torch.randn(
                fb.input_ids.shape[0], vocab_size, device=DEVICE
            )
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

        worker.draft_runner = SimpleNamespace(forward=fake_forward)

        parent_list, top_scores_index, draft_tokens, draft_probs = (
            worker.draft_forward(forward_batch)
        )

        self.assertEqual(captured_logits_batch_size["bs_times_k"], bs * K)
        self.assertEqual(tuple(parent_list.shape), (bs, K))
        self.assertEqual(tuple(top_scores_index.shape), (bs, K))
        self.assertEqual(tuple(draft_tokens.shape), (bs, K))
        self.assertEqual(draft_tokens.dtype, torch.int64)
        self.assertIsNone(draft_probs)

    def test_input_ids_none_gets_real_placeholder_not_left_none(self):
        """A live server caught this: input_ids can legitimately be None on
        entry (the base sequential loop never reads it before overwriting
        it fresh each step, so the framework tolerates that). But
        eager_runner.py's load_batch derives raw_num_tokens from
        input_ids.shape[0], falling back to 0 if it's None -- and several
        registry buffer slots are sized off that token-count axis. Leaving
        input_ids as None (rather than building a same-shape placeholder)
        crashed with "tensor a (0) must match tensor b (4)" on first real
        request. input_ids must always be a real, correctly-shaped tensor
        by the time draft_runner.forward() is called, even though its
        content doesn't drive the actual computation (parallel_inputs does)."""
        bs, K, hidden_dim, vocab_size = 2, 3, 16, 64
        worker = _make_worker(num_steps=K, hidden_dim=hidden_dim, vocab_size=vocab_size)
        forward_batch = _make_forward_batch(worker, bs, hidden_dim, vocab_size)
        forward_batch.input_ids = None

        captured = {}

        def fake_forward(fb, skip_attn_backend_init=False):
            captured["input_ids"] = fb.input_ids
            logits = torch.randn(fb.input_ids.shape[0], vocab_size, device=DEVICE)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

        worker.draft_runner = SimpleNamespace(forward=fake_forward)
        worker.draft_forward(forward_batch)

        self.assertIsNotNone(captured["input_ids"])
        self.assertEqual(captured["input_ids"].shape[0], bs * K)
        # Restored to None afterward, matching the pre-call state.
        self.assertIsNone(forward_batch.input_ids)

    def test_forward_batch_restored_after_call(self):
        """draft_forward mutates forward_batch in place to K-expand it for
        the single batched call -- it must restore the original
        batch-sized tensors before returning, or the caller (draft(), which
        goes on to build the tree mask from the *original* batch_size) sees
        corrupted shapes."""
        bs, K, hidden_dim, vocab_size = 2, 3, 16, 64
        worker = _make_worker(num_steps=K, hidden_dim=hidden_dim, vocab_size=vocab_size)
        forward_batch = _make_forward_batch(worker, bs, hidden_dim, vocab_size)

        orig_input_ids = forward_batch.input_ids.clone()
        orig_req_pool_indices = forward_batch.req_pool_indices.clone()
        orig_seq_lens = forward_batch.seq_lens.clone()
        orig_positions = forward_batch.positions.clone()
        orig_hidden_states = forward_batch.spec_info.hidden_states.clone()

        captured_batch_size_during_call = {}

        def fake_forward(fb, skip_attn_backend_init=False):
            captured_batch_size_during_call["value"] = fb.batch_size
            logits = torch.randn(fb.input_ids.shape[0], vocab_size, device=DEVICE)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

        worker.draft_runner = SimpleNamespace(forward=fake_forward)
        worker.draft_forward(forward_batch)

        # batch_size is a separate field from the tensor shapes -- the eager
        # runner's buffer registry reads it directly (not input_ids.shape[0])
        # to size its copy buffers, so it must be expanded during the call
        # and restored after, same as the tensors.
        self.assertEqual(captured_batch_size_during_call["value"], bs * K)
        self.assertEqual(forward_batch.batch_size, bs)
        self.assertEqual(forward_batch.input_ids.shape[0], bs)
        self.assertTrue(torch.equal(forward_batch.input_ids, orig_input_ids))
        self.assertTrue(
            torch.equal(forward_batch.req_pool_indices, orig_req_pool_indices)
        )
        self.assertTrue(torch.equal(forward_batch.seq_lens, orig_seq_lens))
        self.assertTrue(torch.equal(forward_batch.positions, orig_positions))
        self.assertEqual(forward_batch.spec_info.hidden_states.shape[0], bs)
        self.assertTrue(
            torch.equal(forward_batch.spec_info.hidden_states, orig_hidden_states)
        )

    def test_position0_uses_real_hidden_positions_1plus_share_context(self):
        """Exercises the real fused_parallel_draft_input kernel: position 0
        of the expanded input must equal h_fused[seq] + embed[last_token];
        positions 1..K-1 must all equal the same h_shared + embed[MASK]."""
        bs, K, hidden_dim, vocab_size = 2, 3, 16, 64
        worker = _make_worker(num_steps=K, hidden_dim=hidden_dim, vocab_size=vocab_size)
        forward_batch = _make_forward_batch(worker, bs, hidden_dim, vocab_size)

        captured = {}

        def fake_forward(fb, skip_attn_backend_init=False):
            captured["parallel_inputs"] = fb.spec_info.hidden_states.clone()
            logits = torch.randn(fb.input_ids.shape[0], vocab_size, device=DEVICE)
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

        worker.draft_runner = SimpleNamespace(forward=fake_forward)

        h_fused = forward_batch.spec_info.hidden_states.clone()
        last_tokens = forward_batch.spec_info.bonus_tokens.clone().long()
        embed = worker._embed_table

        worker.draft_forward(forward_batch)

        parallel_inputs = captured["parallel_inputs"].view(bs, K, hidden_dim)
        h_shared_expected = h_fused.mean(dim=0)

        for seq in range(bs):
            expected_pos0 = h_fused[seq] + embed[last_tokens[seq]]
            torch.testing.assert_close(
                parallel_inputs[seq, 0], expected_pos0, atol=1e-4, rtol=1e-4
            )
            expected_shared = h_shared_expected + embed[worker._mask_token_id]
            for pos in range(1, K):
                torch.testing.assert_close(
                    parallel_inputs[seq, pos], expected_shared, atol=1e-4, rtol=1e-4
                )

    def test_dsl_masks_low_confidence_positions_to_sentinel_token(self):
        """With DSL enabled and logits rigged so every position after the
        first is a uniform (zero-margin, hence zero-confidence) distribution,
        positions 1..K-1 must be masked to the sentinel token id 0."""
        bs, K, hidden_dim, vocab_size = 2, 4, 16, 32
        worker = _make_worker(
            num_steps=K,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            enable_dsl=True,
            dsl_threshold=1.0,
        )
        forward_batch = _make_forward_batch(worker, bs, hidden_dim, vocab_size)

        def fake_forward(fb, skip_attn_backend_init=False):
            n = fb.input_ids.shape[0]
            logits = torch.zeros(n, vocab_size, device=DEVICE)
            # Position 0 (every K-th row starting at 0): confident, one-hot.
            logits[0::K, 5] = 100.0
            # Positions 1..K-1: uniform -> confidence == 0 < threshold=1.0.
            return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=logits))

        worker.draft_runner = SimpleNamespace(forward=fake_forward)
        _parent, _idx, draft_tokens, _probs = worker.draft_forward(forward_batch)

        self.assertTrue(torch.all(draft_tokens[:, 0] == 5))
        self.assertTrue(torch.all(draft_tokens[:, 1:] == 0))

    def test_rejects_topk_greater_than_1(self):
        # Stand in for EagleDraftWorker.__init__ (which loads a real model)
        # so only PEAGLEDraftWorker's own validation runs.
        def fake_super_init(self, *args, **kwargs):
            self.topk = 2
            self.server_args = _fake_server_args(
                speculative_use_rejection_sampling=False
            )

        with unittest.mock.patch(
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
            fake_super_init,
        ):
            with self.assertRaisesRegex(ValueError, "topk"):
                PEAGLEDraftWorker(enable_dsl=False)

    def test_rejects_rejection_sampling(self):
        def fake_super_init(self, *args, **kwargs):
            self.topk = 1
            self.server_args = _fake_server_args(
                speculative_use_rejection_sampling=True
            )

        with unittest.mock.patch(
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
            fake_super_init,
        ):
            with self.assertRaisesRegex(ValueError, "rejection"):
                PEAGLEDraftWorker(enable_dsl=False)

    def test_rejects_cuda_graph_enabled(self):
        # draft_forward's batch*K expansion only runs once, at capture time,
        # inside EagleDraftCudaGraphRunner.capture_one_shape -- real requests
        # then replay through execute(), which never re-runs the expansion
        # and only refreshes the unexpanded buffers.hidden_states[:raw_bs].
        # CUDA graphs must be disabled until this class gets its own
        # capture_one_shape/execute overrides.
        def fake_super_init(self, *args, **kwargs):
            self.topk = 1
            self.server_args = _fake_server_args(
                speculative_use_rejection_sampling=False,
                disable_cuda_graph=False,
            )

        with unittest.mock.patch(
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
            fake_super_init,
        ):
            with self.assertRaisesRegex(ValueError, "CUDA graph"):
                PEAGLEDraftWorker(enable_dsl=False)


if __name__ == "__main__":
    unittest.main()
