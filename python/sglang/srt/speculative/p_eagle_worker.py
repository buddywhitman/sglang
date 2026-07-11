"""
P-EAGLE + confidence-gated draft masking for SGLang (V2 speculative worker).

Ports the parallel-draft idea from the deprecated V1 EAGLEWorker path onto
sglang's V2 speculative-decoding architecture (EAGLEWorkerV2 / EagleDraftWorker,
see spec_registry.py / spec_info.py:create_worker). V1 was removed in
sgl-project/sglang#25464 (merged 2026-06-08, one day after this file's original
PR was opened); this file targets V2 exclusively.

Only the draft loop differs from vanilla EAGLE/EAGLE3 -- draft_extend, verify,
CUDA graph capture, and KV cache/tree bookkeeping are all inherited unchanged
from EagleDraftWorker / EAGLEWorkerV2.

1. P-EAGLE (Parallel EAGLE):
   Generates all K draft tokens in ONE forward pass instead of K sequential
   passes. Position 0 uses h_fused + embed(last_accept_token); positions
   1..K-1 share a single mean(h_fused) + embed(MASK) placeholder, removing
   the sequential dependency between draft positions. Requires
   --speculative-eagle-topk 1 (chain drafting only -- branching trees need
   each branch its own hidden state, which the shared-placeholder trick
   does not produce).

2. Confidence-gated draft masking (formerly called "Sync-Free DSL"):
   A fused Triton kernel scores each of the K parallel positions
   (max-logit-margin confidence) in one pass. Positions below
   peagle_dsl_threshold, and everything after the first such position in a
   sequence's chain, are replaced with a sentinel token (id 0) instead of a
   real sampled token, so verification rejects them immediately rather than
   spending target-model compute on draft tokens that were very unlikely to
   be accepted anyway. This is a verify-side token-budget optimization, not
   a cross-round compute skip: P-EAGLE's whole draft round is already a
   single batched forward pass, so there is no later "step" within a round
   left to skip. An earlier version of this file described a persistent
   cross-round GPU buffer ("_dsl_continue_buf") that a later draft_forward
   call would read to skip compute for already-exited sequences; that
   mechanism was written but never actually read anywhere and has been
   removed rather than kept as decorative dead state.

References:
  - P-EAGLE: vLLM v0.16.0 vllm/spec_decode/proposers/p_eagle_proposer.py
  - Open issue: github.com/sgl-project/sglang/issues/23171
  - V1 deprecation this file ports across: sgl-project/sglang#25464

Usage:
  --speculative-algorithm PEAGLE     --speculative-eagle-topk 1
  --speculative-algorithm PEAGLE_DSL --speculative-eagle-topk 1  (+ masking)

CAUTION: written and syntax-checked without GPU hardware access -- treat as
an unverified draft. The highest-risk, least-verified part is the in-place
K-expansion of `forward_batch` in PEAGLEDraftWorker.draft_forward
(repeat_interleave over input_ids/req_pool_indices/seq_lens/positions, reuse
of the pre-allocated topk=1 out_cache_loc slots, and reuse of
draft_attn_backend.attn_backends[0]'s planned metadata for the expanded
batch). Validate on real hardware starting from the simplest possible
config (--disable-cuda-graph --page-size 1 --speculative-eagle-topk 1)
before trusting the CUDA graph capture path, DP attention, or paged KV
cache with topk>1-style branch layouts (which P-EAGLE never uses, but
neighboring code paths might assume are always possible).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    EagleDraftWorker,
    _get_plan_stream,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.triton_ops.fused_draft_input import (
    fused_parallel_draft_input,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence-gated draft sampling: Triton kernel -- samples K draft tokens
# per sequence AND scores each position's confidence, in one fused pass.
# ---------------------------------------------------------------------------
@triton.jit
def _draft_sample_with_confidence_kernel(
    logits_ptr,  # [batch, K, vocab_size] float32
    output_tokens_ptr,  # [batch, K] int32 output
    output_scores_ptr,  # [batch, K] float32 output -- log-prob, or -1e9 if masked
    confidence_threshold: tl.constexpr,
    vocab_size: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Per sequence: greedy-sample K draft tokens. Once a position's confidence
    (top1-logit margin over top2) drops below confidence_threshold, that
    position and every position after it in the chain get the sentinel
    token id 0 / score -1e9, so verification rejects them immediately
    instead of spending compute confirming a low-confidence guess.
    """
    # Triton's AST-to-IR pass does not support `continue`/`break` inside a
    # `for` loop -- structure this as "do the real work only if not already
    # masked" instead of early-continuing out of masked iterations.
    seq_id = tl.program_id(0)
    masked = False

    for k in range(K):
        if not masked:
            base_ptr = logits_ptr + seq_id * K * vocab_size + k * vocab_size

            # Pass 1: global max1 and its position (argmax)
            max1 = -1e9
            argmax = 0
            for v_start in range(0, vocab_size, BLOCK_V):
                v_offs = v_start + tl.arange(0, BLOCK_V)
                v_mask = v_offs < vocab_size
                logits_block = tl.load(base_ptr + v_offs, mask=v_mask, other=-1e9)
                block_max = tl.max(logits_block, axis=0)
                block_argmax = tl.argmax(logits_block, axis=0) + v_start
                if block_max > max1:
                    max1 = block_max
                    argmax = block_argmax

            # Pass 2: global max2, masking out the argmax position
            # specifically (not just its block) so co-located top-1/top-2
            # logits don't collapse to the same value.
            max2 = -1e9
            for v_start in range(0, vocab_size, BLOCK_V):
                v_offs = v_start + tl.arange(0, BLOCK_V)
                v_mask = (v_offs < vocab_size) & (v_offs != argmax)
                logits_block = tl.load(base_ptr + v_offs, mask=v_mask, other=-1e9)
                block_max = tl.max(logits_block, axis=0)
                if block_max > max2:
                    max2 = block_max

            confidence = max1 - max2
            if confidence < confidence_threshold:
                masked = True
            else:
                token = argmax

                # log_prob of the sampled token via numerically-stable
                # log-sum-exp. token == argmax and max1 == logits[argmax]
                # (pinned together in pass 1), so log_prob = max1 -
                # log_sum_exp without re-loading token_logit from memory.
                sum_exp = 0.0
                for v_start in range(0, vocab_size, BLOCK_V):
                    v_offs = v_start + tl.arange(0, BLOCK_V)
                    v_mask = v_offs < vocab_size
                    logits_block = tl.load(
                        base_ptr + v_offs, mask=v_mask, other=-1e9
                    )
                    sum_exp += tl.sum(tl.exp(logits_block - max1), axis=0)
                log_sum_exp = tl.log(sum_exp) + max1
                log_prob = max1 - log_sum_exp

                tl.store(output_tokens_ptr + seq_id * K + k, token)
                tl.store(output_scores_ptr + seq_id * K + k, log_prob)

        if masked:
            tl.store(output_tokens_ptr + seq_id * K + k, 0)
            tl.store(output_scores_ptr + seq_id * K + k, -1e9)


class PEAGLEDraftWorker(EagleDraftWorker):
    """
    EagleDraftWorker specialization that replaces the K-sequential
    draft_forward loop with P-EAGLE's single batched forward pass.
    Everything else (draft_extend, CUDA graph capture, alloc_memory_pool,
    attention backend init) is inherited unchanged -- CUDA graph capture in
    particular calls self.draft_forward(...) polymorphically (see
    eagle_draft_cuda_graph_runner.py), so it should pick up this override
    automatically. Unverified on real hardware; see module docstring.
    """

    MASK_TOKEN_FALLBACK_ID: int = 2  # <s> token as MASK fallback if model has no MASK

    def __init__(
        self, *args, enable_dsl: bool = False, dsl_threshold: float = 2.0, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if self.topk != 1:
            raise ValueError(
                "P-EAGLE only supports chain drafting: pass "
                "--speculative-eagle-topk 1. Branching trees (topk > 1) need "
                "each branch's own hidden state; P-EAGLE's positions 1..K-1 "
                "share one placeholder hidden state by design."
            )
        if self.server_args.speculative_use_rejection_sampling:
            raise ValueError(
                "P-EAGLE does not implement rejection-sampling draft "
                "probabilities (it always greedy-samples). Do not pass "
                "--speculative-use-rejection-sampling with PEAGLE/PEAGLE_DSL."
            )
        # draft_forward dynamically expands forward_batch to batch*K via fresh
        # tensor allocations (repeat_interleave, fused_parallel_draft_input).
        # Under CUDA graph capture (EagleDraftCudaGraphRunner), draft_forward
        # only ever runs once, inside capture_one_shape's run_once() -- the
        # K-expansion is baked into the captured kernels' one-time allocation
        # addresses. Every real request then goes through execute(), which
        # refreshes only the unexpanded buffers.hidden_states[:raw_bs] (and
        # friends) with real per-request data -- the captured graph's kernels
        # never read from those buffers for the K-expanded computation, so
        # replay silently uses stale capture-time data instead of the real
        # request. This is a structural memory-safety bug (the most likely
        # cause of the FlashInfer BatchDecodeWithPagedKVCache
        # illegal-memory-access crash in verify()), not merely an unverified
        # path -- fail loudly until this class gets its own
        # capture_one_shape/execute overrides that redo the K-expansion at
        # replay time.
        if not self.server_args.disable_cuda_graph:
            raise ValueError(
                "PEAGLE/PEAGLE_DSL do not yet support CUDA graph capture "
                "(pass --disable-cuda-graph). See the comment above this check "
                "for why: draft_forward's batch*K expansion only runs once, at "
                "capture time, and is never refreshed with real data on replay."
            )

        self.enable_dsl = enable_dsl
        self.dsl_threshold = dsl_threshold

        hidden_dim = self._get_hidden_dim()
        model_dtype = self.draft_runner.model_config.dtype
        self._h_shared = torch.zeros(
            hidden_dim, dtype=model_dtype, device=self.device
        )
        self._mask_token_id = self._resolve_mask_token_id()

        logger.info(
            f"PEAGLEDraftWorker initialized: K={self.speculative_num_steps} "
            f"DSL={'on' if enable_dsl else 'off'} mask_token_id={self._mask_token_id}"
        )

    def _get_hidden_dim(self) -> int:
        return self.draft_runner.model_config.hf_config.hidden_size

    def _resolve_mask_token_id(self) -> int:
        """Get MASK token ID from the draft model's config, if it has one."""
        try:
            cfg = self.draft_runner.model_config.hf_config
            # EAGLE-3 stores mask_token_id in eagle_config
            eagle_cfg = getattr(cfg, "eagle_config", {})
            if isinstance(eagle_cfg, dict) and "mask_token_id" in eagle_cfg:
                return eagle_cfg["mask_token_id"]
        except Exception:
            pass
        return self.MASK_TOKEN_FALLBACK_ID

    @property
    def _embed_table(self) -> torch.Tensor:
        """Token embedding table shared with the target model."""
        embed, _ = self.target_worker.model_runner.model.get_embed_and_head()
        return embed.weight if hasattr(embed, "weight") else embed

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info: EagleDraftInput = forward_batch.spec_info
        hidden_states = spec_info.hidden_states

        if (
            forward_batch.forward_mode.is_idle()
            or hidden_states is None
            or len(hidden_states) == 0
            or self.speculative_num_steps <= 1
        ):
            # Degenerate/edge cases: defer to the proven sequential path
            # rather than special-casing them here.
            return super().draft_forward(forward_batch)

        batch_size = hidden_states.shape[0]
        K = self.speculative_num_steps

        h_fused = hidden_states  # [batch, hidden_dim] -- EAGLE-3 tri-layer fusion, done upstream
        self._h_shared.copy_(h_fused.mean(dim=0))
        last_tokens = spec_info.bonus_tokens.to(torch.int64)

        parallel_inputs = fused_parallel_draft_input(
            h_fused=h_fused,
            embed_table=self._embed_table,
            last_tokens=last_tokens,
            h_shared=self._h_shared,
            mask_token_id=self._mask_token_id,
            K=K,
        )  # [batch*K, hidden_dim]

        # Expand forward_batch to batch*K in place, mirroring how the base
        # sequential loop mutates the same object per-step, rather than
        # hand-constructing a new ForwardBatch -- that would need to
        # re-derive V2's DP/MoE/paging setup that prepare_for_draft already
        # did correctly for us at the original batch size.
        orig_input_ids = forward_batch.input_ids
        orig_req_pool_indices = forward_batch.req_pool_indices
        orig_seq_lens = forward_batch.seq_lens
        orig_positions = forward_batch.positions
        orig_hidden_states = spec_info.hidden_states
        orig_batch_size = forward_batch.batch_size

        # input_ids is not read anywhere in the base sequential loop before
        # it overwrites it fresh each step (eagle_worker_v2.py:671) -- but it
        # DOES always write a real (non-None) tensor there before the actual
        # forward call. eager_runner.py's load_batch derives raw_num_tokens
        # from input_ids.shape[0], falling back to 0 if input_ids is None;
        # several registry buffer slots are sized off that token-count axis,
        # so leaving input_ids as None here (rather than just not reading
        # its old value) breaks unrelated slot sizing downstream. Build a
        # real, correctly-shaped placeholder the same way the base loop
        # effectively does -- content is irrelevant since parallel_inputs
        # (assigned to hidden_states below) drives the actual computation,
        # not input_ids.
        forward_batch.input_ids = (
            orig_input_ids.repeat_interleave(K, dim=0)
            if orig_input_ids is not None
            else last_tokens.repeat_interleave(K, dim=0)
        )
        forward_batch.req_pool_indices = orig_req_pool_indices.repeat_interleave(
            K, dim=0
        )
        forward_batch.seq_lens = orig_seq_lens.repeat_interleave(K, dim=0)
        forward_batch.positions = orig_positions.repeat_interleave(K, dim=0)
        # out_cache_loc was already pre-allocated at [bs * topk * num_steps]
        # by prepare_for_draft (topk=1 here, so that's exactly [bs * K]) --
        # reused as-is, not touched.
        spec_info.hidden_states = parallel_inputs
        # batch_size is a separate field from the tensor shapes -- the eager
        # runner's buffer registry (cuda_graph_buffer_registry.py) reads it
        # directly to size/locate its copy buffers, independent of
        # input_ids.shape[0]. Leaving it stale caused a real crash here
        # ("tensor a (0) must match tensor b (4)") the first time this ran
        # against a live server.
        forward_batch.batch_size = batch_size * K

        # The caller (draft()) already pre-planned + marked attention
        # metadata ready for the *original* (unexpanded) batch shape.
        # skip_attn_backend_init=True would trust that stale plan for our
        # batch*K shape instead of re-planning -- caught live with
        # flashinfer rejecting the mismatched Q tensor ("q.shape[0] (4)
        # does not match batch_size * q_len_per_req (1 * 1 = 1)"). The base
        # sequential loop never hits this because its per-step batch_size
        # never changes, only the content does. Force a fresh plan for the
        # expanded shape instead of reusing the stale one.
        orig_metadata_ready = forward_batch.forward_metadata_ready
        orig_metadata_planned_bs = forward_batch.forward_metadata_planned_bs
        orig_metadata_planned_num_tokens = (
            forward_batch.forward_metadata_planned_num_tokens
        )
        orig_metadata_replan_equivalent = (
            forward_batch.forward_metadata_replan_equivalent
        )
        forward_batch.forward_metadata_ready = False
        self.draft_attn_backend.init_forward_metadata(forward_batch)
        forward_batch.mark_forward_metadata_ready()

        try:
            with forward_context(
                ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[0])
            ):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
        finally:
            forward_batch.input_ids = orig_input_ids
            forward_batch.req_pool_indices = orig_req_pool_indices
            forward_batch.seq_lens = orig_seq_lens
            forward_batch.positions = orig_positions
            spec_info.hidden_states = orig_hidden_states
            forward_batch.batch_size = orig_batch_size
            forward_batch.forward_metadata_ready = orig_metadata_ready
            forward_batch.forward_metadata_planned_bs = orig_metadata_planned_bs
            forward_batch.forward_metadata_planned_num_tokens = (
                orig_metadata_planned_num_tokens
            )
            forward_batch.forward_metadata_replan_equivalent = (
                orig_metadata_replan_equivalent
            )
            # init_forward_metadata may configure attention-backend-internal
            # state (workspace buffers, plan objects) beyond what
            # forward_batch's own bookkeeping fields track. Re-plan back to
            # the original (restored) shape so no state sized for the
            # K-expanded batch lingers for whatever runs next (verify()).
            if orig_metadata_ready:
                self.draft_attn_backend.init_forward_metadata(forward_batch)

        all_logits = logits_output.next_token_logits.view(batch_size, K, -1)

        if self.enable_dsl:
            draft_tokens, _draft_scores = self._sample_with_confidence(
                all_logits, batch_size, K
            )
        else:
            draft_tokens = torch.argmax(all_logits, dim=-1).to(torch.int32)

        if self.hot_token_id is not None:
            draft_tokens = self.hot_token_id[draft_tokens.long()].to(torch.int32)

        bs = batch_size
        assert bs <= self._topk1_parents_prealloc.shape[0], (
            f"P-EAGLE batch size {bs} exceeds preallocated topk=1 chain "
            f"buffers ({self._topk1_parents_prealloc.shape[0]}); this should "
            "be bounded by the same capture_bs bucketing as vanilla EAGLE."
        )
        draft_tokens_out = draft_tokens.to(torch.int64)
        top_scores_index = self._topk1_score_indices_prealloc[:bs]
        parent_list = self._topk1_parents_prealloc[:bs]
        return parent_list, top_scores_index, draft_tokens_out, None

    def _sample_with_confidence(
        self, all_logits: torch.Tensor, batch_size: int, K: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab_size = all_logits.shape[-1]
        output_tokens = torch.empty(
            batch_size, K, dtype=torch.int32, device=self.device
        )
        output_scores = torch.empty(
            batch_size, K, dtype=torch.float32, device=self.device
        )
        BLOCK_V = min(triton.next_power_of_2(vocab_size), 4096)

        _draft_sample_with_confidence_kernel[(batch_size,)](
            all_logits.contiguous(),
            output_tokens,
            output_scores,
            confidence_threshold=self.dsl_threshold,
            vocab_size=vocab_size,
            K=K,
            BLOCK_V=BLOCK_V,
        )
        return output_tokens, output_scores


class PEAGLEWorkerV2(EAGLEWorkerV2):
    """
    EAGLEWorkerV2 specialization that wires in PEAGLEDraftWorker instead of
    EagleDraftWorker.

    Duplicates EAGLEWorkerV2.__init__ (rather than calling super().__init__()
    and then swapping self._draft_worker) because __init__ has the side
    effect of loading draft model weights via TpModelWorker(...) -- calling
    it twice would double-load. MultiLayerEagleWorkerV2 uses the same
    pattern for the same reason. Every method past __init__
    (forward_batch_generation, verify, alloc_memory_pool, init_cuda_graphs,
    clear_cache_pool, ...) is inherited from EAGLEWorkerV2 unchanged --
    those only ever touch self.draft_worker generically through the
    BaseSpecWorker contract, never EagleDraftWorker directly.
    """

    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as
        # the target model (same as EAGLEWorkerV2.__init__).
        server_args.override(
            "spec_worker.match_target_context_length",
            context_length=target_worker.model_runner.model_config.context_len,
        )

        enable_dsl = self.speculative_algorithm == SpeculativeAlgorithm.PEAGLE_DSL
        self._draft_worker = PEAGLEDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
            enable_dsl=enable_dsl,
            dsl_threshold=server_args.peagle_dsl_threshold,
        )

        # Adaptive speculative decoding (dynamic speculative_num_steps) is a
        # separate existing feature (AdaptiveController / SpecRuntimeState)
        # this port does not integrate with -- fail loudly rather than
        # silently ignoring the flag.
        if server_args.speculative_adaptive:
            raise ValueError(
                "PEAGLE/PEAGLE_DSL do not support --speculative-adaptive."
            )
        self.adaptive_controller = None

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)


PEAGLEWorker = PEAGLEWorkerV2
PEAGLEDSLWorker = PEAGLEWorkerV2
