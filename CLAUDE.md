# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Fork context

This is `buddywhitman/sglang`, a fork of `sgl-project/sglang` (`upstream` remote), used for
upstream contribution work. See `.claude/rules/*.md` for this repo's own code-style rules
(speculative-decoding changes must read `.claude/skills/speculative-naming/SKILL.md` first,
no `dataclasses.dataclass` for new code, no defensive `getattr`/`hasattr`, etc.) -- those
apply to every change in `python/sglang/srt/`, including the work below.

## P-EAGLE V2 port -- current state (2026-07)

`python/sglang/srt/speculative/p_eagle_worker.py` was fully rewritten against the V2
speculative-decoding architecture (`BaseSpecWorker`/`EagleDraftWorkerBase`,
`EagleDraftWorker`/`EAGLEWorkerV2`) after the original PR's target (the deprecated V1
worker) was removed upstream. `PEAGLEDraftWorker(EagleDraftWorker)` overrides only
`draft_forward`; `PEAGLEWorkerV2(EAGLEWorkerV2)` composes it. `PEAGLEWorker =
PEAGLEDSLWorker = PEAGLEWorkerV2` are the public aliases dispatched from
`spec_info.py`'s `create_worker()`.

Real-GPU hardware testing (RTX 3070 Ti, 8GB VRAM, WSL2) found and fixed 6 bugs invisible
to unit tests alone -- see `git log --oneline` on this file for the individual commits
(Triton `continue`-in-loop, `is_eagle3()` dispatch/lm-head misclassification,
`batch_size` not expanded during K-fanout, `input_ids` left `None` instead of a real
placeholder, stale attention-metadata reuse after K-expansion, attn-backend state not
re-planned back to original shape after the call).

### Open bug: CUDA illegal memory access in `verify()`

**Root cause identified (2026-07-11), not yet hardware-confirmed.** Reproduces at both
K=4 and K=2, `CUDA_LAUNCH_BLOCKING=1` consistently points to FlashInfer's
`BatchDecodeWithPagedKVCache` paged-KV-cache decode kernel as the failing launch. The
documented repro command below never passed `--disable-cuda-graph` -- CUDA graphs were
enabled (the default) on every reproduction so far.

**Leading hypothesis: P-EAGLE's CUDA graph integration is structurally broken, not
just unverified.** Traced through `eagle_draft_cuda_graph_runner.py`:

- `capture_one_shape()`'s `run_once()` (line ~445) calls `self.eagle_worker.draft_forward(forward_batch)`
  **exactly once per captured bucket size**, against a fixed capture-time
  `forward_batch`. Inside that one call, `PEAGLEDraftWorker.draft_forward` allocates
  `parallel_inputs` (shape `[batch*K, hidden_dim]`, via `fused_parallel_draft_input`)
  and repeat-interleaves `input_ids`/`req_pool_indices`/`seq_lens`/`positions` to
  `batch*K` -- fresh tensors, new addresses, baked into whatever CUDA kernels
  `self.draft_runner.forward(...)` launches during *this one capture pass*.
- Every real request instead goes through `execute()` (the generic replay driver,
  shared by all EAGLE variants, unaware of P-EAGLE's expansion). It only refreshes
  the **unexpanded** buffers -- `buffers.hidden_states[:raw_bs]` (line ~581),
  `buffers.out_cache_loc[:raw_num_token * speculative_num_steps]` (line ~545), sized
  and shaped for the sequential per-step draft convention -- then replays the
  captured graph.
- The captured graph's kernels never read from `buffers.hidden_states`; they read
  from `parallel_inputs`'s one-time capture-time address, which is **never updated
  with real request data**. Every real replay reads stale/garbage capture-time state
  for the K-expanded computation. This independently matches the module docstring's
  own caution (written before this trace, based on code-reading intuition alone) to
  validate with `--disable-cuda-graph` before trusting the graph-capture path -- a
  step the repro command below skipped.

**Fix applied**: `PEAGLEDraftWorker.__init__` now raises `ValueError` if
`server_args.disable_cuda_graph` is not set, converting this from a silent
illegal-memory-access crash into an immediate, clear startup error. This does **not**
fix CUDA graph support (that needs real `capture_one_shape`/`execute` overrides that
redo the K-expansion at replay time -- nontrivial, not attempted yet) -- it just makes
the currently-broken path fail loudly instead of corrupting memory. Covered by
`test_rejects_cuda_graph_enabled` in
`test/registered/unit/spec/test_p_eagle_v2_worker.py` (21/21 passing).

**Not yet confirmed**: this hypothesis is based on static code tracing only -- no GPU
access to actually run `--disable-cuda-graph` and see whether `verify()` still
crashes. If it does still crash with graphs disabled, there's a second bug in the
eager-mode path that the two previously-ruled-out hypotheses below didn't catch
either. Confirm on bare metal before treating this as closed.

Previously ruled out (still true, re-verified 2026-07-11 by reading
`assign_draft_cache_locs_contiguous` in `triton_ops/cache_locs.py` directly -- it
writes each sequence's `topk*K` slots as one contiguous block, i.e. seq-major,
which is exactly what `repeat_interleave(K, dim=0)` produces -- no mismatch):
- `out_cache_loc` layout mismatch between `draft_forward`'s `repeat_interleave`
  ordering and `assign_draft_cache_locs_contiguous`'s pre-allocation layout.
- Attention-backend internal state not restored after the K-expanded call -- the
  re-plan-back fix in `finally` (see `p_eagle_worker.py`) is correct state hygiene
  and stays in the code, but on its own didn't stop the crash.

**Diagnostic tooling gap on WSL2**: `compute-sanitizer --tool memcheck
--target-processes all` fails outright on this WSL2 + consumer-GPU setup --
`Failed to initialize WDDM debugger interface` / `Device not supported`. This is a hard
environment limitation (WDDM debugger interface isn't exposed through WSL2's GPU
passthrough), not something fixable by retrying or reconfiguring. This is the reason
for moving this specific investigation to bare-metal Linux.

### Repro command (run once compute-sanitizer is available -- i.e. on bare metal)

Run with `--disable-cuda-graph` first -- this tests the eager-mode-only path, which is
what the 6 hardware-found-and-fixed bugs were validated against, and is what the new
CUDA-graph guard in `PEAGLEDraftWorker.__init__` now requires. Re-enabling CUDA
graphs is not expected to work yet -- the guard will refuse to start until
`capture_one_shape`/`execute` get proper overrides.

```bash
conda activate gpu
cd ~/contribution/sglang
CUDA_LAUNCH_BLOCKING=1 compute-sanitizer --tool memcheck --target-processes all \
  python -m sglang.launch_server \
    --model-path ~/models/qwen3-1.7b \
    --speculative-algorithm PEAGLE \
    --speculative-draft-model-path ~/models/eagle3-draft/qwen3-1.7b \
    --speculative-draft-model-quantization unquant \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --dtype float16 --mem-fraction-static 0.75 --port 31900 \
    --disable-cuda-graph \
  2>&1 | tee /tmp/peagle_sanitizer.log
```

Then issue one real generation request against `http://127.0.0.1:31900/generate` (a
health-check / empty-prompt request will not trigger `verify()`'s speculative path).
The sanitizer log will show the exact kernel + memory region on the crashing access,
which WSL2 could only approximate via `CUDA_LAUNCH_BLOCKING=1` stack traces.

## Elementary OS bring-up (continuing hardware debugging off WSL2)

The P-EAGLE `verify()` bug above needs `compute-sanitizer`, which doesn't work under
WSL2. Continue the investigation on a bare-metal elementary OS 7.x ("Horus", Ubuntu
22.04 jammy base) box instead.

**Run `scripts/setup_elementary_os.sh` as root on a fresh install.** It installs, in
order: apt build deps, a real NVIDIA driver + CUDA 13.x toolkit (bare-metal driver
install -- WSL2's GPU passthrough is a completely different mechanism and does not
translate, don't skip this step expecting WSL2-style passthrough), miniforge with
`gpu` and `vllm-test` conda envs (torch 2.11.0+cu130, matching the WSL2 dev box), and a
single-node Slurm+munge+OpenMPI "cluster" (`ClusterName=devcluster`, mirrors the WSL2
box's config exactly, sized to the new machine's actual CPU/RAM instead of the WSL2
box's 6 CPU / 5.9GB).

The script prints its own remaining-manual-steps list at the end. In short, after it
finishes (and after the one required reboot for the driver):

1. `git clone -b peagle-v2-port https://github.com/buddywhitman/sglang.git ~/contribution/sglang && cd
   ~/contribution/sglang && git remote add upstream https://github.com/sgl-project/sglang.git`

   (`main` on this fork is an untouched upstream mirror and `feat/p-eagle-parallel-spec-decode`
   is the actual open PR #27498's head branch, frozen at its June 8 state pre-V2-rewrite --
   `peagle-v2-port`, the branch you're reading this file on, is where all the current work lives.)
2. `conda activate gpu && pip install -e "python[all]"`
3. Download the model pair (small enough for 8GB-class VRAM; swap paths/repo IDs if
   the new GPU has more headroom and you want to go back to the original 8B pair):
   ```bash
   hf download Qwen/Qwen3-1.7B --local-dir ~/models/qwen3-1.7b
   hf download AngelSlim/Qwen3-1.7B_eagle3 --local-dir ~/models/eagle3-draft/qwen3-1.7b
   ```
   (The `curl -4` / forced-IPv4 workaround used on the WSL2 box was for a WSL2-specific
   IPv6 PMTU blackhole -- skip it on bare metal unless downloads actually stall.)
4. Run the repro command above under `compute-sanitizer`, with a real `/generate`
   request, and read the sanitizer's memory-region output to root-cause the crash.

Minimal manual effort after cloning is exactly steps 2-4 above -- everything else is
handled by the script.
