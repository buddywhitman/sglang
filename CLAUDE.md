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

**Unresolved.** Reproduces at both K=4 and K=2, in unmodified inherited code, inside
`verify()` at `eagle_utils.py:531`. `CUDA_LAUNCH_BLOCKING=1` consistently points to
FlashInfer's `BatchDecodeWithPagedKVCache` paged-KV-cache decode kernel as the failing
launch.

Ruled out so far:
- `out_cache_loc` layout mismatch between `draft_forward`'s `repeat_interleave`
  ordering and `per_step_draft_out_cache_loc`/`assign_draft_cache_locs_contiguous` --
  read through carefully, layouts match.
- Attention-backend internal state not restored after the K-expanded call -- added an
  explicit re-plan back to the original shape in `finally` (see the latest commit on
  `p_eagle_worker.py`), tested at K=2, **same crash still occurs**. The re-plan is a
  correct state-hygiene fix and stays in the code, but it isn't the root cause.

**Diagnostic tooling gap on WSL2**: `compute-sanitizer --tool memcheck
--target-processes all` fails outright on this WSL2 + consumer-GPU setup --
`Failed to initialize WDDM debugger interface` / `Device not supported`. This is a hard
environment limitation (WDDM debugger interface isn't exposed through WSL2's GPU
passthrough), not something fixable by retrying or reconfiguring. This is the reason
for moving this specific investigation to bare-metal Linux.

### Repro command (run once compute-sanitizer is available -- i.e. on bare metal)

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

1. `git clone https://github.com/buddywhitman/sglang.git ~/contribution/sglang && cd
   ~/contribution/sglang && git remote add upstream https://github.com/sgl-project/sglang.git`
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
