"""Unit tests for fp4_utils.py changes introduced in this PR.

PR changes:
1. initialize_fp4_gemm_config auto-backend now prioritises SM120 (Blackwell)
   -> flashinfer_cudnn *before* checking SM100 -> flashinfer_cutedsl.
2. Fp4GemmRunnerBackend enum exposes typed predicate methods.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(backend: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(fp4_gemm_runner_backend=backend)


def _initialize(backend: str = "auto", *, sm120=False, sm100=False, is_cuda=True, cap=(9, 0)):
    """
    Call initialize_fp4_gemm_config with controlled hardware stubs and return
    the resulting FP4_GEMM_RUNNER_BACKEND value.
    """
    import sglang.srt.layers.quantization.fp4_utils as fp4_mod

    with (
        patch(
            "sglang.srt.layers.quantization.fp4_utils.is_sm120_supported",
            return_value=sm120,
        ),
        patch(
            "sglang.srt.layers.quantization.fp4_utils.is_sm100_supported",
            return_value=sm100,
        ),
        patch(
            "sglang.srt.layers.quantization.fp4_utils.is_cuda",
            return_value=is_cuda,
        ),
        patch(
            "sglang.srt.layers.quantization.fp4_utils.get_device_capability",
            return_value=cap,
        ),
    ):
        # Reset global so each test starts clean
        fp4_mod.FP4_GEMM_RUNNER_BACKEND = None
        fp4_mod.initialize_fp4_gemm_config(_server_args(backend))
        return fp4_mod.FP4_GEMM_RUNNER_BACKEND


class TestFp4GemmAutoBackendSelection(unittest.TestCase):
    """Tests for the auto-selection priority: SM120 > SM100 > SM80-90 > else."""

    def test_auto_sm120_selects_flashinfer_cudnn(self):
        """SM120 (Blackwell) should pick flashinfer_cudnn (new behaviour)."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(sm120=True, sm100=False)
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUDNN)

    def test_auto_sm100_only_selects_flashinfer_cutedsl(self):
        """SM100 without SM120 should pick flashinfer_cutedsl."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(sm120=False, sm100=True)
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL)

    def test_auto_sm120_takes_priority_over_sm100(self):
        """If both sm120 and sm100 are reported (unlikely but guards ordering),
        sm120 wins because the check comes first in the code."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(sm120=True, sm100=True)
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUDNN)

    def test_auto_sm80_selects_marlin(self):
        """SM80-SM90 range on CUDA should pick marlin."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        # Cap (8, 0) is within [8.0, 10.0)
        result = _initialize(sm120=False, sm100=False, is_cuda=True, cap=(8, 0))
        self.assertEqual(result, Fp4GemmRunnerBackend.MARLIN)

    def test_auto_sm90_selects_marlin(self):
        """SM90 is within the marlin range."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(sm120=False, sm100=False, is_cuda=True, cap=(9, 0))
        self.assertEqual(result, Fp4GemmRunnerBackend.MARLIN)

    def test_auto_sm100_capability_but_sm100_false_falls_to_marlin(self):
        """Cap (10, 0) matches < (10, 0) == False, so marlin is not chosen; falls to cutlass."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        # cap (10, 0): NOT < (10, 0), not >= (8, 0)?
        # condition: is_cuda and (10, 0) > cap >= (8, 0) -> (10,0) > (10,0) is False
        result = _initialize(sm120=False, sm100=False, is_cuda=True, cap=(10, 0))
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUTLASS)

    def test_auto_non_cuda_falls_to_flashinfer_cutlass(self):
        """Non-CUDA (e.g. ROCm) falls through to flashinfer_cutlass."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(sm120=False, sm100=False, is_cuda=False, cap=(9, 0))
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUTLASS)

    def test_explicit_backend_bypasses_auto(self):
        """An explicit backend string skips all auto-detection."""
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(
            backend="marlin", sm120=True, sm100=True
        )
        self.assertEqual(result, Fp4GemmRunnerBackend.MARLIN)

    def test_explicit_cutlass_backend(self):
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(backend="cutlass")
        self.assertEqual(result, Fp4GemmRunnerBackend.CUTLASS)

    def test_explicit_flashinfer_cudnn_backend(self):
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        result = _initialize(backend="flashinfer_cudnn")
        self.assertEqual(result, Fp4GemmRunnerBackend.FLASHINFER_CUDNN)


class TestFp4GemmRunnerBackendEnum(unittest.TestCase):
    """Tests for Fp4GemmRunnerBackend enum predicate methods."""

    def setUp(self):
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        self.B = Fp4GemmRunnerBackend

    def test_is_auto(self):
        self.assertTrue(self.B.AUTO.is_auto())
        self.assertFalse(self.B.MARLIN.is_auto())

    def test_is_cutlass(self):
        self.assertTrue(self.B.CUTLASS.is_cutlass())
        self.assertFalse(self.B.AUTO.is_cutlass())

    def test_is_flashinfer_cudnn(self):
        self.assertTrue(self.B.FLASHINFER_CUDNN.is_flashinfer_cudnn())
        self.assertFalse(self.B.FLASHINFER_CUTLASS.is_flashinfer_cudnn())

    def test_is_flashinfer_cutlass(self):
        self.assertTrue(self.B.FLASHINFER_CUTLASS.is_flashinfer_cutlass())
        self.assertFalse(self.B.FLASHINFER_CUDNN.is_flashinfer_cutlass())

    def test_is_flashinfer_trtllm(self):
        self.assertTrue(self.B.FLASHINFER_TRTLLM.is_flashinfer_trtllm())
        self.assertFalse(self.B.CUTLASS.is_flashinfer_trtllm())

    def test_is_flashinfer_cutedsl(self):
        self.assertTrue(self.B.FLASHINFER_CUTEDSL.is_flashinfer_cutedsl())
        self.assertFalse(self.B.FLASHINFER_CUDNN.is_flashinfer_cutedsl())

    def test_is_marlin(self):
        self.assertTrue(self.B.MARLIN.is_marlin())
        self.assertFalse(self.B.CUTLASS.is_marlin())

    def test_is_flashinfer_generic(self):
        """All flashinfer_* variants should return True for is_flashinfer()."""
        flashinfer_variants = [
            self.B.FLASHINFER_CUDNN,
            self.B.FLASHINFER_CUTEDSL,
            self.B.FLASHINFER_CUTLASS,
            self.B.FLASHINFER_TRTLLM,
        ]
        for variant in flashinfer_variants:
            with self.subTest(variant=variant):
                self.assertTrue(variant.is_flashinfer())

        non_flashinfer = [self.B.AUTO, self.B.CUTLASS, self.B.MARLIN]
        for variant in non_flashinfer:
            with self.subTest(variant=variant):
                self.assertFalse(variant.is_flashinfer())

    def test_get_flashinfer_backend_cutedsl(self):
        """FLASHINFER_CUTEDSL maps to 'cute-dsl' (special remapping)."""
        self.assertEqual(self.B.FLASHINFER_CUTEDSL.get_flashinfer_backend(), "cute-dsl")

    def test_get_flashinfer_backend_others(self):
        """Other flashinfer variants strip the 'flashinfer_' prefix."""
        self.assertEqual(self.B.FLASHINFER_CUDNN.get_flashinfer_backend(), "cudnn")
        self.assertEqual(self.B.FLASHINFER_CUTLASS.get_flashinfer_backend(), "cutlass")
        self.assertEqual(self.B.FLASHINFER_TRTLLM.get_flashinfer_backend(), "trtllm")

    def test_get_fp4_gemm_runner_backend_defaults_to_auto(self):
        """get_fp4_gemm_runner_backend returns AUTO when not initialised."""
        import sglang.srt.layers.quantization.fp4_utils as fp4_mod
        from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend

        fp4_mod.FP4_GEMM_RUNNER_BACKEND = None
        result = fp4_mod.get_fp4_gemm_runner_backend()
        self.assertEqual(result, Fp4GemmRunnerBackend.AUTO)


if __name__ == "__main__":
    unittest.main()