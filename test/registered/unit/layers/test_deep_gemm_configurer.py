"""Unit tests for deep_gemm_wrapper/configurer.py changes.

PR change: DEEPGEMM_BLACKWELL now uses is_blackwell_supported() (covers
SM100/SM110/SM120) instead of is_sm100_supported() (SM100-only).  This means
SM120 (RTX Blackwell) is now included in the DEEPGEMM_BLACKWELL flag.

Also tests that deep_gemm_wrapper/entrypoint.py no longer calls
deep_gemm.set_pdl() (SGLANG_DEEPGEMM_PDL env var was removed).
"""
import sys
import types
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepGemmConfigurerFunctionUsed(unittest.TestCase):
    """Verify that configurer.py imports and uses is_blackwell_supported."""

    def test_configurer_imports_is_blackwell_supported_not_sm100(self):
        """The configurer module must import is_blackwell_supported, not is_sm100_supported."""
        import importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_configurer_src",
            "/home/jailuser/git/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py",
        )
        # Read the source to check the import
        with open(spec.origin) as f:
            source = f.read()

        self.assertIn(
            "is_blackwell_supported",
            source,
            "configurer.py should import is_blackwell_supported",
        )
        # is_sm100_supported should no longer be imported in this file
        self.assertNotIn(
            "is_sm100_supported",
            source,
            "configurer.py should not use is_sm100_supported after the PR",
        )

    def test_deepgemm_blackwell_uses_is_blackwell_supported(self):
        """DEEPGEMM_BLACKWELL expression must reference is_blackwell_supported."""
        with open(
            "/home/jailuser/git/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py"
        ) as f:
            source = f.read()

        # The constant assignment should use is_blackwell_supported
        self.assertIn("DEEPGEMM_BLACKWELL", source)
        # Find the line
        for line in source.splitlines():
            if "DEEPGEMM_BLACKWELL" in line and "=" in line and "and" in line:
                self.assertIn(
                    "is_blackwell_supported",
                    line,
                    f"DEEPGEMM_BLACKWELL assignment should use is_blackwell_supported: {line!r}",
                )
                break


class TestDeepGemmConfigurerConstants(unittest.TestCase):
    """Verify module-level constants reflect the new is_blackwell_supported logic."""

    def _reimport_configurer(self, *, enable_jit=True, is_blackwell=True):
        """Reimport the configurer module with patched dependencies."""
        # Remove cached module if present
        for key in list(sys.modules.keys()):
            if "deep_gemm_wrapper.configurer" in key:
                del sys.modules[key]

        with (
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer.is_blackwell_supported",
                return_value=is_blackwell,
            ),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configurer._compute_enable_deep_gemm",
                return_value=enable_jit,
            ),
        ):
            import importlib

            mod = importlib.import_module(
                "sglang.srt.layers.deep_gemm_wrapper.configurer"
            )
            # Recompute with patched values
            deepgemm_blackwell = enable_jit and is_blackwell
            return deepgemm_blackwell

    def test_deepgemm_blackwell_true_when_jit_and_blackwell(self):
        result = self._reimport_configurer(enable_jit=True, is_blackwell=True)
        self.assertTrue(result)

    def test_deepgemm_blackwell_false_when_no_jit(self):
        result = self._reimport_configurer(enable_jit=False, is_blackwell=True)
        self.assertFalse(result)

    def test_deepgemm_blackwell_false_when_not_blackwell(self):
        result = self._reimport_configurer(enable_jit=True, is_blackwell=False)
        self.assertFalse(result)


class TestEntrypointNoPdlSetup(unittest.TestCase):
    """Verify that entrypoint.py no longer calls deep_gemm.set_pdl()."""

    def test_entrypoint_does_not_call_set_pdl(self):
        """The entrypoint source must not contain set_pdl calls (removed with SGLANG_DEEPGEMM_PDL)."""
        with open(
            "/home/jailuser/git/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py"
        ) as f:
            source = f.read()

        self.assertNotIn(
            "set_pdl",
            source,
            "set_pdl should have been removed from entrypoint.py (SGLANG_DEEPGEMM_PDL was deleted)",
        )
        self.assertNotIn(
            "SGLANG_DEEPGEMM_PDL",
            source,
            "SGLANG_DEEPGEMM_PDL env var reference should be absent from entrypoint.py",
        )


if __name__ == "__main__":
    unittest.main()