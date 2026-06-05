"""Unit tests verifying that environment variables removed in this PR no longer
exist on the Envs class and that remaining related vars are still present.

PR change: Removed from Envs:
  - FLASHINFER_NVFP4_4OVER6
  - FLASHINFER_NVFP4_4OVER6_E4M3_USE_256
  - SGLANG_DEEPGEMM_PDL

Also verifies that SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION (kept) and
SGLANG_DEEPGEMM_SANITY_CHECK (kept) are still accessible.
"""
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REMOVED_ATTRS = [
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    "SGLANG_DEEPGEMM_PDL",
]

KEPT_ATTRS = [
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION",
    "SGLANG_DEEPGEMM_SANITY_CHECK",
    "SGLANG_ENABLE_JIT_DEEPGEMM",
]


class TestEnvsRemovedVariables(unittest.TestCase):
    """Verify that removed env-var descriptors no longer exist on Envs."""

    def test_removed_vars_absent_from_envs_class(self):
        """Removed variables must not be attributes of the Envs class."""
        from sglang.srt.environ import Envs

        for attr in REMOVED_ATTRS:
            with self.subTest(attr=attr):
                self.assertFalse(
                    hasattr(Envs, attr),
                    f"Envs.{attr} should have been removed by this PR but is still present",
                )

    def test_kept_vars_still_accessible(self):
        """Variables intentionally kept must still be present on the Envs class."""
        from sglang.srt.environ import Envs

        for attr in KEPT_ATTRS:
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(Envs, attr),
                    f"Envs.{attr} was unexpectedly removed",
                )

    def test_removed_vars_not_readable_via_envs_singleton(self):
        """Accessing removed vars via the module-level `envs` singleton should raise AttributeError."""
        from sglang.srt import environ

        for attr in REMOVED_ATTRS:
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError):
                    getattr(environ.envs, attr)

    def test_per_token_activation_default_is_false(self):
        """SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION should default to False."""
        from sglang.srt.environ import Envs

        descriptor = Envs.__dict__.get("SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION")
        self.assertIsNotNone(descriptor)
        # The EnvBool default is stored as the `default` attribute
        self.assertFalse(descriptor.default)

    def test_deepgemm_sanity_check_default_is_false(self):
        """SGLANG_DEEPGEMM_SANITY_CHECK should default to False."""
        from sglang.srt.environ import Envs

        descriptor = Envs.__dict__.get("SGLANG_DEEPGEMM_SANITY_CHECK")
        self.assertIsNotNone(descriptor)
        self.assertFalse(descriptor.default)


if __name__ == "__main__":
    unittest.main()
