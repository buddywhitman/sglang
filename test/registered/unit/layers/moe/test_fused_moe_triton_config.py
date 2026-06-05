"""Unit tests for fused_moe_triton_config.py changes.

PR change: try_get_optimal_moe_config replaced a warning-and-override of a
mismatched down_config BLOCK_SIZE_M with a hard assertion.  The old code silently
corrected the mismatch; the new code asserts that the configs are already
consistent and raises AssertionError when they are not.
"""
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _make_config(block_size_m: int, **extra) -> dict:
    cfg = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    cfg.update(extra)
    return cfg


# Minimal fake shapes: (E, K, N) for w2, (E, K, N//2) for w1
_W1_SHAPE = (8, 64, 32)
_W2_SHAPE = (8, 32, 64)  # E=8, K=32, N=64


def _call_try_get_optimal(
    up_block_m: int,
    down_block_m: int | None,
    return_down_config: bool = True,
):
    """Helper to invoke try_get_optimal_moe_config with mocked internals."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        try_get_optimal_moe_config,
    )

    up_cfg = _make_config(up_block_m)
    # Build down configs only when a BLOCK_SIZE_M is requested
    if down_block_m is not None:
        down_cfgs = {16: _make_config(down_block_m)}
    else:
        down_cfgs = None

    with (
        patch(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_config",
            return_value=None,
        ),
        patch(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_moe_configs",
            side_effect=[{16: up_cfg}, down_cfgs] if return_down_config else [{16: up_cfg}],
        ),
    ):
        return try_get_optimal_moe_config(
            w1_shape=_W1_SHAPE,
            w2_shape=_W2_SHAPE,
            top_k=2,
            dtype=None,
            M=16,
            return_down_config=return_down_config,
        )


class TestTryGetOptimalMoeConfigAssertion(unittest.TestCase):
    """Tests for the assertion that replaced the warning+override logic."""

    def test_matching_block_size_m_returns_successfully(self):
        """When up and down BLOCK_SIZE_M match, the function returns (config, (down, max_m))."""
        config, (down_config, max_block_m) = _call_try_get_optimal(
            up_block_m=16, down_block_m=16
        )
        self.assertEqual(config["BLOCK_SIZE_M"], 16)
        self.assertIsNotNone(down_config)
        self.assertEqual(down_config["BLOCK_SIZE_M"], 16)
        self.assertEqual(max_block_m, 16)

    def test_mismatched_block_size_m_raises_assertion_error(self):
        """When up and down BLOCK_SIZE_M differ, AssertionError must be raised (no silent override)."""
        with self.assertRaises(AssertionError):
            _call_try_get_optimal(up_block_m=16, down_block_m=32)

    def test_down_config_none_does_not_raise(self):
        """down_config=None is explicitly allowed by the assertion."""
        config, (down_config, max_block_m) = _call_try_get_optimal(
            up_block_m=64, down_block_m=None
        )
        self.assertIsNone(down_config)
        self.assertIsNone(max_block_m)

    def test_return_down_config_false_skips_assertion(self):
        """When return_down_config=False the assertion branch is never entered."""
        result = _call_try_get_optimal(
            up_block_m=16, down_block_m=32, return_down_config=False
        )
        # Returns only the up config dict, not a tuple
        self.assertIsInstance(result, dict)
        self.assertEqual(result["BLOCK_SIZE_M"], 16)

    def test_override_config_bypasses_moe_configs_lookup(self):
        """When an override config is set, get_moe_configs is not consulted."""
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            try_get_optimal_moe_config,
        )

        override = _make_config(32)
        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_config",
                return_value=override,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_moe_configs",
            ) as mock_get_configs,
        ):
            result = try_get_optimal_moe_config(
                w1_shape=_W1_SHAPE,
                w2_shape=_W2_SHAPE,
                top_k=2,
                dtype=None,
                M=16,
                return_down_config=False,
            )
        mock_get_configs.assert_not_called()
        self.assertEqual(result["BLOCK_SIZE_M"], 32)


if __name__ == "__main__":
    unittest.main()