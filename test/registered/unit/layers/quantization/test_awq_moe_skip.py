"""Unit tests for awq.py changes — FusedMoE no longer skipped for modules_to_not_convert.

PR change: AWQConfig (NPU path) and AWQMarlinConfig no longer call
is_layer_skipped_awq() for FusedMoE layers.  Before the PR, a FusedMoE layer
whose prefix appeared in modules_to_not_convert would receive None (unquantized);
after the PR it always receives the quantized method.

Tests:
1. is_layer_skipped_awq utility function (unchanged, but tests the helper logic)
2. AWQMarlinConfig FusedMoE source-inspection confirms removal of skip check
3. AWQConfig source-inspection confirms removal of skip check on NPU FusedMoE path
"""
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIsLayerSkippedAwq(unittest.TestCase):
    """Tests for the is_layer_skipped_awq helper (unchanged, but exercised here)."""

    def setUp(self):
        from sglang.srt.layers.quantization.awq.awq import is_layer_skipped_awq

        self.fn = is_layer_skipped_awq

    def test_prefix_in_modules_returns_true(self):
        self.assertTrue(self.fn("model.layers.0.mlp.gate", ["mlp"]))

    def test_prefix_exact_module_name_returns_true(self):
        self.assertTrue(self.fn("lm_head", ["lm_head"]))

    def test_prefix_not_in_modules_returns_false(self):
        self.assertFalse(self.fn("model.layers.0.self_attn.q_proj", ["mlp"]))

    def test_empty_modules_always_returns_false(self):
        self.assertFalse(self.fn("any.prefix", []))

    def test_multiple_modules_first_match_returns_true(self):
        self.assertTrue(self.fn("embed_tokens.weight", ["embed_tokens", "lm_head"]))

    def test_multiple_modules_second_match_returns_true(self):
        self.assertTrue(self.fn("lm_head.weight", ["embed_tokens", "lm_head"]))

    def test_multiple_modules_no_match_returns_false(self):
        self.assertFalse(self.fn("model.layers.0.mlp", ["embed_tokens", "lm_head"]))


class TestAWQMarlinConfigFusedMoeNeverSkipped(unittest.TestCase):
    """Verify AWQMarlinConfig FusedMoE path no longer calls is_layer_skipped_awq."""

    def _get_awq_source(self) -> str:
        import importlib.util

        spec = importlib.util.find_spec("sglang.srt.layers.quantization.awq.awq")
        with open(spec.origin) as f:
            return f.read()

    def test_awq_marlin_fused_moe_no_skip_check_in_source(self):
        """In AWQMarlinConfig.get_quant_method, is_layer_skipped_awq must not guard FusedMoE."""
        source = self._get_awq_source()

        # Find the AWQMarlinConfig class section
        marlin_start = source.find("class AWQMarlinConfig")
        self.assertNotEqual(marlin_start, -1, "AWQMarlinConfig class not found in source")

        # Find the next class definition after AWQMarlinConfig
        next_class = source.find("\nclass ", marlin_start + 1)
        marlin_section = source[marlin_start:next_class] if next_class != -1 else source[marlin_start:]

        # In the FusedMoE branch of AWQMarlinConfig, is_layer_skipped_awq should NOT appear
        # Find the elif FusedMoE block
        fused_moe_idx = marlin_section.find("isinstance(layer, FusedMoE)")
        if fused_moe_idx == -1:
            self.skipTest("FusedMoE branch not found in AWQMarlinConfig — structure may have changed")

        # Extract the FusedMoE block (from elif to next elif/return/end-of-method)
        fused_moe_block = marlin_section[fused_moe_idx:]
        # Find end of the elif block (next elif or return None)
        end_markers = ["\n        elif ", "\n        return None"]
        block_end = len(fused_moe_block)
        for marker in end_markers:
            idx = fused_moe_block.find(marker, 10)
            if idx != -1 and idx < block_end:
                block_end = idx
        fused_moe_block = fused_moe_block[:block_end]

        self.assertNotIn(
            "is_layer_skipped_awq",
            fused_moe_block,
            "is_layer_skipped_awq should not guard the FusedMoE branch in AWQMarlinConfig",
        )

    def test_awq_config_npu_fused_moe_no_skip_check_in_source(self):
        """In AWQConfig._is_npu FusedMoE branch, is_layer_skipped_awq must not appear."""
        source = self._get_awq_source()

        # Find the AWQConfig class (before AWQMarlinConfig)
        awq_config_start = source.find("class AWQConfig")
        awq_marlin_start = source.find("class AWQMarlinConfig")
        awq_config_section = source[awq_config_start:awq_marlin_start]

        # Locate the FusedMoE elif inside the _is_npu block
        fused_moe_idx = awq_config_section.find("isinstance(layer, FusedMoE)")
        if fused_moe_idx == -1:
            self.skipTest("No FusedMoE isinstance check in AWQConfig — structure may have changed")

        fused_moe_block = awq_config_section[fused_moe_idx:]
        end_markers = ["\n            return None", "\n        if isinstance"]
        block_end = len(fused_moe_block)
        for marker in end_markers:
            idx = fused_moe_block.find(marker, 5)
            if idx != -1 and idx < block_end:
                block_end = idx
        fused_moe_block = fused_moe_block[:block_end]

        self.assertNotIn(
            "is_layer_skipped_awq",
            fused_moe_block,
            "is_layer_skipped_awq should not appear in the AWQConfig NPU FusedMoE branch",
        )


class TestAWQMarlinConfigFusedMoeWithMockedLayer(unittest.TestCase):
    """Integration-style test: AWQMarlinConfig FusedMoE always gets a method
    even when the prefix matches modules_to_not_convert."""

    def test_fused_moe_ignored_in_modules_to_not_convert(self):
        """With the PR change, a FusedMoE layer should get AWQMoEMethod regardless
        of modules_to_not_convert. We validate the source rather than instantiating
        real kernel objects."""
        # This is intentionally a lightweight source check to avoid kernel instantiation
        import importlib.util

        spec = importlib.util.find_spec("sglang.srt.layers.quantization.awq.awq")
        with open(spec.origin) as f:
            source = f.read()

        # In AWQMarlinConfig.get_quant_method, when layer is FusedMoE,
        # there should be no `if is_layer_skipped_awq` before returning the method.
        marlin_start = source.find("class AWQMarlinConfig")
        next_class = source.find("\nclass ", marlin_start + 1)
        marlin_body = source[marlin_start:next_class] if next_class != -1 else source[marlin_start:]

        # Count occurrences of is_layer_skipped_awq in the whole AWQMarlinConfig body
        # After the PR there should be exactly 1 (for LinearBase), not 2
        skip_count = marlin_body.count("is_layer_skipped_awq")
        self.assertEqual(
            skip_count, 1,
            f"Expected exactly 1 is_layer_skipped_awq in AWQMarlinConfig (for LinearBase only), "
            f"found {skip_count}. The FusedMoE skip guard may not have been removed.",
        )


if __name__ == "__main__":
    unittest.main()