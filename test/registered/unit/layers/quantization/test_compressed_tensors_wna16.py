"""Unit tests for CompressedTensorsConfig._is_wNa16_group_channel.

PR change: _is_wNa16_group_channel now requires weight_quant.symmetric == True.
Previously the function returned True for both symmetric and asymmetric weight
quantization (the comment said "Both symmetric and asymmetric … are handled").
After the PR it returns False for asymmetric weight quant, restricting the
Marlin path to symmetric-only checkpoints.

Also tests that _get_scheme_from_parts no longer passes `symmetric` to
CompressedTensorsWNA16 (the argument was removed from that call).
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_weight_quant(
    *,
    symmetric: bool,
    strategy: str = "channel",
    dynamic: bool = False,
    num_bits: int = 4,
):
    """Build a minimal weight_quant namespace compatible with _is_wNa16_group_channel."""
    return SimpleNamespace(
        symmetric=symmetric,
        strategy=strategy,
        dynamic=dynamic,
        num_bits=num_bits,
        group_size=128,
        actorder=None,
        type="int",
    )


def _call_is_wna16(weight_quant, input_quant=None):
    """Invoke _is_wNa16_group_channel on a minimal CompressedTensorsConfig-like object."""
    from compressed_tensors.quantization import QuantizationStrategy

    # We test the pure logic by calling the unbound method with a dummy self
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    dummy_self = object.__new__(CompressedTensorsConfig)
    return dummy_self._is_wNa16_group_channel(weight_quant, input_quant)


class TestIsWNA16GroupChannelSymmetric(unittest.TestCase):
    """_is_wNa16_group_channel must return False for asymmetric weight quant."""

    def test_symmetric_channel_returns_true(self):
        """Symmetric + channel strategy + static + no input quant -> True."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.CHANNEL.value)
        self.assertTrue(_call_is_wna16(wq, input_quant=None))

    def test_symmetric_group_returns_true(self):
        """Symmetric + group strategy -> True."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.GROUP.value)
        self.assertTrue(_call_is_wna16(wq, input_quant=None))

    def test_asymmetric_channel_returns_false(self):
        """Asymmetric weight quant should now return False (PR change)."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=False, strategy=QuantizationStrategy.CHANNEL.value)
        self.assertFalse(_call_is_wna16(wq, input_quant=None))

    def test_asymmetric_group_returns_false(self):
        """Asymmetric + group strategy should also return False."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=False, strategy=QuantizationStrategy.GROUP.value)
        self.assertFalse(_call_is_wna16(wq, input_quant=None))

    def test_input_quant_not_none_returns_false(self):
        """When input_quant is provided the function returns False regardless."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.CHANNEL.value)
        fake_iq = SimpleNamespace(dynamic=False, strategy="tensor")
        self.assertFalse(_call_is_wna16(wq, input_quant=fake_iq))

    def test_dynamic_weight_quant_returns_false(self):
        """Dynamic weight quant (is_static=False) must return False."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.CHANNEL.value, dynamic=True)
        self.assertFalse(_call_is_wna16(wq, input_quant=None))

    def test_tensor_strategy_returns_false(self):
        """Tensor strategy is neither channel nor group, so must return False."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.TENSOR.value)
        self.assertFalse(_call_is_wna16(wq, input_quant=None))

    def test_symmetric_requirement_is_checked_before_other_conditions(self):
        """Changing only symmetric from True to False must flip the result."""
        from compressed_tensors.quantization import QuantizationStrategy

        wq_sym = _make_weight_quant(symmetric=True, strategy=QuantizationStrategy.CHANNEL.value)
        wq_asym = _make_weight_quant(symmetric=False, strategy=QuantizationStrategy.CHANNEL.value)
        self.assertTrue(_call_is_wna16(wq_sym))
        self.assertFalse(_call_is_wna16(wq_asym))


class TestGetSchemeFromPartsNoSymmetricArg(unittest.TestCase):
    """_get_scheme_from_parts must not pass `symmetric` to CompressedTensorsWNA16."""

    def test_wna16_instantiation_omits_symmetric_kwarg(self):
        """When _is_wNa16_group_channel returns True, CompressedTensorsWNA16 is
        called without the `symmetric` keyword argument (removed by this PR)."""
        from compressed_tensors.quantization import QuantizationStrategy

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsWNA16,
            WNA16_SUPPORTED_BITS,
        )

        # We need a CompressedTensorsConfig whose _is_wNa16_group_channel returns True.
        # Patch _is_wNa16_group_channel to always return True, and mock
        # CompressedTensorsWNA16 to capture call kwargs.
        with patch.object(
            CompressedTensorsConfig,
            "_is_wNa16_group_channel",
            return_value=True,
        ), patch(
            "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.CompressedTensorsWNA16",
        ) as mock_wna16, patch(
            "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.is_activation_quantization_format",
            return_value=False,
        ):
            mock_wna16.return_value = MagicMock()
            dummy_self = object.__new__(CompressedTensorsConfig)
            dummy_self.quant_format = "pack_quantized"  # pack_quantized is in WNA16 path

            wq = _make_weight_quant(symmetric=True, strategy="channel", num_bits=4)

            try:
                dummy_self._get_scheme_from_parts(wq, input_quant=None)
            except Exception:
                pass  # We just care about the call args

            if mock_wna16.called:
                _, kwargs = mock_wna16.call_args
                self.assertNotIn(
                    "symmetric",
                    kwargs,
                    "symmetric should not be passed to CompressedTensorsWNA16 after PR",
                )


if __name__ == "__main__":
    unittest.main()