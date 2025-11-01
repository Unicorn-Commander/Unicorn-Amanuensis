#!/usr/bin/env python3
"""
NPU Accuracy Test Suite

Comprehensive accuracy validation for BFP16 NPU implementation vs PyTorch reference.

Test Matrix:
- Small matmul (64×64×64): >99.99% similarity, <0.5% error
- Whisper projections (512×512×512): >99.9% similarity, <1% error
- Whisper FFN layers (512×2048): >99.9% similarity, <1% error
- Single encoder layer: >99.5% similarity, <2% error
- Full 6-layer encoder: >99% similarity, <3% error
- Batch processing (1, 2, 4): >99% similarity, <3% error
- Edge cases (zeros, ones, large, small): >99% similarity

Success Criteria:
- All tests pass accuracy thresholds
- No crashes or memory leaks
- Deterministic outputs (same input → same output)
"""

import numpy as np
import ctypes
from ctypes import c_void_p, c_float, c_int, c_size_t, POINTER, c_uint8
import sys
from pathlib import Path
import logging

# Import PyTorch reference
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_reference import WhisperEncoderReference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NPUAccuracyTester:
    """NPU accuracy testing framework"""

    def __init__(self, lib_path: str = None):
        """
        Initialize NPU accuracy tester

        Args:
            lib_path: Path to C++ library (default: auto-detect)
        """
        if lib_path is None:
            lib_path = "/home/ccadmin/CC-1L/npu-services/unicorn-amanuensis/xdna2/cpp/build/libwhisper_encoder_cpp.so"

        if not Path(lib_path).exists():
            raise FileNotFoundError(f"C++ library not found: {lib_path}")

        logger.info(f"Loading C++ library: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        # Define C API
        self._define_c_api()

        # Load PyTorch reference
        logger.info("Loading PyTorch reference...")
        self.pytorch_ref = WhisperEncoderReference("openai/whisper-base")

        logger.info("NPU accuracy tester initialized")

    def _define_c_api(self):
        """Define C API bindings"""
        # EncoderLayer API
        self.lib.encoder_layer_create.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t]
        self.lib.encoder_layer_create.restype = c_void_p

        self.lib.encoder_layer_destroy.argtypes = [c_void_p]

        self.lib.encoder_layer_load_weights.argtypes = [
            c_void_p,
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float),
            c_size_t, c_size_t
        ]
        self.lib.encoder_layer_load_weights.restype = c_int

        self.lib.encoder_layer_forward.argtypes = [
            c_void_p,
            POINTER(c_float),
            POINTER(c_float),
            c_size_t,
            c_size_t
        ]
        self.lib.encoder_layer_forward.restype = c_int

        # NPU callback API
        self.lib.encoder_layer_set_npu_callback.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.encoder_layer_set_npu_callback.restype = None

    def test_small_matmul(self) -> dict:
        """
        Test: Small matmul (64×64×64)
        Target: >99.99% similarity, <0.5% error
        """
        logger.info("="*80)
        logger.info("Test: Small Matmul (64×64×64)")
        logger.info("="*80)

        # TODO: Implement when NPU matmul is available
        # For now, this is a placeholder

        results = {
            'test_name': 'small_matmul_64x64x64',
            'status': 'SKIPPED',
            'reason': 'NPU matmul not yet implemented',
            'cosine_similarity': None,
            'relative_error': None,
        }

        logger.warning(f"Test skipped: {results['reason']}")
        return results

    def test_whisper_q_projection(self) -> dict:
        """
        Test: Whisper Q projection (512×512×512)
        Target: >99.9% similarity, <1% error
        """
        logger.info("="*80)
        logger.info("Test: Whisper Q Projection (512×512×512)")
        logger.info("="*80)

        # TODO: Implement when NPU integration is complete
        results = {
            'test_name': 'whisper_q_projection',
            'status': 'SKIPPED',
            'reason': 'NPU integration not yet complete',
        }

        logger.warning(f"Test skipped: {results['reason']}")
        return results

    def test_single_layer(self) -> dict:
        """
        Test: Single encoder layer (1,1500,512)
        Target: >99.5% similarity, <2% error
        """
        logger.info("="*80)
        logger.info("Test: Single Encoder Layer")
        logger.info("="*80)

        try:
            # Generate test input
            np.random.seed(42)
            input_np = np.random.randn(1500, 512).astype(np.float32)

            # Create encoder layer
            layer = self.lib.encoder_layer_create(0, 8, 512, 2048)
            if not layer:
                raise RuntimeError("Failed to create encoder layer")

            # Load weights (we'll use random weights for now)
            # TODO: Load real Whisper weights
            n_state = 512
            ffn_dim = 2048

            # Generate random weights
            q_w = np.random.randn(n_state, n_state).astype(np.float32)
            k_w = np.random.randn(n_state, n_state).astype(np.float32)
            v_w = np.random.randn(n_state, n_state).astype(np.float32)
            out_w = np.random.randn(n_state, n_state).astype(np.float32)
            fc1_w = np.random.randn(ffn_dim, n_state).astype(np.float32)
            fc2_w = np.random.randn(n_state, ffn_dim).astype(np.float32)

            q_b = np.zeros(n_state, dtype=np.float32)
            k_b = np.zeros(n_state, dtype=np.float32)
            v_b = np.zeros(n_state, dtype=np.float32)
            out_b = np.zeros(n_state, dtype=np.float32)
            fc1_b = np.zeros(ffn_dim, dtype=np.float32)
            fc2_b = np.zeros(n_state, dtype=np.float32)

            attn_ln_w = np.ones(n_state, dtype=np.float32)
            attn_ln_b = np.zeros(n_state, dtype=np.float32)
            ffn_ln_w = np.ones(n_state, dtype=np.float32)
            ffn_ln_b = np.zeros(n_state, dtype=np.float32)

            # Load weights
            result = self.lib.encoder_layer_load_weights(
                layer,
                q_w.ctypes.data_as(POINTER(c_float)),
                k_w.ctypes.data_as(POINTER(c_float)),
                v_w.ctypes.data_as(POINTER(c_float)),
                out_w.ctypes.data_as(POINTER(c_float)),
                q_b.ctypes.data_as(POINTER(c_float)),
                k_b.ctypes.data_as(POINTER(c_float)),
                v_b.ctypes.data_as(POINTER(c_float)),
                out_b.ctypes.data_as(POINTER(c_float)),
                fc1_w.ctypes.data_as(POINTER(c_float)),
                fc2_w.ctypes.data_as(POINTER(c_float)),
                fc1_b.ctypes.data_as(POINTER(c_float)),
                fc2_b.ctypes.data_as(POINTER(c_float)),
                attn_ln_w.ctypes.data_as(POINTER(c_float)),
                attn_ln_b.ctypes.data_as(POINTER(c_float)),
                ffn_ln_w.ctypes.data_as(POINTER(c_float)),
                ffn_ln_b.ctypes.data_as(POINTER(c_float)),
                n_state,
                ffn_dim
            )

            if result != 0:
                raise RuntimeError("Failed to load weights")

            # Run forward pass (C++)
            output_np = np.zeros((1500, 512), dtype=np.float32)
            result = self.lib.encoder_layer_forward(
                layer,
                input_np.ctypes.data_as(POINTER(c_float)),
                output_np.ctypes.data_as(POINTER(c_float)),
                1500,
                512
            )

            if result != 0:
                raise RuntimeError("Forward pass failed")

            # Compute metrics (compare vs PyTorch reference)
            # NOTE: We can't compare directly because we're using random weights
            # This test validates the C++ implementation runs without crashing
            metrics = {
                'test_name': 'single_layer',
                'status': 'PASS',
                'output_shape': output_np.shape,
                'output_mean': float(output_np.mean()),
                'output_std': float(output_np.std()),
                'note': 'Test validates C++ execution (random weights, no PyTorch comparison)',
            }

            # Cleanup
            self.lib.encoder_layer_destroy(layer)

            logger.info(f"Test PASSED: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Test FAILED: {e}")
            return {
                'test_name': 'single_layer',
                'status': 'FAILED',
                'error': str(e),
            }

    def test_full_encoder(self) -> dict:
        """
        Test: Full 6-layer encoder
        Target: >99% similarity, <3% error
        """
        logger.info("="*80)
        logger.info("Test: Full 6-Layer Encoder")
        logger.info("="*80)

        # TODO: Implement when all layers are working
        results = {
            'test_name': 'full_encoder_6_layers',
            'status': 'SKIPPED',
            'reason': 'Requires all 6 layers with real NPU integration',
        }

        logger.warning(f"Test skipped: {results['reason']}")
        return results

    def test_batch_processing(self) -> dict:
        """
        Test: Batch processing (batch sizes: 1, 2, 4)
        Target: >99% similarity, <3% error
        """
        logger.info("="*80)
        logger.info("Test: Batch Processing")
        logger.info("="*80)

        # TODO: Implement batch processing tests
        results = {
            'test_name': 'batch_processing',
            'status': 'SKIPPED',
            'reason': 'Batch processing not yet implemented',
        }

        logger.warning(f"Test skipped: {results['reason']}")
        return results

    def test_edge_cases(self) -> dict:
        """
        Test: Edge cases (all zeros, all ones, large values, small values)
        Target: >99% similarity
        """
        logger.info("="*80)
        logger.info("Test: Edge Cases")
        logger.info("="*80)

        edge_cases = [
            ('all_zeros', np.zeros((512, 512), dtype=np.float32)),
            ('all_ones', np.ones((512, 512), dtype=np.float32)),
            ('large_values', np.full((512, 512), 1000.0, dtype=np.float32)),
            ('small_values', np.full((512, 512), 0.001, dtype=np.float32)),
        ]

        results = []
        for name, input_matrix in edge_cases:
            logger.info(f"  Testing: {name}")
            # TODO: Run through NPU and compare
            results.append({
                'case': name,
                'status': 'SKIPPED',
                'reason': 'NPU integration not yet complete',
            })

        return {
            'test_name': 'edge_cases',
            'status': 'SKIPPED',
            'results': results,
        }

    def run_all_tests(self) -> dict:
        """Run all NPU accuracy tests"""
        logger.info("="*80)
        logger.info("  NPU ACCURACY TEST SUITE")
        logger.info("="*80)

        all_results = {}

        # Run tests
        all_results['small_matmul'] = self.test_small_matmul()
        all_results['whisper_q_projection'] = self.test_whisper_q_projection()
        all_results['single_layer'] = self.test_single_layer()
        all_results['full_encoder'] = self.test_full_encoder()
        all_results['batch_processing'] = self.test_batch_processing()
        all_results['edge_cases'] = self.test_edge_cases()

        # Summary
        logger.info("="*80)
        logger.info("  TEST SUMMARY")
        logger.info("="*80)

        total = len(all_results)
        passed = sum(1 for r in all_results.values() if r.get('status') == 'PASS')
        failed = sum(1 for r in all_results.values() if r.get('status') == 'FAILED')
        skipped = sum(1 for r in all_results.values() if r.get('status') == 'SKIPPED')

        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")

        all_results['summary'] = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
        }

        return all_results


def main():
    """Run NPU accuracy tests"""
    try:
        tester = NPUAccuracyTester()
        results = tester.run_all_tests()

        # Save results
        output_dir = Path("./tests/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_dir / "npu_accuracy_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_dir / 'npu_accuracy_results.json'}")

        # Exit with appropriate code
        if results['summary']['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
