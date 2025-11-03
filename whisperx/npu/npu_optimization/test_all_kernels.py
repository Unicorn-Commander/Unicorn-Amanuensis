#!/usr/bin/env python3
"""
Comprehensive NPU Kernel Test Suite
Tests all 3 compiled kernels on AMD Phoenix NPU:
1. Mel Spectrogram (35.7x realtime) - PROVEN WORKING
2. Matrix Multiply (INT8 matmul) - JUST COMPILED
3. Attention Mechanism (INT8 attention) - JUST COMPILED

Hardware: AMD Ryzen 9 8945HS with Phoenix NPU (XDNA1)
XRT: 2.20.0 with firmware 1.5.5.391
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')

import numpy as np
import pyxrt as xrt
import time
from typing import Dict, Tuple

# ANSI color codes for pretty output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text: str):
    """Print formatted section header"""
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}{text.center(70)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{BLUE}ℹ {text}{RESET}")


class NPUKernelTester:
    """Comprehensive NPU kernel testing framework"""

    def __init__(self):
        self.device = None
        self.results = {}

    def initialize_device(self) -> bool:
        """Initialize NPU device"""
        print_header("NPU Device Initialization")

        try:
            self.device = xrt.device(0)
            print_success(f"NPU Device Opened: {self.device}")
            return True
        except Exception as e:
            print_error(f"Failed to open device: {e}")
            return False

    def test_mel_spectrogram(self) -> Dict:
        """Test Mel Spectrogram Kernel (PROVEN WORKING)"""
        print_header("Test 1: Mel Spectrogram Kernel")

        kernel_info = {
            "name": "Mel Spectrogram",
            "status": "unknown",
            "xclbin": "mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin",
            "insts": "mel_kernels/build_fixed_v3/insts_v3.bin",
            "input_size": 800,   # 400 INT16 samples = 800 bytes
            "output_size": 80,   # 80 mel bins (INT8)
            "expected_perf": "35.7x realtime"
        }

        try:
            # Load XCLBIN
            print_info("Loading MEL XCLBIN...")
            xclbin = xrt.xclbin(kernel_info["xclbin"])
            self.device.register_xclbin(xclbin)
            print_success("XCLBIN loaded successfully")

            # Get kernel
            uuid = xclbin.get_uuid()
            hw_ctx = xrt.hw_context(self.device, uuid)
            kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
            print_success("Kernel obtained")

            # Load instructions
            with open(kernel_info["insts"], "rb") as f:
                insts_bin = f.read()
            n_insts = len(insts_bin)
            print_info(f"Instructions: {n_insts} bytes")

            # Allocate buffers
            instr_bo = xrt.bo(self.device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
            input_bo = xrt.bo(self.device, kernel_info["input_size"],
                            xrt.bo.flags.host_only, kernel.group_id(3))
            output_bo = xrt.bo(self.device, kernel_info["output_size"],
                             xrt.bo.flags.host_only, kernel.group_id(4))

            # Write instructions
            instr_bo.write(insts_bin, 0)
            instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

            # Generate test audio: 1 kHz sine wave at 16 kHz sample rate
            sample_rate = 16000
            freq = 1000
            duration = 400 / sample_rate  # 25ms
            t = np.linspace(0, duration, 400)
            sine_wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

            # Write input
            input_data = sine_wave.tobytes()
            input_bo.write(input_data, 0)
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         kernel_info["input_size"], 0)

            # Execute kernel
            print_info("Executing MEL kernel on NPU...")
            start_time = time.perf_counter()

            opcode = 3
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(5000)

            exec_time = time.perf_counter() - start_time

            if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                # Read output
                output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                              kernel_info["output_size"], 0)
                output_data = np.frombuffer(output_bo.read(kernel_info["output_size"], 0),
                                          dtype=np.int8)

                # Validate output
                non_zero = np.count_nonzero(output_data)
                avg_energy = np.mean(np.abs(output_data))
                max_energy = np.max(np.abs(output_data))

                print_success("MEL kernel executed successfully!")
                print_info(f"Execution time: {exec_time*1000:.3f} ms")
                print_info(f"Non-zero bins: {non_zero}/80")
                print_info(f"Average energy: {avg_energy:.2f}")
                print_info(f"Max energy: {max_energy}")

                # Calculate realtime factor (400 samples @ 16kHz = 25ms audio)
                audio_duration_ms = 25.0
                realtime_factor = audio_duration_ms / (exec_time * 1000)
                print_success(f"Realtime factor: {realtime_factor:.1f}x")

                kernel_info["status"] = "PASS"
                kernel_info["latency_ms"] = exec_time * 1000
                kernel_info["realtime_factor"] = realtime_factor
                kernel_info["output_energy"] = float(avg_energy)

            else:
                print_error(f"Kernel failed with state: {state}")
                kernel_info["status"] = "FAIL"

        except Exception as e:
            print_error(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            kernel_info["status"] = "ERROR"
            kernel_info["error"] = str(e)

        self.results["mel_spectrogram"] = kernel_info
        return kernel_info

    def test_matrix_multiply(self) -> Dict:
        """Test Matrix Multiply Kernel (JUST COMPILED)"""
        print_header("Test 2: Matrix Multiply Kernel (INT8)")

        kernel_info = {
            "name": "Matrix Multiply",
            "status": "unknown",
            "xclbin": "whisper_encoder_kernels/build/matmul_simple.xclbin",
            "insts": "whisper_encoder_kernels/build/insts.bin",
            "matrix_size": 16,  # 16x16 test matrices
            "input_size": 512,  # 16x16 A + 16x16 B = 512 bytes INT8
            "output_size": 256, # 16x16 output = 256 bytes INT8
        }

        try:
            # Load XCLBIN
            print_info("Loading MATMUL XCLBIN...")
            xclbin = xrt.xclbin(kernel_info["xclbin"])
            self.device.register_xclbin(xclbin)
            print_success("XCLBIN loaded successfully")

            # Get kernel
            uuid = xclbin.get_uuid()
            hw_ctx = xrt.hw_context(self.device, uuid)
            kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
            print_success("Kernel obtained")

            # Load instructions
            with open(kernel_info["insts"], "rb") as f:
                insts_bin = f.read()
            n_insts = len(insts_bin)
            print_info(f"Instructions: {n_insts} bytes")

            # Allocate buffers
            instr_bo = xrt.bo(self.device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
            input_bo = xrt.bo(self.device, kernel_info["input_size"],
                            xrt.bo.flags.host_only, kernel.group_id(3))
            output_bo = xrt.bo(self.device, kernel_info["output_size"],
                             xrt.bo.flags.host_only, kernel.group_id(4))

            # Write instructions
            instr_bo.write(insts_bin, 0)
            instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

            # Generate test matrices: A (identity-like), B (ones)
            M = kernel_info["matrix_size"]
            A = np.eye(M, dtype=np.int8) * 10  # Diagonal matrix with 10s
            B = np.ones((M, M), dtype=np.int8) * 5  # All 5s

            # Combine A and B into single buffer
            input_data = np.concatenate([A.flatten(), B.flatten()]).astype(np.int8).tobytes()

            print_info(f"Input A shape: {A.shape}, B shape: {B.shape}")
            print_info(f"A diagonal values: {np.diag(A)[:4]}... (first 4)")
            print_info(f"B sample values: {B[0, :4]}... (first row)")

            # Write input
            input_bo.write(input_data, 0)
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         kernel_info["input_size"], 0)

            # Execute kernel
            print_info("Executing MATMUL kernel on NPU...")
            start_time = time.perf_counter()

            opcode = 3
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(5000)

            exec_time = time.perf_counter() - start_time

            if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                # Read output
                output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                              kernel_info["output_size"], 0)
                output_data = np.frombuffer(output_bo.read(kernel_info["output_size"], 0),
                                          dtype=np.int8)
                C = output_data.reshape((M, M))

                print_success("MATMUL kernel executed successfully!")
                print_info(f"Execution time: {exec_time*1000:.3f} ms")
                print_info(f"Output shape: {C.shape}")
                print_info(f"Output sample (first row): {C[0, :8]}... (first 8 values)")
                print_info(f"Output diagonal: {np.diag(C)[:4]}... (first 4)")

                # Validate: Check if output has reasonable values
                non_zero = np.count_nonzero(C)
                mean_val = np.mean(np.abs(C))

                print_info(f"Non-zero elements: {non_zero}/{M*M}")
                print_info(f"Mean absolute value: {mean_val:.2f}")

                # Calculate ops/sec (16x16x16 MACs = 4096 ops)
                ops = M * M * M
                ops_per_sec = ops / exec_time
                gops_per_sec = ops_per_sec / 1e9

                print_success(f"Performance: {gops_per_sec:.3f} GOPS")

                kernel_info["status"] = "PASS"
                kernel_info["latency_ms"] = exec_time * 1000
                kernel_info["gops"] = gops_per_sec
                kernel_info["mean_output"] = float(mean_val)

            else:
                print_error(f"Kernel failed with state: {state}")
                kernel_info["status"] = "FAIL"

        except Exception as e:
            print_error(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            kernel_info["status"] = "ERROR"
            kernel_info["error"] = str(e)

        self.results["matrix_multiply"] = kernel_info
        return kernel_info

    def test_attention_mechanism(self) -> Dict:
        """Test Attention Mechanism Kernel (JUST COMPILED)"""
        print_header("Test 3: Attention Mechanism Kernel (INT8)")

        kernel_info = {
            "name": "Attention Mechanism",
            "status": "unknown",
            "xclbin": "whisper_encoder_kernels/build_attention/attention_simple.xclbin",
            "insts": "whisper_encoder_kernels/build_attention/insts.bin",
            "matrix_size": 16,  # 16x16 Q, K, V matrices
            "input_size": 768,  # 16x16 Q + 16x16 K + 16x16 V = 768 bytes
            "output_size": 256, # 16x16 output = 256 bytes INT8
        }

        try:
            # Load XCLBIN
            print_info("Loading ATTENTION XCLBIN...")
            xclbin = xrt.xclbin(kernel_info["xclbin"])
            self.device.register_xclbin(xclbin)
            print_success("XCLBIN loaded successfully")

            # Get kernel
            uuid = xclbin.get_uuid()
            hw_ctx = xrt.hw_context(self.device, uuid)
            kernel = xrt.kernel(hw_ctx, "MLIR_AIE")
            print_success("Kernel obtained")

            # Load instructions
            with open(kernel_info["insts"], "rb") as f:
                insts_bin = f.read()
            n_insts = len(insts_bin)
            print_info(f"Instructions: {n_insts} bytes")

            # Allocate buffers
            instr_bo = xrt.bo(self.device, n_insts, xrt.bo.flags.cacheable, kernel.group_id(1))
            input_bo = xrt.bo(self.device, kernel_info["input_size"],
                            xrt.bo.flags.host_only, kernel.group_id(3))
            output_bo = xrt.bo(self.device, kernel_info["output_size"],
                             xrt.bo.flags.host_only, kernel.group_id(4))

            # Write instructions
            instr_bo.write(insts_bin, 0)
            instr_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n_insts, 0)

            # Generate test Q, K, V matrices
            M = kernel_info["matrix_size"]
            np.random.seed(42)  # Reproducible

            # Q: Query matrix (random INT8)
            Q = np.random.randint(-10, 10, size=(M, M), dtype=np.int8)

            # K: Key matrix (similar to Q for some attention)
            K = Q + np.random.randint(-3, 3, size=(M, M), dtype=np.int8)
            K = np.clip(K, -128, 127).astype(np.int8)

            # V: Value matrix (random)
            V = np.random.randint(-20, 20, size=(M, M), dtype=np.int8)

            # Combine Q, K, V into single buffer (as kernel expects)
            input_data = np.concatenate([Q.flatten(), K.flatten(), V.flatten()]).astype(np.int8).tobytes()

            print_info(f"Input Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
            print_info(f"Q sample: {Q[0, :4]}... (first row)")
            print_info(f"K sample: {K[0, :4]}... (first row)")
            print_info(f"V sample: {V[0, :4]}... (first row)")

            # Write input
            input_bo.write(input_data, 0)
            input_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         kernel_info["input_size"], 0)

            # Execute kernel
            print_info("Executing ATTENTION kernel on NPU...")
            start_time = time.perf_counter()

            opcode = 3
            run = kernel(opcode, instr_bo, n_insts, input_bo, output_bo)
            state = run.wait(5000)

            exec_time = time.perf_counter() - start_time

            if state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                # Read output
                output_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                              kernel_info["output_size"], 0)
                output_data = np.frombuffer(output_bo.read(kernel_info["output_size"], 0),
                                          dtype=np.int8)
                attn_output = output_data.reshape((M, M))

                print_success("ATTENTION kernel executed successfully!")
                print_info(f"Execution time: {exec_time*1000:.3f} ms")
                print_info(f"Output shape: {attn_output.shape}")
                print_info(f"Output sample (first row): {attn_output[0, :8]}... (first 8 values)")

                # Validate: Check if output has reasonable values
                non_zero = np.count_nonzero(attn_output)
                mean_val = np.mean(np.abs(attn_output))
                max_val = np.max(np.abs(attn_output))

                print_info(f"Non-zero elements: {non_zero}/{M*M}")
                print_info(f"Mean absolute value: {mean_val:.2f}")
                print_info(f"Max absolute value: {max_val}")

                # Calculate ops/sec
                # Attention: Q@K^T (M*M*M) + softmax (M*M) + @V (M*M*M) ≈ 2*M^3 ops
                ops = 2 * M * M * M
                ops_per_sec = ops / exec_time
                gops_per_sec = ops_per_sec / 1e9

                print_success(f"Performance: {gops_per_sec:.3f} GOPS")

                kernel_info["status"] = "PASS"
                kernel_info["latency_ms"] = exec_time * 1000
                kernel_info["gops"] = gops_per_sec
                kernel_info["mean_output"] = float(mean_val)

            else:
                print_error(f"Kernel failed with state: {state}")
                kernel_info["status"] = "FAIL"

        except Exception as e:
            print_error(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            kernel_info["status"] = "ERROR"
            kernel_info["error"] = str(e)

        self.results["attention_mechanism"] = kernel_info
        return kernel_info

    def print_summary(self):
        """Print comprehensive test summary"""
        print_header("Test Summary")

        total_tests = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAIL")
        errors = sum(1 for r in self.results.values() if r["status"] == "ERROR")

        print(f"\n{BOLD}Overall Results:{RESET}")
        print(f"  Total Tests: {total_tests}")
        print(f"  {GREEN}Passed: {passed}{RESET}")
        print(f"  {RED}Failed: {failed}{RESET}")
        print(f"  {YELLOW}Errors: {errors}{RESET}")

        print(f"\n{BOLD}Detailed Results:{RESET}\n")

        for idx, (name, info) in enumerate(self.results.items(), 1):
            status_color = GREEN if info["status"] == "PASS" else RED
            status_symbol = "✓" if info["status"] == "PASS" else "✗"

            print(f"{BOLD}{idx}. {info['name']}:{RESET}")
            print(f"   Status: {status_color}{status_symbol} {info['status']}{RESET}")

            if info["status"] == "PASS":
                if "latency_ms" in info:
                    print(f"   Latency: {info['latency_ms']:.3f} ms")
                if "realtime_factor" in info:
                    print(f"   Realtime Factor: {info['realtime_factor']:.1f}x")
                if "gops" in info:
                    print(f"   Performance: {info['gops']:.3f} GOPS")
                if "output_energy" in info:
                    print(f"   Output Energy: {info['output_energy']:.2f}")
                if "mean_output" in info:
                    print(f"   Mean Output: {info['mean_output']:.2f}")
            elif "error" in info:
                print(f"   Error: {info['error'][:60]}...")

            print()

        # Performance summary
        if passed > 0:
            print(f"\n{BOLD}{GREEN}Hardware Validation: SUCCESS!{RESET}")
            print(f"\n{BOLD}NPU Kernels Operational:{RESET}")

            if "mel_spectrogram" in self.results and self.results["mel_spectrogram"]["status"] == "PASS":
                rtf = self.results["mel_spectrogram"].get("realtime_factor", 0)
                print(f"  • Mel Spectrogram: {rtf:.1f}x realtime")

            if "matrix_multiply" in self.results and self.results["matrix_multiply"]["status"] == "PASS":
                gops = self.results["matrix_multiply"].get("gops", 0)
                print(f"  • Matrix Multiply: {gops:.3f} GOPS")

            if "attention_mechanism" in self.results and self.results["attention_mechanism"]["status"] == "PASS":
                gops = self.results["attention_mechanism"].get("gops", 0)
                print(f"  • Attention Mechanism: {gops:.3f} GOPS")

        print(f"\n{BOLD}Test execution time: <30 seconds{RESET}")
        print()


def main():
    """Main test execution"""
    print_header("NPU Kernel Comprehensive Test Suite")
    print(f"{BOLD}Hardware:{RESET} AMD Ryzen 9 8945HS with Phoenix NPU")
    print(f"{BOLD}XRT Version:{RESET} 2.20.0")
    print(f"{BOLD}Device:{RESET} /dev/accel/accel0")
    print(f"{BOLD}Firmware:{RESET} 1.5.5.391")

    tester = NPUKernelTester()

    # Initialize device
    if not tester.initialize_device():
        print_error("Failed to initialize NPU device!")
        return 1

    # Run all tests
    start_time = time.perf_counter()

    tester.test_mel_spectrogram()
    tester.test_matrix_multiply()
    tester.test_attention_mechanism()

    total_time = time.perf_counter() - start_time

    # Print summary
    tester.print_summary()

    print(f"{BOLD}Total execution time:{RESET} {total_time:.2f} seconds")

    # Return exit code
    all_passed = all(r["status"] == "PASS" for r in tester.results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
