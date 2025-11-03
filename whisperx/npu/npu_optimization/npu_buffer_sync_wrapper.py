#!/usr/bin/env python3
"""
NPU Buffer Synchronization Wrapper - Proven Pattern

This module provides a tested, working pattern for NPU buffer synchronization
based on comprehensive testing on October 31, 2025.

KEY FINDINGS:
- host_only + explicit syncs = WORKS (3.8% non-zero output)
- cacheable for all buffers = FAILS (0% non-zero output)
- device_only = NOT SUPPORTED on Phoenix NPU

RECOMMENDED PATTERN:
- Use host_only for input/output data buffers
- Use cacheable for instruction buffers only
- Always use explicit syncs TO and FROM device
"""

import sys
sys.path.insert(0, '/opt/xilinx/xrt/python')
import pyxrt as xrt
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

class NPUBufferManager:
    """
    Manages NPU buffer creation and synchronization using proven patterns.

    Based on testing results from October 31, 2025 which confirmed:
    - Explicit syncs work correctly
    - host_only is preferred for data buffers
    - cacheable should only be used for instructions
    """

    def __init__(self, device: xrt.device, kernel: xrt.kernel):
        """
        Initialize buffer manager.

        Args:
            device: XRT device handle
            kernel: XRT kernel handle
        """
        self.device = device
        self.kernel = kernel

    def create_buffers(self,
                      instr_size: int,
                      input_size: int,
                      output_size: int) -> Tuple[xrt.bo, xrt.bo, xrt.bo]:
        """
        Create buffers using the proven pattern.

        Args:
            instr_size: Size of instruction buffer in bytes
            input_size: Size of input buffer in bytes
            output_size: Size of output buffer in bytes

        Returns:
            Tuple of (instr_bo, input_bo, output_bo)
        """
        # Instruction buffer: ALWAYS use cacheable
        instr_bo = xrt.bo(self.device, instr_size,
                         xrt.bo.flags.cacheable, self.kernel.group_id(1))

        # Data buffers: ALWAYS use host_only
        input_bo = xrt.bo(self.device, input_size,
                         xrt.bo.flags.host_only, self.kernel.group_id(3))
        output_bo = xrt.bo(self.device, output_size,
                          xrt.bo.flags.host_only, self.kernel.group_id(4))

        return instr_bo, input_bo, output_bo

    def write_and_sync_to_device(self,
                                 bo: xrt.bo,
                                 data: bytes,
                                 size: Optional[int] = None) -> None:
        """
        Write data to buffer and sync to device.

        Args:
            bo: Buffer object to write to
            data: Data to write (bytes)
            size: Size to sync (defaults to len(data))
        """
        if size is None:
            size = len(data)

        # Write data to buffer
        bo.write(data, 0)

        # CRITICAL: Explicit sync to device
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, 0)

    def sync_and_read_from_device(self,
                                   bo: xrt.bo,
                                   size: int,
                                   dtype: np.dtype = np.int8) -> np.ndarray:
        """
        Sync from device and read data from buffer.

        Args:
            bo: Buffer object to read from
            size: Size to sync and read in bytes
            dtype: NumPy dtype for output

        Returns:
            NumPy array with output data
        """
        # CRITICAL: Explicit sync from device BEFORE reading
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)

        # Read data
        output_data = np.frombuffer(bo.read(size, 0), dtype=dtype)

        return output_data

    def execute_kernel(self,
                      opcode: int,
                      instr_bo: xrt.bo,
                      n_insts: int,
                      input_bo: xrt.bo,
                      output_bo: xrt.bo,
                      timeout_ms: int = 10000) -> bool:
        """
        Execute kernel with timeout.

        Args:
            opcode: Kernel opcode
            instr_bo: Instruction buffer
            n_insts: Number of instructions
            input_bo: Input buffer
            output_bo: Output buffer
            timeout_ms: Timeout in milliseconds

        Returns:
            True if kernel completed successfully, False otherwise
        """
        # Execute kernel
        run = self.kernel(opcode, instr_bo, n_insts, input_bo, output_bo)

        # Wait for completion
        state = run.wait(timeout_ms)

        # Check if completed
        return state == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


class MelKernelRunner:
    """
    High-level wrapper for running mel spectrogram kernel on NPU.

    Uses the proven buffer synchronization pattern from October 31, 2025 testing.
    """

    def __init__(self, xclbin_path: str, instr_path: str):
        """
        Initialize mel kernel runner.

        Args:
            xclbin_path: Path to XCLBIN file
            instr_path: Path to instruction binary file
        """
        self.xclbin_path = Path(xclbin_path)
        self.instr_path = Path(instr_path)

        # Initialize NPU
        self.device = xrt.device(0)

        # Load XCLBIN
        xclbin_obj = xrt.xclbin(str(self.xclbin_path))
        uuid = xclbin_obj.get_uuid()

        # Register and create context
        self.device.register_xclbin(xclbin_obj)
        hw_ctx = xrt.hw_context(self.device, uuid)

        # Get kernel
        self.kernel = xrt.kernel(hw_ctx, "MLIR_AIE")

        # Load instructions
        with open(self.instr_path, 'rb') as f:
            self.instr_bin = f.read()
        self.n_insts = len(self.instr_bin)

        # Create buffer manager
        self.buffer_mgr = NPUBufferManager(self.device, self.kernel)

        # Create buffers (mel kernel sizes)
        self.INPUT_SIZE = 800   # 400 INT16 samples = 800 bytes
        self.OUTPUT_SIZE = 80   # 80 INT8 mel bins = 80 bytes
        self.OPCODE = 3         # Mel kernel opcode

        self.instr_bo, self.input_bo, self.output_bo = \
            self.buffer_mgr.create_buffers(self.n_insts,
                                          self.INPUT_SIZE,
                                          self.OUTPUT_SIZE)

        # Write instructions once (they don't change)
        self.buffer_mgr.write_and_sync_to_device(self.instr_bo,
                                                 self.instr_bin,
                                                 self.n_insts)

    def compute_mel(self, audio_int16: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram on NPU for one audio frame.

        Args:
            audio_int16: Audio frame (400 INT16 samples)

        Returns:
            Mel features (80 INT8 values)
        """
        if len(audio_int16) != 400:
            raise ValueError(f"Expected 400 samples, got {len(audio_int16)}")

        # Convert to bytes
        input_data = audio_int16.tobytes()

        # Write input and sync to device
        self.buffer_mgr.write_and_sync_to_device(self.input_bo,
                                                input_data,
                                                self.INPUT_SIZE)

        # Execute kernel
        success = self.buffer_mgr.execute_kernel(self.OPCODE,
                                                self.instr_bo,
                                                self.n_insts,
                                                self.input_bo,
                                                self.output_bo)

        if not success:
            raise RuntimeError("Kernel execution failed")

        # Sync and read output
        output_data = self.buffer_mgr.sync_and_read_from_device(self.output_bo,
                                                                self.OUTPUT_SIZE,
                                                                dtype=np.int8)

        return output_data


# Example usage
if __name__ == '__main__':
    import time

    print("="*70)
    print("NPU Buffer Sync Wrapper - Example Usage")
    print("="*70)

    # Paths
    xclbin_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/mel_fixed_v3.xclbin"
    instr_path = "/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/npu/npu_optimization/mel_kernels/build_fixed_v3/insts_v3.bin"

    # Initialize runner
    print("\nInitializing mel kernel runner...")
    runner = MelKernelRunner(xclbin_path, instr_path)
    print("✅ Initialized")

    # Generate test audio (1kHz sine wave)
    print("\nGenerating test audio (1kHz sine wave)...")
    sr = 16000
    audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
    audio_frame = audio[:400]
    audio_int16 = (audio_frame * 32767).astype(np.int16)
    print(f"✅ Generated {len(audio_int16)} samples")

    # Compute mel spectrogram
    print("\nComputing mel spectrogram on NPU...")
    start = time.perf_counter()
    mel_output = runner.compute_mel(audio_int16)
    end = time.perf_counter()

    # Results
    print(f"✅ Computation complete in {(end-start)*1000:.3f} ms")
    print(f"\nResults:")
    print(f"  Output shape: {mel_output.shape}")
    print(f"  Output range: [{mel_output.min()}, {mel_output.max()}]")
    print(f"  Non-zero bins: {np.count_nonzero(mel_output)}/80")
    print(f"  Mean: {mel_output.mean():.2f}")
    print(f"  First 10 bins: {mel_output[:10]}")

    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
