#!/usr/bin/env python3
"""
Mel Spectrogram Data Loader

Utility module for loading and using generated test mel spectrograms.
Provides convenient access to synthetic and real audio mel data.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


class MelDataLoader:
    """Load and manage test mel spectrogram data."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize mel data loader.

        Args:
            data_dir: Directory containing test mel files. If None, uses default location.
        """
        if data_dir is None:
            # Use default location relative to this script
            script_dir = Path(__file__).parent
            data_dir = script_dir / "test_data"

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self._cache = {}

    def load_synthetic(self, batched: bool = False) -> np.ndarray:
        """
        Load synthetic mel spectrogram.

        Args:
            batched: If True, load batched version (1, 80, 3000).
                     If False, load unbatched version (80, 3000).

        Returns:
            Mel spectrogram array
        """
        filename = "test_mel_batched.npy" if batched else "test_mel_synthetic.npy"
        path = self.data_dir / filename

        if filename in self._cache:
            return self._cache[filename].copy()

        if not path.exists():
            raise FileNotFoundError(f"Synthetic mel file not found: {path}")

        data = np.load(path)
        self._cache[filename] = data
        return data.copy()

    def load_real_audio(self, audio_name: str = "jfk") -> Tuple[np.ndarray, Dict]:
        """
        Load real audio mel spectrogram with metadata.

        Args:
            audio_name: Name of audio file (default: "jfk")
                       Can be partial name like "jfk" or full "test_audio_jfk"

        Returns:
            Tuple of (mel_spectrogram, metadata_dict)
        """
        # Find matching file
        npz_files = list(self.data_dir.glob("*.npz"))

        matching_files = [f for f in npz_files if audio_name.lower() in f.name.lower()]

        if not matching_files:
            available = [f.name for f in npz_files]
            raise FileNotFoundError(
                f"No mel file matching '{audio_name}' found. "
                f"Available: {available}"
            )

        path = matching_files[0]

        data = np.load(path, allow_pickle=True)
        mel_spec = data['mel_spec']
        metadata = data['metadata'].item()

        return mel_spec, metadata

    def list_available(self) -> Dict[str, list]:
        """
        List all available test data files.

        Returns:
            Dictionary with 'synthetic' and 'real_audio' file lists
        """
        synthetic_files = list(self.data_dir.glob("test_mel_synthetic*.npy"))
        real_audio_files = list(self.data_dir.glob("test_mel_*.npz"))

        return {
            'synthetic': [f.name for f in synthetic_files],
            'real_audio': [f.name for f in real_audio_files]
        }

    def get_info(self, verbose: bool = True) -> Dict:
        """
        Get information about available test data.

        Args:
            verbose: If True, print information

        Returns:
            Dictionary with data information
        """
        info = {
            'data_directory': str(self.data_dir),
            'synthetic_data': {},
            'real_audio_data': {}
        }

        # Synthetic data info
        synthetic_file = self.data_dir / "test_mel_synthetic.npy"
        if synthetic_file.exists():
            data = np.load(synthetic_file)
            info['synthetic_data'] = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'size_mb': synthetic_file.stat().st_size / (1024*1024)
            }

        batched_file = self.data_dir / "test_mel_batched.npy"
        if batched_file.exists():
            data = np.load(batched_file)
            info['batched_data'] = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'size_mb': batched_file.stat().st_size / (1024*1024)
            }

        # Real audio data info
        for npz_file in self.data_dir.glob("test_mel_*.npz"):
            data = np.load(npz_file, allow_pickle=True)
            mel_spec = data['mel_spec']
            metadata = data['metadata'].item()

            info['real_audio_data'][npz_file.stem] = {
                'shape': mel_spec.shape,
                'dtype': str(mel_spec.dtype),
                'min': float(mel_spec.min()),
                'max': float(mel_spec.max()),
                'mean': float(mel_spec.mean()),
                'std': float(mel_spec.std()),
                'size_mb': npz_file.stat().st_size / (1024*1024),
                'audio_name': metadata.get('audio_name', 'unknown'),
                'duration_sec': float(metadata.get('audio_duration_sec', 0)),
                'sample_rate': metadata.get('sample_rate', 0)
            }

        if verbose:
            self._print_info(info)

        return info

    @staticmethod
    def _print_info(info: Dict) -> None:
        """Print formatted information about test data."""
        print("\n" + "="*70)
        print("TEST MEL SPECTROGRAM DATA INFORMATION")
        print("="*70)

        print(f"\nData Directory: {info['data_directory']}")

        if info['synthetic_data']:
            print("\nSynthetic Data (Unbatched):")
            print(f"  Shape:     {info['synthetic_data']['shape']}")
            print(f"  Dtype:     {info['synthetic_data']['dtype']}")
            print(f"  Range:     [{info['synthetic_data']['min']:.4f}, "
                  f"{info['synthetic_data']['max']:.4f}]")
            print(f"  Mean/Std:  {info['synthetic_data']['mean']:.4f} / "
                  f"{info['synthetic_data']['std']:.4f}")
            print(f"  Size:      {info['synthetic_data']['size_mb']:.2f} MB")

        if 'batched_data' in info:
            print("\nSynthetic Data (Batched):")
            print(f"  Shape:     {info['batched_data']['shape']}")
            print(f"  Dtype:     {info['batched_data']['dtype']}")
            print(f"  Size:      {info['batched_data']['size_mb']:.2f} MB")

        if info['real_audio_data']:
            print("\nReal Audio Data:")
            for name, data_info in info['real_audio_data'].items():
                print(f"  {name}:")
                print(f"    Audio:     {data_info['audio_name']}")
                print(f"    Duration:  {data_info['duration_sec']:.2f} sec")
                print(f"    Sample Rate: {data_info['sample_rate']} Hz")
                print(f"    Shape:     {data_info['shape']}")
                print(f"    Dtype:     {data_info['dtype']}")
                print(f"    Range:     [{data_info['min']:.2f}, {data_info['max']:.2f}]")
                print(f"    Mean/Std:  {data_info['mean']:.2f} / {data_info['std']:.2f}")
                print(f"    Size:      {data_info['size_mb']:.2f} MB")

        print("\n" + "="*70 + "\n")

    def create_batch(
        self,
        data_source: str = 'synthetic',
        batch_size: int = 4
    ) -> np.ndarray:
        """
        Create a batch of mel spectrograms.

        Args:
            data_source: 'synthetic' or 'real_audio'
            batch_size: Number of samples in batch

        Returns:
            Batched mel spectrogram (batch_size, 80, 3000)
        """
        if data_source == 'synthetic':
            single = self.load_synthetic(batched=False)
        elif data_source == 'real_audio':
            single, _ = self.load_real_audio()
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        # Replicate to create batch
        batch = np.repeat(single[np.newaxis, :, :], batch_size, axis=0)

        # Add slight variation to each sample
        for i in range(batch_size):
            if i > 0:
                noise = np.random.randn(*single.shape) * 0.01
                batch[i] = single + noise

        return batch

    def get_tensor(
        self,
        data_type: str = 'synthetic',
        batched: bool = True,
        as_torch: bool = False
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Get test data as tensor.

        Args:
            data_type: 'synthetic' or 'real_audio'
            batched: Include batch dimension
            as_torch: Return as torch tensor (requires pytorch)

        Returns:
            Mel spectrogram as numpy array or torch tensor
        """
        if data_type == 'synthetic':
            data = self.load_synthetic(batched=batched)
        elif data_type == 'real_audio':
            data, _ = self.load_real_audio()
            if batched and len(data.shape) == 2:
                data = data[np.newaxis, :, :]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if as_torch:
            try:
                import torch
                return torch.from_numpy(data)
            except ImportError:
                raise ImportError("PyTorch not available. Install with: pip install torch")

        return data


# Convenience functions
def load_synthetic_mel(batched: bool = False) -> np.ndarray:
    """Quick load of synthetic mel spectrogram."""
    loader = MelDataLoader()
    return loader.load_synthetic(batched=batched)


def load_real_mel(audio_name: str = "jfk") -> Tuple[np.ndarray, Dict]:
    """Quick load of real audio mel spectrogram."""
    loader = MelDataLoader()
    return loader.load_real_audio(audio_name=audio_name)


def list_test_data() -> Dict[str, list]:
    """List all available test data files."""
    loader = MelDataLoader()
    return loader.list_available()


def print_test_data_info() -> None:
    """Print information about available test data."""
    loader = MelDataLoader()
    loader.get_info(verbose=True)


if __name__ == "__main__":
    import sys

    # Test usage
    try:
        loader = MelDataLoader()

        # Print available data
        print("Available test data:")
        available = loader.list_available()
        for data_type, files in available.items():
            print(f"  {data_type}: {files}")

        # Print detailed info
        loader.get_info(verbose=True)

        # Load and display synthetic data
        synthetic = loader.load_synthetic(batched=False)
        print(f"\nLoaded synthetic mel shape: {synthetic.shape}")

        # Load and display batched synthetic data
        batched = loader.load_synthetic(batched=True)
        print(f"Loaded batched mel shape: {batched.shape}")

        # Try to load real audio
        try:
            mel, metadata = loader.load_real_audio(audio_name="jfk")
            print(f"\nLoaded real audio mel shape: {mel.shape}")
            print(f"Audio: {metadata.get('audio_name', 'unknown')}")
            print(f"Duration: {metadata.get('audio_duration_sec', 0):.2f} seconds")
        except FileNotFoundError as e:
            print(f"\nNote: {e}")

        print("\nData loader test successful!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
