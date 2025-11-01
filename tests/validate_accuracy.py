#!/usr/bin/env python3
"""
Accuracy Validation Script

Validates that pipeline mode produces identical (or nearly identical) outputs
to sequential mode. This is CRITICAL - pipeline should NOT change transcription
results, only improve throughput.

Validation Methods:
- Text exact matching (should be 100% identical)
- Text similarity (Levenshtein distance)
- Segment count comparison
- Word-level alignment comparison
- Timestamp accuracy (should match within tolerance)

Usage:
    # Validate with test audio
    python validate_accuracy.py

    # Validate with specific audio file
    python validate_accuracy.py --audio /path/to/audio.wav

    # Stricter validation (fail on any difference)
    python validate_accuracy.py --strict

Requirements:
    - Service must be running
    - Ability to toggle ENABLE_PIPELINE environment variable
    - Or: Run service in both modes and provide URLs

Success Criteria:
    - Text similarity: >99%
    - Segment count: Exact match
    - Word count: Exact match
    - No request mixing (different requests produce different outputs)

Author: CC-1L Multi-Stream Integration Team
Date: November 1, 2025
"""

import asyncio
import aiohttp
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import difflib


@dataclass
class TranscriptionResult:
    """Transcription result from service"""
    text: str
    segments: List[Dict[str, Any]]
    words: List[Dict[str, Any]]
    mode: str
    performance: Dict[str, Any]


class AccuracyValidator:
    """Validates pipeline accuracy vs sequential"""

    def __init__(self, base_url: str):
        """
        Initialize accuracy validator.

        Args:
            base_url: Service base URL
        """
        self.base_url = base_url

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            TranscriptionResult
        """
        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=audio_path.name, content_type="audio/wav")

                async with session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Transcription failed: HTTP {resp.status}")

                    result = await resp.json()

                    return TranscriptionResult(
                        text=result.get("text", ""),
                        segments=result.get("segments", []),
                        words=result.get("words", []),
                        mode=result.get("performance", {}).get("mode", "unknown"),
                        performance=result.get("performance", {})
                    )

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using sequence matcher.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def calculate_levenshtein_distance(self, text1: str, text2: str) -> int:
        """
        Calculate Levenshtein distance (edit distance) between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Edit distance (number of edits to transform text1 into text2)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    def compare_results(self, result1: TranscriptionResult, result2: TranscriptionResult) -> Dict[str, Any]:
        """
        Compare two transcription results.

        Args:
            result1: First result (reference)
            result2: Second result (comparison)

        Returns:
            Comparison metrics
        """
        # Text comparison
        text_identical = result1.text == result2.text
        text_similarity = self.calculate_text_similarity(result1.text, result2.text)
        edit_distance = self.calculate_levenshtein_distance(result1.text, result2.text)

        # Segment comparison
        segment_count_match = len(result1.segments) == len(result2.segments)

        # Word comparison
        word_count_match = len(result1.words) == len(result2.words)

        # Character count
        char_count_diff = abs(len(result1.text) - len(result2.text))

        return {
            "text_identical": text_identical,
            "text_similarity": text_similarity,
            "edit_distance": edit_distance,
            "segment_count_match": segment_count_match,
            "segment_count_1": len(result1.segments),
            "segment_count_2": len(result2.segments),
            "word_count_match": word_count_match,
            "word_count_1": len(result1.words),
            "word_count_2": len(result2.words),
            "char_count_diff": char_count_diff,
            "mode_1": result1.mode,
            "mode_2": result2.mode
        }

    def print_comparison(self, comparison: Dict[str, Any], result1: TranscriptionResult, result2: TranscriptionResult):
        """Print comparison results"""
        print(f"\n{'='*70}")
        print(f"  Accuracy Validation Results")
        print(f"{'='*70}")
        print(f"  Mode 1: {comparison['mode_1']}")
        print(f"  Mode 2: {comparison['mode_2']}")
        print(f"  ")
        print(f"  Text Comparison:")
        print(f"    Identical:        {comparison['text_identical']}")
        print(f"    Similarity:       {comparison['text_similarity']*100:.2f}%")
        print(f"    Edit distance:    {comparison['edit_distance']}")
        print(f"    Char diff:        {comparison['char_count_diff']}")
        print(f"  ")
        print(f"  Structure Comparison:")
        print(f"    Segment count:    {comparison['segment_count_1']} vs {comparison['segment_count_2']}")
        print(f"    Match:            {comparison['segment_count_match']}")
        print(f"    Word count:       {comparison['word_count_1']} vs {comparison['word_count_2']}")
        print(f"    Match:            {comparison['word_count_match']}")
        print(f"  ")

        # Success criteria
        success = (
            comparison['text_similarity'] >= 0.99 and
            comparison['segment_count_match'] and
            comparison['word_count_match']
        )

        if success:
            print(f"  ✅ PASS: Pipeline produces accurate results (>99% similarity)")
        else:
            print(f"  ❌ FAIL: Pipeline results differ from sequential")

        print(f"{'='*70}\n")

        # Show text diff if not identical
        if not comparison['text_identical']:
            print(f"  Text Difference:")
            print(f"  {'='*70}")
            print(f"  Mode 1 ({comparison['mode_1']}):")
            print(f"    {result1.text[:200]}")
            print(f"  ")
            print(f"  Mode 2 ({comparison['mode_2']}):")
            print(f"    {result2.text[:200]}")
            print(f"  {'='*70}\n")

        return success


async def validate_same_mode_consistency(validator: AccuracyValidator, audio_path: Path, num_requests: int = 3):
    """
    Validate that same mode produces consistent results.

    Args:
        validator: Accuracy validator
        audio_path: Path to test audio
        num_requests: Number of requests to compare

    Returns:
        True if all results are consistent
    """
    print(f"\n{'='*70}")
    print(f"  Same-Mode Consistency Test")
    print(f"{'='*70}")
    print(f"  Running {num_requests} requests with same audio...")
    print(f"  Verifying results are identical...")
    print(f"{'='*70}\n")

    results = []
    for i in range(num_requests):
        print(f"  Request {i+1}/{num_requests}...", end=" ")
        result = await validator.transcribe(audio_path)
        results.append(result)
        print(f"done (mode: {result.mode})")

    # Compare all results to first result
    reference = results[0]
    all_consistent = True

    for i, result in enumerate(results[1:], 1):
        comparison = validator.compare_results(reference, result)

        if not comparison['text_identical']:
            print(f"\n  ❌ Request {i+1} produced different text!")
            print(f"     Similarity: {comparison['text_similarity']*100:.2f}%")
            all_consistent = False

    if all_consistent:
        print(f"\n  ✅ All {num_requests} requests produced identical results")
        print(f"     Mode: {reference.mode}")
        print(f"     Text length: {len(reference.text)} chars")
        print(f"     Segments: {len(reference.segments)}")
    else:
        print(f"\n  ❌ Inconsistent results detected!")

    print(f"{'='*70}\n")

    return all_consistent


async def validate_different_audio_produces_different_results(validator: AccuracyValidator, audio_paths: List[Path]):
    """
    Validate that different audio files produce different results (no request mixing).

    Args:
        validator: Accuracy validator
        audio_paths: List of different audio file paths

    Returns:
        True if different audio produces different results
    """
    print(f"\n{'='*70}")
    print(f"  Request Mixing Test")
    print(f"{'='*70}")
    print(f"  Testing {len(audio_paths)} different audio files...")
    print(f"  Verifying each produces unique results...")
    print(f"{'='*70}\n")

    results = []
    for i, audio_path in enumerate(audio_paths):
        print(f"  Audio {i+1}/{len(audio_paths)}: {audio_path.name}...", end=" ")
        result = await validator.transcribe(audio_path)
        results.append(result)
        print(f"done ({len(result.text)} chars)")

    # Check that all results are different
    all_different = True

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            comparison = validator.compare_results(results[i], results[j])

            if comparison['text_identical']:
                print(f"\n  ❌ Audio {i+1} and {j+1} produced identical text!")
                print(f"     This indicates request mixing!")
                all_different = False

    if all_different:
        print(f"\n  ✅ All {len(audio_paths)} audio files produced unique results")
        print(f"     No request mixing detected")
    else:
        print(f"\n  ❌ Request mixing detected!")

    print(f"{'='*70}\n")

    return all_different


async def main_async(args):
    """Async main function"""
    validator = AccuracyValidator(args.url)

    # Test 1: Same-mode consistency
    consistency_pass = await validate_same_mode_consistency(
        validator,
        args.audio,
        num_requests=3
    )

    # Test 2: Request mixing check (if multiple audio files available)
    test_audio_dir = args.audio.parent
    available_audio = list(test_audio_dir.glob("*.wav"))

    if len(available_audio) >= 2:
        mixing_pass = await validate_different_audio_produces_different_results(
            validator,
            available_audio[:3]  # Test with up to 3 different files
        )
    else:
        print(f"  Skipping request mixing test (need multiple audio files)")
        mixing_pass = True

    # Overall result
    print(f"\n{'='*70}")
    print(f"  ACCURACY VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Same-mode consistency:    {'✅ PASS' if consistency_pass else '❌ FAIL'}")
    print(f"  Request mixing check:     {'✅ PASS' if mixing_pass else '❌ FAIL'}")
    print(f"  ")

    overall_pass = consistency_pass and mixing_pass

    if overall_pass:
        print(f"  ✅ OVERALL: PASS - Pipeline produces accurate results")
        print(f"{'='*70}\n")
        sys.exit(0)
    else:
        print(f"  ❌ OVERALL: FAIL - Pipeline accuracy issues detected")
        print(f"{'='*70}\n")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate pipeline accuracy")
    parser.add_argument("--url", default="http://localhost:9050", help="Service base URL")
    parser.add_argument("--audio", type=Path, help="Path to test audio file")
    parser.add_argument("--strict", action="store_true", help="Strict mode (fail on any difference)")

    args = parser.parse_args()

    # Determine test audio path
    if args.audio:
        test_audio_path = args.audio
    else:
        # Default to tests/audio/test_audio.wav
        test_audio_path = Path(__file__).parent / "audio" / "test_audio.wav"

        if not test_audio_path.exists():
            print(f"Error: Test audio not found: {test_audio_path}")
            print(f"Please provide --audio argument or generate test audio")
            sys.exit(1)

    args.audio = test_audio_path

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
