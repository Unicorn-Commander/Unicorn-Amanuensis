#!/usr/bin/env python3
"""
BFP16 Migration Helper Script

This script analyzes C++ code for INT8 quantization patterns and suggests
BFP16 replacements. It helps migrate from INT8 to BFP16 quantization by:

1. Finding all INT8 quantization calls
2. Suggesting BFP16 replacements
3. Generating diff preview
4. Optionally applying changes

Usage:
    # Dry run (show changes only)
    python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp

    # Show detailed analysis
    python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --verbose

    # Generate diff file
    python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --output encoder_layer.diff

    # Apply changes (use with caution!)
    python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --apply

Copyright (C) 2025, Magic Unicorn Unconventional Technology & Stuff Inc
Licensed under the Apache License v2.0 with LLVM Exceptions
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Migration:
    """Represents a single migration from INT8 to BFP16"""
    line_number: int
    original_line: str
    suggested_line: str
    reason: str
    confidence: str  # "high", "medium", "low"


class BFP16MigrationAnalyzer:
    """Analyzes C++ code and suggests BFP16 migrations"""

    # Patterns to detect INT8 quantization code
    PATTERNS = {
        # Include patterns
        'include_quantization': r'#include\s+"quantization\.hpp"',

        # Type patterns
        'int8_buffer': r'Eigen::Matrix<int8_t,\s*Eigen::Dynamic,\s*Eigen::Dynamic>',
        'int32_buffer': r'Eigen::Matrix<int32_t,\s*Eigen::Dynamic,\s*Eigen::Dynamic>',
        'int8_ptr': r'const\s+int8_t\s*\*',
        'int32_ptr': r'int32_t\s*\*',

        # Scale patterns
        'scale_declaration': r'float\s+(\w+_scale_)\s*;',
        'scale_assignment': r'(\w+_scale_)\s*=',

        # Quantization function calls
        'compute_scale': r'\.compute_scale\(',
        'quantize_tensor': r'\.quantize_tensor\(',
        'quantize_tensor_with_scale': r'\.quantize_tensor_with_scale\(',
        'dequantize_matmul_output': r'\.dequantize_matmul_output\(',
        'dequantize_tensor': r'\.dequantize_tensor\(',

        # NPU callback patterns
        'npu_callback_int8': r'typedef\s+int\s+\(\*NPUCallback\)\(.*int8_t.*int32_t',

        # run_npu_linear with scale
        'run_npu_linear_with_scale': r'run_npu_linear\([^)]*,\s*\w+_scale_[^)]*\)',
    }

    # Replacement suggestions
    REPLACEMENTS = {
        'include_quantization': (
            r'#include "quantization.hpp"',
            '#include "quantization.hpp"\n#include "bfp16_quantization.hpp"',
            "Add BFP16 quantization header"
        ),

        'int8_buffer': (
            r'Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>',
            'Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>',
            "Change INT8 buffer to uint8_t (BFP16)"
        ),

        'int32_buffer': (
            r'Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>',
            'Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>',
            "Change INT32 output buffer to uint8_t (BFP16)"
        ),

        'int8_ptr': (
            r'const int8_t\*',
            'const uint8_t*',
            "Change INT8 pointer to uint8_t (BFP16)"
        ),

        'int32_ptr': (
            r'int32_t\*',
            'uint8_t*',
            "Change INT32 pointer to uint8_t (BFP16)"
        ),

        'scale_declaration': (
            r'float (\w+_scale_);',
            '// REMOVED: \\1 (BFP16 scales embedded in exponents)',
            "Remove scale declaration"
        ),

        'compute_scale': (
            r'\.compute_scale\(',
            '// REMOVED: compute_scale() (BFP16 uses block exponents)',
            "Remove compute_scale call"
        ),

        'quantize_tensor': (
            r'quantizer\.quantize_tensor\((\w+),\s*(\w+),\s*(\w+)\)',
            'bfp16_quantizer.prepare_for_npu(\\1, \\2)',
            "Replace quantize_tensor with prepare_for_npu"
        ),

        'dequantize_matmul_output': (
            r'quantizer\.dequantize_matmul_output\(([^,]+),\s*([^,]+),\s*[^,]+,\s*[^)]+\)',
            'bfp16_quantizer.read_from_npu(\\1, \\2, M, N)',
            "Replace dequantize_matmul_output with read_from_npu"
        ),
    }

    def __init__(self, file_path: Path, verbose: bool = False):
        self.file_path = file_path
        self.verbose = verbose
        self.lines = []
        self.migrations: List[Migration] = []

    def analyze(self) -> List[Migration]:
        """Analyze the file and generate migration suggestions"""
        # Read file
        with open(self.file_path, 'r') as f:
            self.lines = f.readlines()

        # Analyze each line
        for line_num, line in enumerate(self.lines, start=1):
            self._analyze_line(line_num, line)

        return self.migrations

    def _analyze_line(self, line_num: int, line: str):
        """Analyze a single line for migration patterns"""
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
            return

        # Check each pattern
        for pattern_name, pattern_regex in self.PATTERNS.items():
            if re.search(pattern_regex, line):
                self._suggest_migration(line_num, line, pattern_name)

    def _suggest_migration(self, line_num: int, line: str, pattern_name: str):
        """Suggest a migration for a matched pattern"""
        if pattern_name in self.REPLACEMENTS:
            old_pattern, new_pattern, reason = self.REPLACEMENTS[pattern_name]
            suggested_line = re.sub(old_pattern, new_pattern, line)

            # Determine confidence
            confidence = self._determine_confidence(pattern_name, line)

            migration = Migration(
                line_number=line_num,
                original_line=line.rstrip(),
                suggested_line=suggested_line.rstrip(),
                reason=reason,
                confidence=confidence
            )
            self.migrations.append(migration)

    def _determine_confidence(self, pattern_name: str, line: str) -> str:
        """Determine confidence level for a migration"""
        # High confidence patterns
        high_confidence = [
            'include_quantization',
            'int8_buffer',
            'int8_ptr',
            'scale_declaration',
        ]

        # Medium confidence patterns
        medium_confidence = [
            'int32_buffer',
            'int32_ptr',
            'compute_scale',
            'quantize_tensor',
            'dequantize_matmul_output',
        ]

        if pattern_name in high_confidence:
            return "high"
        elif pattern_name in medium_confidence:
            return "medium"
        else:
            return "low"

    def print_report(self):
        """Print a summary report of migrations"""
        print(f"\n{'='*80}")
        print(f"BFP16 Migration Analysis: {self.file_path}")
        print(f"{'='*80}\n")

        if not self.migrations:
            print("✅ No INT8 quantization patterns found - already migrated or no changes needed.\n")
            return

        print(f"Found {len(self.migrations)} potential migrations:\n")

        # Group by confidence
        by_confidence = {
            "high": [m for m in self.migrations if m.confidence == "high"],
            "medium": [m for m in self.migrations if m.confidence == "medium"],
            "low": [m for m in self.migrations if m.confidence == "low"],
        }

        for confidence in ["high", "medium", "low"]:
            migrations = by_confidence[confidence]
            if not migrations:
                continue

            print(f"\n{confidence.upper()} CONFIDENCE ({len(migrations)} migrations):")
            print("-" * 80)

            for migration in migrations:
                print(f"\nLine {migration.line_number}: {migration.reason}")
                print(f"  BEFORE: {migration.original_line}")
                print(f"  AFTER:  {migration.suggested_line}")

        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY:")
        print(f"  Total migrations: {len(self.migrations)}")
        print(f"  High confidence: {len(by_confidence['high'])}")
        print(f"  Medium confidence: {len(by_confidence['medium'])}")
        print(f"  Low confidence: {len(by_confidence['low'])}")
        print(f"{'='*80}\n")

    def generate_diff(self, output_file: Path = None):
        """Generate a unified diff"""
        import difflib

        # Create modified version
        modified_lines = self.lines.copy()
        for migration in sorted(self.migrations, key=lambda m: m.line_number, reverse=True):
            modified_lines[migration.line_number - 1] = migration.suggested_line + '\n'

        # Generate diff
        diff = difflib.unified_diff(
            self.lines,
            modified_lines,
            fromfile=f"{self.file_path} (original)",
            tofile=f"{self.file_path} (migrated)",
            lineterm=''
        )

        diff_text = '\n'.join(diff)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(diff_text)
            print(f"\n✅ Diff saved to: {output_file}")
        else:
            print("\n" + "="*80)
            print("DIFF PREVIEW:")
            print("="*80)
            print(diff_text)
            print("="*80 + "\n")

        return diff_text

    def apply_migrations(self, backup: bool = True):
        """Apply migrations to the file (with backup)"""
        if not self.migrations:
            print("No migrations to apply.")
            return

        # Create backup
        if backup:
            backup_path = self.file_path.with_suffix(self.file_path.suffix + '.int8_backup')
            import shutil
            shutil.copy2(self.file_path, backup_path)
            print(f"✅ Backup created: {backup_path}")

        # Apply migrations
        modified_lines = self.lines.copy()
        for migration in sorted(self.migrations, key=lambda m: m.line_number, reverse=True):
            modified_lines[migration.line_number - 1] = migration.suggested_line + '\n'

        # Write modified file
        with open(self.file_path, 'w') as f:
            f.writelines(modified_lines)

        print(f"✅ Applied {len(self.migrations)} migrations to: {self.file_path}")
        print("\nNOTE: Please review the changes carefully and run tests!")


def main():
    parser = argparse.ArgumentParser(
        description="BFP16 Migration Helper - Analyze and migrate INT8 to BFP16 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze file (dry run)
  python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp

  # Generate diff file
  python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --output encoder_layer.diff

  # Apply migrations (with backup)
  python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --apply

  # Apply without backup (use with caution!)
  python3 migrate_to_bfp16.py cpp/src/encoder_layer.cpp --apply --no-backup
        """
    )

    parser.add_argument(
        'file',
        type=Path,
        help='C++ source file to analyze'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Write diff to output file'
    )

    parser.add_argument(
        '-a', '--apply',
        action='store_true',
        help='Apply migrations to file (creates backup)'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup when applying (use with caution!)'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.file.exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)

    if not args.file.suffix in ['.cpp', '.hpp', '.h', '.cc']:
        print(f"⚠️  Warning: {args.file} doesn't look like a C++ file")

    # Run analysis
    analyzer = BFP16MigrationAnalyzer(args.file, verbose=args.verbose)
    analyzer.analyze()

    # Print report
    analyzer.print_report()

    # Generate diff if requested
    if args.output or args.verbose:
        analyzer.generate_diff(args.output)

    # Apply migrations if requested
    if args.apply:
        response = input("\n⚠️  Apply migrations? This will modify the file. [y/N] ")
        if response.lower() == 'y':
            analyzer.apply_migrations(backup=not args.no_backup)
        else:
            print("Aborted.")
    else:
        print("💡 Tip: Use --output to save diff, or --apply to apply changes")


if __name__ == "__main__":
    main()
