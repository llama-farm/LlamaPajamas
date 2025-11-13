#!/usr/bin/env python3
"""Test script to validate domain calibration data and synthetic generation.

This script tests:
1. Built-in domain calibration data loading
2. Domain calibration data export
3. Synthetic generation workflow (without actually calling APIs)
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_pajamas_quant.calibration import (
    TOOL_CALLING_CALIBRATION,
    SUMMARIZATION_CALIBRATION,
    RAG_CALIBRATION,
    MILITARY_CALIBRATION,
    MEDICAL_CALIBRATION,
    TONE_ANALYSIS_CALIBRATION,
    get_tool_calling_calibration_text,
    get_summarization_calibration_text,
    get_rag_calibration_text,
    get_military_calibration_text,
    get_medical_calibration_text,
    get_tone_analysis_calibration_text,
    get_military_seed_examples,
    get_medical_seed_examples,
    get_tone_analysis_seed_examples,
    get_military_domain_description,
    get_medical_domain_description,
    get_tone_analysis_domain_description,
)


def test_calibration_data():
    """Test that all calibration data is loaded correctly."""
    print("=" * 70)
    print("Testing Domain Calibration Data")
    print("=" * 70)

    domains = [
        ("Tool Calling", TOOL_CALLING_CALIBRATION, get_tool_calling_calibration_text),
        ("Summarization", SUMMARIZATION_CALIBRATION, get_summarization_calibration_text),
        ("RAG", RAG_CALIBRATION, get_rag_calibration_text),
        ("Military", MILITARY_CALIBRATION, get_military_calibration_text),
        ("Medical", MEDICAL_CALIBRATION, get_medical_calibration_text),
        ("Tone Analysis", TONE_ANALYSIS_CALIBRATION, get_tone_analysis_calibration_text),
    ]

    all_passed = True

    for name, calibration_list, getter_func in domains:
        print(f"\nTesting {name}:")
        try:
            # Test list is not empty
            assert len(calibration_list) > 0, f"{name} calibration list is empty"
            print(f"  ✓ Calibration list has {len(calibration_list)} samples")

            # Test getter function returns text
            text = getter_func()
            assert isinstance(text, str), f"{name} getter did not return string"
            assert len(text) > 0, f"{name} getter returned empty string"
            print(f"  ✓ Getter function returns text ({len(text)} chars)")

            # Test that text contains samples
            lines = text.strip().split('\n')
            assert len(lines) > 0, f"{name} text has no lines"
            print(f"  ✓ Text has {len(lines)} lines")

            # Test that each sample is a string
            for i, sample in enumerate(calibration_list[:3]):  # Check first 3
                assert isinstance(sample, str), f"{name} sample {i} is not a string"
                assert len(sample) > 0, f"{name} sample {i} is empty"
            print(f"  ✓ Samples are valid strings")

        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def test_seed_examples():
    """Test that seed examples and domain descriptions work."""
    print("\n" + "=" * 70)
    print("Testing Seed Examples and Domain Descriptions")
    print("=" * 70)

    domains = [
        ("Military", get_military_seed_examples, get_military_domain_description),
        ("Medical", get_medical_seed_examples, get_medical_domain_description),
        ("Tone Analysis", get_tone_analysis_seed_examples, get_tone_analysis_domain_description),
    ]

    all_passed = True

    for name, examples_func, description_func in domains:
        print(f"\nTesting {name}:")
        try:
            # Test seed examples
            examples = examples_func()
            assert isinstance(examples, list), f"{name} seed examples not a list"
            assert len(examples) >= 3, f"{name} needs at least 3 seed examples"
            assert len(examples) <= 10, f"{name} should have at most 10 seed examples"
            print(f"  ✓ Seed examples: {len(examples)} samples")

            # Test domain description
            description = description_func()
            assert isinstance(description, str), f"{name} description not a string"
            assert len(description) > 50, f"{name} description too short"
            print(f"  ✓ Domain description: {len(description)} chars")

            # Test examples are valid strings
            for i, ex in enumerate(examples[:3]):
                assert isinstance(ex, str), f"{name} example {i} not a string"
                assert len(ex) > 0, f"{name} example {i} is empty"
            print(f"  ✓ Examples are valid strings")

        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def test_domain_coverage():
    """Test that domains cover different use cases appropriately."""
    print("\n" + "=" * 70)
    print("Testing Domain Coverage and Diversity")
    print("=" * 70)

    # Test total calibration data size
    total_samples = (
        len(TOOL_CALLING_CALIBRATION) +
        len(SUMMARIZATION_CALIBRATION) +
        len(RAG_CALIBRATION) +
        len(MILITARY_CALIBRATION) +
        len(MEDICAL_CALIBRATION) +
        len(TONE_ANALYSIS_CALIBRATION)
    )

    print(f"\nTotal calibration samples across all domains: {total_samples}")

    # Test minimum samples per domain
    min_samples = 15
    domains = {
        "Tool Calling": TOOL_CALLING_CALIBRATION,
        "Summarization": SUMMARIZATION_CALIBRATION,
        "RAG": RAG_CALIBRATION,
        "Military": MILITARY_CALIBRATION,
        "Medical": MEDICAL_CALIBRATION,
        "Tone Analysis": TONE_ANALYSIS_CALIBRATION,
    }

    all_passed = True

    for name, calibration in domains.items():
        count = len(calibration)
        if count >= min_samples:
            print(f"  ✓ {name}: {count} samples (>= {min_samples})")
        else:
            print(f"  ✗ {name}: {count} samples (< {min_samples})")
            all_passed = False

    return all_passed


def test_synthetic_generator_structure():
    """Test that synthetic generator can be imported and instantiated."""
    print("\n" + "=" * 70)
    print("Testing Synthetic Generator Structure")
    print("=" * 70)

    try:
        from llama_pajamas_quant.calibration.synthetic_generator import (
            SyntheticCalibrationGenerator,
            generate_domain_calibration,
        )
        print("  ✓ Synthetic generator module imported successfully")

        # Test that we can inspect the class
        assert hasattr(SyntheticCalibrationGenerator, 'generate_calibration_data')
        print("  ✓ SyntheticCalibrationGenerator has generate_calibration_data method")

        assert hasattr(SyntheticCalibrationGenerator, 'save_calibration_data')
        print("  ✓ SyntheticCalibrationGenerator has save_calibration_data method")

        # Test that helper function exists
        assert callable(generate_domain_calibration)
        print("  ✓ generate_domain_calibration function is callable")

        return True

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DOMAIN CALIBRATION VALIDATION TEST SUITE")
    print("=" * 70 + "\n")

    results = {
        "Calibration Data": test_calibration_data(),
        "Seed Examples": test_seed_examples(),
        "Domain Coverage": test_domain_coverage(),
        "Synthetic Generator": test_synthetic_generator_structure(),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!\n")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
