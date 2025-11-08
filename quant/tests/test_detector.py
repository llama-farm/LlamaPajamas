#!/usr/bin/env python3
"""Test the ArchitectureDetector with Qwen3-8B."""

import json
from llama_pajamas_quant import ArchitectureDetector


def main():
    """Test architecture detection on Qwen3-8B."""
    print("=" * 70)
    print("Testing ArchitectureDetector with Qwen/Qwen3-8B")
    print("=" * 70)
    print()

    # Initialize detector
    detector = ArchitectureDetector()

    # Detect architecture
    print("Loading model config from HuggingFace...")
    arch = detector.detect("Qwen/Qwen3-8B")

    # Print results
    print()
    print(arch)
    print()

    # Print quantization strategy
    print("=" * 70)
    print("Quantization Strategy Recommendation")
    print("=" * 70)
    strategy = arch.recommend_quantization()
    print(json.dumps(strategy, indent=2))
    print()

    # Validate expected values per comprehensive plan
    print("=" * 70)
    print("Validation Against MVP Plan Requirements")
    print("=" * 70)

    validations = [
        ("Model type", arch.model_type, "qwen3"),
        ("Parameters", arch.params_total, "8.2B"),
        ("Attention type", arch.attention_type.value, "gqa"),
        ("Query heads", arch.num_query_heads, 32),
        ("KV heads", arch.num_kv_heads, 8),
        ("GQA ratio", arch.gqa_ratio, 4),
        ("Family", arch.family.value, "dense_decoder"),
    ]

    all_passed = True
    for name, actual, expected in validations:
        status = "✅" if actual == expected else "❌"
        print(f"{status} {name}: {actual} (expected: {expected})")
        if actual != expected:
            all_passed = False

    print()
    if all_passed:
        print("✅ All validations PASSED!")
    else:
        print("❌ Some validations FAILED!")

    # Check strategy recommendations
    print()
    print("=" * 70)
    print("Strategy Validation")
    print("=" * 70)

    checks = [
        ("GQA KV cache optimization", "gqa_kv_cache_optimization" in strategy.get("special_handling", [])),
        ("GGUF method is Q4_K_M", strategy.get("gguf_method") == "Q4_K_M"),
        ("MLX body bits is 4", strategy.get("mlx_config", {}).get("body_bits") == 4),
        ("MLX embedding bits is 6", strategy.get("mlx_config", {}).get("embedding_bits") == 6),
        ("Target memory < 2GB", strategy.get("target_memory_gb", 0) <= 2.0),
        ("Quality threshold < 5%", strategy.get("quality_threshold", 1.0) <= 0.05),
    ]

    all_checks_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_checks_passed = False

    print()
    if all_passed and all_checks_passed:
        print("🎉 ArchitectureDetector is working correctly for Qwen3-8B!")
    else:
        print("⚠️  Some checks failed, review the output above.")


if __name__ == "__main__":
    main()
