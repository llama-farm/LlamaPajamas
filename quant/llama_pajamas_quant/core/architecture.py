"""Architecture detection and metadata for LLM models."""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum


class ArchitectureFamily(Enum):
    """Model architecture families."""
    DENSE_DECODER = "dense_decoder"
    MOE = "moe"
    SPARSE_MOE_ALTERNATING_ATTENTION = "sparse_moe_alternating_attention"
    HYBRID_ATTENTION = "hybrid_attention"
    HYBRID_MAMBA_TRANSFORMER = "hybrid_mamba_transformer"
    UNKNOWN = "unknown"


class AttentionType(Enum):
    """Attention mechanism types."""
    MULTI_HEAD = "mha"  # Multi-Head Attention
    GROUPED_QUERY = "gqa"  # Grouped Query Attention
    MULTI_QUERY = "mqa"  # Multi-Query Attention
    HYBRID = "hybrid"  # Hybrid local/global
    ALTERNATING = "alternating"  # Dense/sparse alternating
    UNKNOWN = "unknown"


@dataclass
class ArchitectureInfo:
    """Comprehensive architecture information for an LLM model.

    This class holds all detected architecture characteristics and provides
    quantization strategy recommendations based on the architecture.
    """

    # Basic model info
    model_id: str
    model_type: str  # e.g., "qwen3", "llama", "mistral"

    # Architecture classification
    family: ArchitectureFamily
    attention_type: AttentionType

    # Model size
    params_total: str  # e.g., "8.2B"
    params_total_num: int  # Numeric value
    params_non_embedding: Optional[int] = None

    # Attention configuration
    num_layers: int = 0
    num_attention_heads: int = 0
    num_query_heads: int = 0
    num_kv_heads: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0

    # MoE configuration (if applicable)
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_active: Optional[int] = None
    moe_layer_interval: Optional[int] = None  # e.g., every N layers

    # Context and special features
    max_position_embeddings: int = 0
    rope_theta: Optional[float] = None
    sliding_window: Optional[int] = None

    # Additional metadata
    vocab_size: int = 0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Raw config for reference
    raw_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate GQA ratio if applicable
        if self.num_query_heads > 0 and self.num_kv_heads > 0:
            if self.num_query_heads == self.num_kv_heads:
                if self.attention_type == AttentionType.UNKNOWN:
                    self.attention_type = AttentionType.MULTI_HEAD
            elif self.num_kv_heads == 1:
                if self.attention_type == AttentionType.UNKNOWN:
                    self.attention_type = AttentionType.MULTI_QUERY
            else:
                if self.attention_type == AttentionType.UNKNOWN:
                    self.attention_type = AttentionType.GROUPED_QUERY

    @property
    def gqa_ratio(self) -> Optional[int]:
        """Calculate GQA ratio (query heads per KV head)."""
        if self.num_query_heads > 0 and self.num_kv_heads > 0:
            return self.num_query_heads // self.num_kv_heads
        return None

    @property
    def is_gqa(self) -> bool:
        """Check if model uses Grouped Query Attention."""
        return self.attention_type == AttentionType.GROUPED_QUERY

    @property
    def is_mqa(self) -> bool:
        """Check if model uses Multi-Query Attention."""
        return self.attention_type == AttentionType.MULTI_QUERY

    def recommend_quantization(self) -> Dict[str, Any]:
        """Recommend quantization strategy based on architecture.

        Returns:
            Dictionary with quantization recommendations including:
            - precision: Target precision level
            - method: Quantization method
            - special_handling: Architecture-specific optimizations
        """
        strategy = {
            "model_id": self.model_id,
            "family": self.family.value,
            "precision": "mixed",  # Default to mixed precision
            "formats": ["gguf", "mlx"],
            "special_handling": []
        }

        # Architecture-specific recommendations
        if self.family == ArchitectureFamily.DENSE_DECODER:
            strategy.update({
                "gguf_method": "Q4_K_M",
                "mlx_config": {
                    "body_bits": 4,
                    "embedding_bits": 6,
                    "output_bits": 6,
                    "group_size": 64
                },
                "target_memory_gb": 2.0,
                "quality_threshold": 0.05  # <5% loss
            })

            if self.is_gqa:
                strategy["special_handling"].append("gqa_kv_cache_optimization")
                strategy["mlx_config"]["kv_cache_bits"] = 4
                strategy["notes"] = f"GQA {self.gqa_ratio}:1 ratio enables aggressive KV cache quantization"

        elif self.family == ArchitectureFamily.MOE:
            strategy.update({
                "gguf_method": "Q4_K_M",
                "mlx_config": {
                    "body_bits": 4,
                    "embedding_bits": 6,
                    "output_bits": 6,
                    "group_size": 64
                },
                "calibration": "expert_balanced",
                "target_memory_gb": 2.5,
                "quality_threshold": 0.06  # <6% loss
            })
            strategy["special_handling"].extend([
                "expert_balanced_calibration",
                "per_expert_precision_allocation",
                "router_preservation"
            ])
            if self.num_experts:
                strategy["moe_config"] = {
                    "num_experts": self.num_experts,
                    "active_experts": self.num_experts_active,
                    "calibration_samples_per_expert": 64
                }

        elif self.family == ArchitectureFamily.SPARSE_MOE_ALTERNATING_ATTENTION:
            strategy.update({
                "gguf_method": "Q4_K_M",
                "mlx_config": {
                    "body_bits": 4,
                    "embedding_bits": 6,
                    "output_bits": 6,
                    "group_size": 64
                },
                "calibration": "expert_balanced_attention_aware",
                "target_memory_gb": 2.5,
                "quality_threshold": 0.06
            })
            strategy["special_handling"].extend([
                "expert_balanced_calibration",
                "attention_pattern_preservation",
                "dense_sparse_layer_differentiation"
            ])

        return strategy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "family": self.family.value,
            "attention_type": self.attention_type.value,
            "params_total": self.params_total,
            "params_total_num": self.params_total_num,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_ratio": self.gqa_ratio,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "num_experts_active": self.num_experts_active,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
        }

    def __repr__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"ArchitectureInfo({self.model_id})",
            f"  Type: {self.model_type}",
            f"  Family: {self.family.value}",
            f"  Parameters: {self.params_total}",
            f"  Attention: {self.attention_type.value}",
        ]

        if self.is_gqa:
            lines.append(f"  GQA Ratio: {self.gqa_ratio}:1 ({self.num_query_heads} query, {self.num_kv_heads} KV)")

        if self.is_moe:
            lines.append(f"  MoE: {self.num_experts} experts, {self.num_experts_active} active")

        lines.append(f"  Layers: {self.num_layers}")
        lines.append(f"  Context: {self.max_position_embeddings}")

        return "\n".join(lines)
