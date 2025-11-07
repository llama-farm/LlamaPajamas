"""Architecture detector for LLM models from HuggingFace."""

import logging
from pathlib import Path
from typing import Union, Dict, Any
from transformers import AutoConfig

from .architecture import (
    ArchitectureInfo,
    ArchitectureFamily,
    AttentionType,
)

logger = logging.getLogger(__name__)


class ArchitectureDetector:
    """Detect and analyze LLM model architectures."""

    def __init__(self):
        """Initialize the architecture detector."""
        pass

    def detect(self, model_path: Union[str, Path]) -> ArchitectureInfo:
        """Detect architecture from a HuggingFace model.

        Args:
            model_path: HuggingFace model ID or local path

        Returns:
            ArchitectureInfo with detected characteristics

        Raises:
            ValueError: If model cannot be loaded or architecture unrecognized
        """
        # Load config from HuggingFace
        try:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Failed to load model config from {model_path}: {e}")

        logger.info(f"Loaded config for {model_path}: {config.model_type}")

        # Extract basic info
        model_type = getattr(config, "model_type", "unknown")
        model_id = str(model_path)

        # Detect architecture family and characteristics
        family = self._detect_family(config)
        attention_type = self._detect_attention_type(config)

        # Extract configuration details
        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)

        # Attention heads
        num_attention_heads = getattr(config, "num_attention_heads", 0)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # For Qwen models, they use num_attention_heads for queries
        num_query_heads = num_attention_heads
        num_kv_heads = num_key_value_heads

        # MoE detection
        is_moe = self._is_moe(config)
        num_experts = None
        num_experts_active = None

        if is_moe:
            num_experts = getattr(config, "num_experts", None)
            num_experts_active = getattr(config, "num_experts_per_tok", None)
            if num_experts_active is None:
                num_experts_active = getattr(config, "num_experts_active", None)

        # Context and other features
        max_position_embeddings = getattr(config, "max_position_embeddings", 0)
        rope_theta = getattr(config, "rope_theta", None)
        sliding_window = getattr(config, "sliding_window", None)
        vocab_size = getattr(config, "vocab_size", 0)

        # Token IDs
        bos_token_id = getattr(config, "bos_token_id", None)
        eos_token_id = getattr(config, "eos_token_id", None)

        # Calculate total parameters
        params_total_num = self._estimate_parameters(config)
        params_total = self._format_params(params_total_num)

        # Non-embedding parameters (approximate)
        embedding_params = vocab_size * hidden_size
        params_non_embedding = params_total_num - embedding_params

        return ArchitectureInfo(
            model_id=model_id,
            model_type=model_type,
            family=family,
            attention_type=attention_type,
            params_total=params_total,
            params_total_num=params_total_num,
            params_non_embedding=params_non_embedding,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            is_moe=is_moe,
            num_experts=num_experts,
            num_experts_active=num_experts_active,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            raw_config=config.to_dict(),
        )

    def _detect_family(self, config: Any) -> ArchitectureFamily:
        """Detect architecture family from config.

        Args:
            config: HuggingFace model config

        Returns:
            ArchitectureFamily enum value
        """
        model_type = getattr(config, "model_type", "").lower()

        # Check for MoE
        if self._is_moe(config):
            # Check for alternating attention patterns (GPT-OSS style)
            if hasattr(config, "attention_pattern") or \
               hasattr(config, "sparse_attention") or \
               "alternating" in str(getattr(config, "architectures", [])).lower():
                return ArchitectureFamily.SPARSE_MOE_ALTERNATING_ATTENTION
            return ArchitectureFamily.MOE

        # Check for hybrid attention (Gemma 3 style)
        if hasattr(config, "local_attention_ratio") or \
           hasattr(config, "global_attention_ratio") or \
           "hybrid" in model_type:
            return ArchitectureFamily.HYBRID_ATTENTION

        # Check for Mamba-Transformer hybrid
        if "mamba" in model_type or "granite" in model_type:
            return ArchitectureFamily.HYBRID_MAMBA_TRANSFORMER

        # Default to dense decoder-only
        return ArchitectureFamily.DENSE_DECODER

    def _detect_attention_type(self, config: Any) -> AttentionType:
        """Detect attention mechanism type.

        Args:
            config: HuggingFace model config

        Returns:
            AttentionType enum value
        """
        num_attention_heads = getattr(config, "num_attention_heads", 0)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        if num_attention_heads == 0:
            return AttentionType.UNKNOWN

        # Let __post_init__ handle the classification based on head counts
        return AttentionType.UNKNOWN

    def _is_moe(self, config: Any) -> bool:
        """Check if model uses Mixture of Experts.

        Args:
            config: HuggingFace model config

        Returns:
            True if MoE model, False otherwise
        """
        # Check for common MoE indicators
        moe_indicators = [
            "num_experts",
            "num_experts_per_tok",
            "num_local_experts",
            "expert_count",
        ]

        for indicator in moe_indicators:
            if hasattr(config, indicator) and getattr(config, indicator, 0) > 1:
                return True

        # Check architecture name
        arch_names = getattr(config, "architectures", [])
        for arch_name in arch_names:
            if "moe" in arch_name.lower() or "mixtral" in arch_name.lower():
                return True

        return False

    def _estimate_parameters(self, config: Any) -> int:
        """Estimate total model parameters from config.

        Args:
            config: HuggingFace model config

        Returns:
            Estimated parameter count
        """
        hidden_size = getattr(config, "hidden_size", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        vocab_size = getattr(config, "vocab_size", 0)
        num_attention_heads = getattr(config, "num_attention_heads", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        if hidden_size == 0:
            return 0

        # Embedding parameters
        embedding_params = vocab_size * hidden_size

        # Per-layer parameters
        # Attention: Q, K, V projections + output projection
        q_params = hidden_size * hidden_size  # Query projection
        k_params = hidden_size * (hidden_size * num_kv_heads // num_attention_heads)  # Key projection
        v_params = hidden_size * (hidden_size * num_kv_heads // num_attention_heads)  # Value projection
        o_params = hidden_size * hidden_size  # Output projection

        attention_params = q_params + k_params + v_params + o_params

        # FFN: gate, up, down projections (SwiGLU has 3 projections)
        if self._is_moe(config):
            num_experts = getattr(config, "num_experts", 8)
            # MoE FFN
            ffn_params = num_experts * (
                hidden_size * intermediate_size +  # gate
                hidden_size * intermediate_size +  # up
                intermediate_size * hidden_size    # down
            )
            # Router
            router_params = hidden_size * num_experts
            ffn_params += router_params
        else:
            # Standard FFN with SwiGLU (3 projections: gate, up, down)
            ffn_params = (
                hidden_size * intermediate_size +  # gate
                hidden_size * intermediate_size +  # up
                intermediate_size * hidden_size    # down
            )

        # Layer norm parameters (2 per layer, minimal)
        ln_params = 2 * hidden_size

        layer_params = attention_params + ffn_params + ln_params

        # Total
        total_params = embedding_params + (num_layers * layer_params)

        # Final layer norm
        total_params += hidden_size

        # LM head (often tied with embeddings, but count separately)
        total_params += vocab_size * hidden_size

        return total_params

    def _format_params(self, params: int) -> str:
        """Format parameter count as human-readable string.

        Args:
            params: Parameter count

        Returns:
            Formatted string (e.g., "8.2B")
        """
        if params >= 1_000_000_000:
            return f"{params / 1_000_000_000:.1f}B"
        elif params >= 1_000_000:
            return f"{params / 1_000_000:.1f}M"
        else:
            return f"{params:,}"
