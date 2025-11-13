"""Synthetic calibration data generator using cloud LLMs.

This module provides functionality to generate domain-specific calibration data
using cloud language models like Anthropic Claude or OpenAI GPT-4.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import time


class SyntheticCalibrationGenerator:
    """Generate synthetic calibration data using cloud LLMs."""

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
        """Initialize the synthetic data generator.

        Args:
            api_key: API key for the LLM provider (if None, reads from environment)
            provider: LLM provider to use ('anthropic' or 'openai')
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()

        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model = "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model = "gpt-4-turbo-preview"
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it or pass api_key parameter."
                )
            return key
        elif self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Set it or pass api_key parameter."
                )
            return key

    def generate_calibration_data(
        self,
        domain: str,
        purpose: str,
        examples: List[str],
        num_samples: int = 200,
        temperature: float = 0.8,
        batch_size: int = 50,
    ) -> List[str]:
        """Generate calibration data for a specific domain.

        Args:
            domain: Domain name (e.g., 'medical', 'military', 'tone_analysis')
            purpose: Detailed description of the model's purpose
            examples: 3-10 example prompts that represent the target use case
            num_samples: Number of samples to generate (150-300 recommended)
            temperature: Sampling temperature for diversity (0.7-1.0 recommended)
            batch_size: Number of samples to generate per API call

        Returns:
            List of generated calibration prompts
        """
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        print(f"Generating {num_samples} calibration samples for domain '{domain}'...")
        print(f"Using {num_batches} batches of ~{batch_size} samples each")

        for batch_idx in range(num_batches):
            remaining = num_samples - len(all_samples)
            current_batch_size = min(batch_size, remaining)

            print(f"\nBatch {batch_idx + 1}/{num_batches}: Generating {current_batch_size} samples...")

            prompt = self._create_generation_prompt(
                domain, purpose, examples, current_batch_size
            )

            try:
                batch_samples = self._call_llm(prompt, temperature)
                all_samples.extend(batch_samples)
                print(f"  ✓ Generated {len(batch_samples)} samples (total: {len(all_samples)})")

                # Rate limiting
                if batch_idx < num_batches - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"  ✗ Error generating batch {batch_idx + 1}: {e}")
                continue

        print(f"\n✓ Successfully generated {len(all_samples)} total samples")
        return all_samples

    def _create_generation_prompt(
        self,
        domain: str,
        purpose: str,
        examples: List[str],
        num_samples: int,
    ) -> str:
        """Create the prompt for the LLM to generate calibration data."""
        examples_text = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(examples)])

        return f"""You are an expert at creating high-quality calibration data for quantizing language models.

Your task is to generate {num_samples} diverse, high-quality prompts for the following domain:

**Domain**: {domain}
**Purpose**: {purpose}

**Example Prompts** (use these as inspiration, but create NEW diverse prompts):
{examples_text}

**Requirements**:
1. Generate EXACTLY {num_samples} unique prompts
2. Each prompt should be realistic and representative of the domain
3. Vary the complexity, length, and specific focus within the domain
4. Include edge cases and challenging scenarios
5. Each prompt should be 50-500 tokens long
6. Focus on prompts that will activate domain-specific neural weights
7. Ensure diversity: different perspectives, contexts, and subtasks within the domain

**Output Format**:
Return ONLY a JSON array of strings, with each string being one prompt. No additional text or explanation.

Example format:
["prompt 1 here", "prompt 2 here", ..., "prompt {num_samples} here"]

Generate the {num_samples} prompts now:"""

    def _call_llm(self, prompt: str, temperature: float) -> List[str]:
        """Call the LLM API and parse the response."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content

        # Parse JSON response
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            samples = json.loads(content)
            if not isinstance(samples, list):
                raise ValueError("Response is not a list")

            return [str(s).strip() for s in samples if s and str(s).strip()]
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response content: {content[:500]}...")
            return []

    def save_calibration_data(
        self,
        samples: List[str],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save calibration data to file with metadata.

        Args:
            samples: List of calibration prompts
            output_path: Path to save the calibration.txt file
            metadata: Optional metadata dictionary to save alongside
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save calibration data
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))

        print(f"✓ Saved {len(samples)} samples to {output_path}")

        # Save metadata
        if metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved metadata to {metadata_path}")


def generate_domain_calibration(
    domain: str,
    purpose: str,
    examples: List[str],
    output_dir: Path,
    num_samples: int = 200,
    provider: str = "anthropic",
    api_key: Optional[str] = None,
) -> Path:
    """Convenience function to generate and save domain-specific calibration data.

    Args:
        domain: Domain name (e.g., 'medical', 'military')
        purpose: Description of the model's purpose for this domain
        examples: Example prompts (3-10 recommended)
        output_dir: Directory to save calibration data
        num_samples: Number of samples to generate (150-300 recommended)
        provider: LLM provider ('anthropic' or 'openai')
        api_key: Optional API key (reads from environment if not provided)

    Returns:
        Path to the generated calibration file
    """
    generator = SyntheticCalibrationGenerator(api_key=api_key, provider=provider)

    samples = generator.generate_calibration_data(
        domain=domain,
        purpose=purpose,
        examples=examples,
        num_samples=num_samples,
    )

    output_path = Path(output_dir) / f"calibration_{domain}.txt"

    metadata = {
        "domain": domain,
        "purpose": purpose,
        "num_samples": len(samples),
        "num_examples": len(examples),
        "provider": provider,
        "model": generator.model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    generator.save_calibration_data(samples, output_path, metadata)

    return output_path
