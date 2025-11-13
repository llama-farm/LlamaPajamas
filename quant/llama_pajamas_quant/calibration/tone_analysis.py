"""Tone analysis domain calibration data for GGUF/IQ quantization.

This module contains calibration prompts focused on sentiment analysis, emotional
tone detection, and communication style analysis to optimize quantization for
tone-aware applications.
"""

from pathlib import Path
from typing import List


# Seed examples for tone analysis domain
TONE_ANALYSIS_CALIBRATION = [
    "Analyze the emotional tone and sentiment of this customer feedback: 'I've been waiting for 3 hours and still no response from support. This is completely unacceptable and unprofessional. I'm extremely disappointed with your service.'",

    "Identify the underlying emotions and communication style in this message: 'Hey team! Just wanted to say you all did an absolutely amazing job on the presentation today. The client was thrilled and we landed the account! Celebratory lunch on me tomorrow! 🎉'",

    "Assess the tone and intent of this email: 'Per my previous email, I would appreciate a response regarding the outstanding invoice. This is now the third attempt to reach you on this matter. Please advise at your earliest convenience.'",

    "Analyze the emotional subtext and passive-aggressive elements in: 'Thanks so much for your feedback. I'll definitely take it into consideration. It's always nice to hear fresh perspectives from people who just started last week.'",

    "Determine the sentiment, urgency, and professionalism level in: 'URGENT: System outage affecting all users. Need immediate escalation to senior management. Critical production environment down. Revenue impact estimated at $50K/hour.'",

    "Identify the tone, emotion, and relationship dynamic in this text message: 'K. Fine. Whatever you want. Don't worry about it. I'm sure you're too busy anyway.'",

    "Analyze the emotional progression and sentiment shift in this product review: 'Initially I was skeptical about this purchase, but after using it for a week, I have to say I'm genuinely impressed. It exceeded my expectations in every way.'",

    "Assess the tone and emotional intelligence in this manager's response: 'I understand you're frustrated with the project timeline. Let's schedule time tomorrow to discuss your concerns and see how we can adjust the plan to make this more manageable for everyone.'",

    "Identify sarcasm, irony, or genuine sentiment in: 'Oh great, another mandatory meeting at 4:30pm on Friday. Exactly what I needed to make my week complete. So thoughtful of them to schedule it then.'",

    "Analyze the confidence level, assertiveness, and professionalism in: 'I have significant concerns about this approach and would like to propose an alternative solution that addresses the technical limitations we discussed.'",

    "Determine the emotional state and urgency in: 'I can't believe this is happening again. We discussed this exact issue last month and nothing has changed. I'm at my wit's end and don't know what else to do.'",

    "Assess the tone and relationship quality in: 'Just checking in to see how you're doing! No pressure at all, but wanted to let you know I'm here if you need anything. Hope you're having a great day! 😊'",

    "Identify defensive language and emotional barriers in: 'I did exactly what you asked me to do. It's not my fault if the instructions weren't clear. Maybe if someone had communicated better in the first place, we wouldn't be in this situation.'",

    "Analyze gratitude, sincerity, and professionalism in: 'I want to express my deep appreciation for your mentorship over the past year. Your guidance has been invaluable to my professional development and I'm truly grateful for the time you've invested in my growth.'",

    "Determine anxiety, stress, or overwhelm levels in: 'I have five deadlines this week, two of which got moved up unexpectedly. I'm trying to prioritize but honestly feeling pretty overwhelmed. Can we possibly discuss timeline adjustments?'",

    "Assess conflict escalation and emotional temperature in: 'This is the second time you've undermined my decisions in front of the team. I need you to understand that this behavior is not acceptable and we need to address it immediately.'",

    "Identify enthusiasm, engagement, and authenticity in: 'This is such a fascinating project! I love the creative direction we're taking. I've been thinking about it all weekend and have some ideas I'd love to share with the team.'",

    "Analyze formal vs informal tone, respect level, and relationship context in: 'Hey Sarah, got ur msg. Yeah that works. C u then! 👍' versus 'Dear Dr. Johnson, Thank you for your message. That time works well for me. I look forward to our meeting.'",

    "Determine disappointment, blame, or constructive criticism in: 'The results weren't what we hoped for this quarter. While external factors played a role, we also need to examine our internal processes and identify where we can improve.'",

    "Assess empathy, validation, and support in: 'I can hear how difficult this situation has been for you. It's completely understandable to feel overwhelmed. Let's work together to find a solution that feels manageable.'",

    "Identify excitement versus anxiety in anticipatory statements: 'I can't wait for the presentation tomorrow!' versus 'I can't stop thinking about the presentation tomorrow.'",

    "Analyze professional boundaries and appropriate emotional expression in: 'I appreciate your concern, but I'd prefer to keep my personal matters private. Let's focus on the work-related aspects of this discussion.'",

    "Determine resignation, burnout, or disengagement signals in: 'Sure, I'll add it to the list. Along with the other dozen priorities that all need to be done by yesterday. I'll do my best.'",

    "Assess celebration, recognition, and team morale in: 'Congratulations to the entire team! This milestone represents months of hard work and dedication. Each one of you contributed something vital to this success.'",

    "Identify concern, warning, or escalation in: 'I feel obligated to raise a concern about the current trajectory of this project. If we don't address these issues now, we risk significant problems down the line.'",
]


def get_tone_analysis_calibration_text() -> str:
    """Get tone analysis calibration data as formatted text.

    Returns:
        Newline-separated calibration prompts
    """
    return '\n'.join(TONE_ANALYSIS_CALIBRATION)


def save_tone_analysis_calibration(filepath: Path) -> None:
    """Save tone analysis calibration data to file.

    Args:
        filepath: Path to save the calibration data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(get_tone_analysis_calibration_text())

    print(f"Saved {len(TONE_ANALYSIS_CALIBRATION)} tone analysis calibration samples to {filepath}")


def get_tone_analysis_seed_examples() -> List[str]:
    """Get a subset of examples to use as seeds for synthetic generation.

    Returns:
        List of 5-7 representative examples
    """
    # Return a diverse subset for synthetic generation
    return [
        TONE_ANALYSIS_CALIBRATION[0],  # Negative sentiment
        TONE_ANALYSIS_CALIBRATION[1],  # Positive sentiment
        TONE_ANALYSIS_CALIBRATION[3],  # Passive-aggressive
        TONE_ANALYSIS_CALIBRATION[8],  # Sarcasm detection
        TONE_ANALYSIS_CALIBRATION[14], # Stress/anxiety
        TONE_ANALYSIS_CALIBRATION[19], # Empathy/support
        TONE_ANALYSIS_CALIBRATION[20], # Emotional nuance
    ]


def get_tone_analysis_domain_description() -> str:
    """Get domain description for synthetic generation.

    Returns:
        Detailed description of the tone analysis domain purpose
    """
    return """Tone and sentiment analysis applications requiring expertise in:
- Emotional tone detection (happy, sad, angry, anxious, excited, etc.)
- Sentiment classification (positive, negative, neutral, mixed)
- Communication style analysis (formal, informal, professional, casual)
- Sarcasm and irony detection
- Passive-aggressive language identification
- Urgency and priority assessment
- Empathy and emotional intelligence recognition
- Conflict escalation detection
- Sincerity and authenticity assessment
- Emotional subtext and implicit meaning
- Relationship dynamics and power structures
- Stress, burnout, and overwhelm signals
- Professional boundaries and appropriateness
- Customer satisfaction and feedback analysis
- Team morale and engagement indicators

The model should accurately detect subtle emotional cues, implicit meanings,
and contextual tone variations across different communication contexts (email,
chat, reviews, social media, professional vs personal, etc.)."""


# Example usage
if __name__ == "__main__":
    from synthetic_generator import generate_domain_calibration

    # Generate synthetic tone analysis calibration data
    output = generate_domain_calibration(
        domain="tone_analysis",
        purpose=get_tone_analysis_domain_description(),
        examples=get_tone_analysis_seed_examples(),
        output_dir=Path("../../calibration_data"),
        num_samples=200,
    )
    print(f"Generated tone analysis calibration data: {output}")
