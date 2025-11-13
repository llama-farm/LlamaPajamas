"""Military domain calibration data for GGUF/IQ quantization.

This module contains calibration prompts focused on military, defense, and tactical
operations to optimize quantization for military-specific use cases.
"""

from pathlib import Path
from typing import List


# Seed examples for military domain
MILITARY_CALIBRATION = [
    "Analyze the tactical advantages of establishing a forward operating base in mountainous terrain versus urban environments, considering supply lines, defensive positions, and mobility.",

    "Given a convoy of 15 vehicles transporting critical supplies through hostile territory, develop a security protocol that accounts for IED threats, ambush scenarios, and air support coordination.",

    "Describe the key differences between HUMINT, SIGINT, and GEOINT intelligence gathering methods, and explain when each would be most effective in a counter-insurgency operation.",

    "Create a threat assessment for a coastal military installation, considering potential attack vectors from sea, air, and land, including cyber warfare capabilities.",

    "Explain the command structure and roles in a typical infantry platoon, and how this structure adapts during urban combat operations versus open-field engagements.",

    "Analyze the strategic implications of deploying unmanned aerial vehicles (UAVs) for reconnaissance versus manned aircraft, considering operational costs, risk factors, and intelligence gathering capabilities.",

    "Develop a logistics plan for sustaining a battalion-sized element in a remote location for 90 days, including ammunition, food, water, medical supplies, and fuel requirements.",

    "Compare and contrast the tactical doctrine of maneuver warfare versus attrition warfare, providing historical examples and discussing when each approach is most appropriate.",

    "Describe the process of conducting a tactical site exploitation (TSE) after a successful raid, including evidence collection, biometric data gathering, and intelligence analysis.",

    "Explain the principles of military camouflage and concealment in different environments: desert, jungle, arctic, and urban settings.",

    "Analyze the defensive capabilities and vulnerabilities of modern main battle tanks, considering anti-tank guided missiles, improvised explosive devices, and air strikes.",

    "Develop a counter-sniper strategy for protecting a forward operating base in an urban environment, including detection methods, protective measures, and response protocols.",

    "Describe the planning process for a joint military operation involving Army, Navy, Air Force, and Marine Corps elements, including command relationships and coordination requirements.",

    "Explain the concept of 'combat multipliers' in modern warfare, providing examples such as night vision equipment, GPS navigation, and satellite communications.",

    "Analyze the challenges of maintaining operational security (OPSEC) in the age of social media and ubiquitous smartphone usage among military personnel.",

    "Develop a training curriculum for squad leaders focusing on decision-making under stress, tactical communication, and leadership in high-pressure combat scenarios.",

    "Describe the role and capabilities of military special operations forces, comparing the missions and training of units such as Navy SEALs, Army Rangers, and Air Force Pararescue.",

    "Explain the principles of force protection and antiterrorism measures for deployed military units, including perimeter security, access control, and threat detection.",

    "Analyze the strategic and tactical considerations for conducting amphibious assault operations, from initial planning through beach landing and securing the objective.",

    "Develop a risk assessment framework for evaluating mission proposals, considering factors such as probability of success, potential casualties, strategic value, and political implications.",
]


def get_military_calibration_text() -> str:
    """Get military calibration data as formatted text.

    Returns:
        Newline-separated calibration prompts
    """
    return '\n'.join(MILITARY_CALIBRATION)


def save_military_calibration(filepath: Path) -> None:
    """Save military calibration data to file.

    Args:
        filepath: Path to save the calibration data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(get_military_calibration_text())

    print(f"Saved {len(MILITARY_CALIBRATION)} military calibration samples to {filepath}")


def get_military_seed_examples() -> List[str]:
    """Get a subset of examples to use as seeds for synthetic generation.

    Returns:
        List of 5-7 representative examples
    """
    # Return a diverse subset for synthetic generation
    return [
        MILITARY_CALIBRATION[0],  # Tactical analysis
        MILITARY_CALIBRATION[1],  # Operations planning
        MILITARY_CALIBRATION[2],  # Intelligence methods
        MILITARY_CALIBRATION[5],  # Technology comparison
        MILITARY_CALIBRATION[6],  # Logistics planning
        MILITARY_CALIBRATION[12], # Joint operations
        MILITARY_CALIBRATION[19], # Risk assessment
    ]


def get_military_domain_description() -> str:
    """Get domain description for synthetic generation.

    Returns:
        Detailed description of the military domain purpose
    """
    return """Military and defense applications requiring expertise in:
- Tactical planning and operations
- Strategic analysis and decision-making
- Military intelligence (HUMINT, SIGINT, GEOINT)
- Logistics and supply chain management
- Force protection and security protocols
- Combat operations and battlefield tactics
- Joint operations coordination
- Risk assessment and mission planning
- Military technology and equipment
- Command structure and leadership

The model should understand military terminology, doctrine, and procedures while
maintaining appropriate operational security. Focus on training, planning, and
analysis scenarios rather than specific classified information."""


# Example usage
if __name__ == "__main__":
    from synthetic_generator import generate_domain_calibration

    # Generate synthetic military calibration data
    output = generate_domain_calibration(
        domain="military",
        purpose=get_military_domain_description(),
        examples=get_military_seed_examples(),
        output_dir=Path("../../calibration_data"),
        num_samples=200,
    )
    print(f"Generated military calibration data: {output}")
