"""Medical domain calibration data for GGUF/IQ quantization.

This module contains calibration prompts focused on medical, healthcare, and clinical
applications to optimize quantization for medical-specific use cases.
"""

from pathlib import Path
from typing import List


# Seed examples for medical domain
MEDICAL_CALIBRATION = [
    "A 45-year-old patient presents with acute chest pain radiating to the left arm, shortness of breath, and diaphoresis. Describe the differential diagnosis, immediate interventions, and diagnostic tests you would order to rule out myocardial infarction.",

    "Explain the pathophysiology of Type 2 diabetes mellitus, including insulin resistance, beta-cell dysfunction, and the role of inflammatory markers. Discuss evidence-based treatment approaches including lifestyle modifications and pharmacological interventions.",

    "A patient with chronic kidney disease stage 4 is starting dialysis. Explain the differences between hemodialysis and peritoneal dialysis, including mechanisms, advantages, disadvantages, and factors that would influence the choice of modality.",

    "Describe the mechanism of action, clinical indications, common side effects, and monitoring requirements for direct oral anticoagulants (DOACs) such as apixaban and rivaroxaban in the management of atrial fibrillation.",

    "A 28-year-old pregnant woman in her second trimester presents with severe nausea, vomiting, and dehydration. Discuss the differential diagnosis including hyperemesis gravidarum, and outline a safe treatment plan considering the pregnancy.",

    "Explain the stages of wound healing, factors that impair healing (diabetes, infection, malnutrition), and evidence-based interventions to promote optimal wound closure in both acute and chronic wounds.",

    "A 65-year-old patient with history of hypertension and hyperlipidemia presents with sudden-onset severe headache, neck stiffness, and photophobia. Describe your approach to diagnosing and managing suspected subarachnoid hemorrhage.",

    "Discuss the pathophysiology, clinical presentation, diagnostic criteria, and treatment options for chronic obstructive pulmonary disease (COPD), including the role of bronchodilators, corticosteroids, and pulmonary rehabilitation.",

    "A child presents with fever, barking cough, and inspiratory stridor. Explain the differential diagnosis of croup versus epiglottitis versus foreign body aspiration, and describe the clinical features that help distinguish between these conditions.",

    "Describe the pharmacological management of acute heart failure, including the mechanisms and clinical applications of loop diuretics, vasodilators, and inotropic agents. Discuss monitoring parameters and potential complications.",

    "Explain the principles of antibiotic stewardship in hospital settings, including strategies to reduce antibiotic resistance, appropriate use of broad-spectrum versus narrow-spectrum agents, and de-escalation protocols.",

    "A patient presents with polyuria, polydipsia, and unintentional weight loss. Describe the diagnostic workup to differentiate between Type 1 and Type 2 diabetes mellitus, including laboratory tests and their interpretation.",

    "Discuss the assessment and management of acute pain versus chronic pain, including the WHO pain ladder, multimodal analgesia approaches, and the role of non-pharmacological interventions.",

    "Explain the clinical features, diagnostic criteria, and treatment protocols for sepsis and septic shock, including the Surviving Sepsis Campaign guidelines and the importance of early recognition and intervention.",

    "A 70-year-old patient with Parkinson's disease is experiencing motor fluctuations and dyskinesias. Describe the pathophysiology of these complications and discuss pharmacological and non-pharmacological management strategies.",

    "Describe the screening, diagnosis, and staging of breast cancer, including the role of mammography, ultrasound, MRI, and biopsy. Discuss treatment options based on tumor characteristics and staging.",

    "Explain the pathophysiology and management of diabetic ketoacidosis (DKA), including fluid resuscitation protocols, insulin therapy, electrolyte monitoring, and identification of precipitating factors.",

    "A patient with rheumatoid arthritis requires disease-modifying antirheumatic drug (DMARD) therapy. Compare and contrast the mechanisms, efficacy, and safety profiles of methotrexate, hydroxychloroquine, and biologic agents.",

    "Discuss the principles of immunization, vaccine types (live attenuated, inactivated, subunit, mRNA), contraindications, and special considerations for immunocompromised patients and pregnant women.",

    "Explain the assessment and management of acute stroke, including the time-sensitive nature of thrombolytic therapy, the role of neuroimaging, and the distinction between ischemic and hemorrhagic stroke.",

    "A patient presents with jaundice, right upper quadrant pain, and fever (Charcot's triad). Describe the differential diagnosis, diagnostic approach, and management of suspected ascending cholangitis.",

    "Discuss the pharmacological management of hypertension, including first-line agents, combination therapy strategies, and special considerations for patients with comorbidities such as diabetes or chronic kidney disease.",

    "Explain the pathophysiology, clinical presentation, and treatment of anaphylaxis, including the mechanism of IgE-mediated reactions, the role of epinephrine, and prevention strategies for at-risk patients.",

    "A patient with chronic liver disease develops ascites and hepatic encephalopathy. Describe the pathophysiology of these complications and outline evidence-based management strategies.",

    "Discuss the principles of mechanical ventilation in acute respiratory distress syndrome (ARDS), including lung-protective ventilation strategies, PEEP titration, and weaning protocols.",
]


def get_medical_calibration_text() -> str:
    """Get medical calibration data as formatted text.

    Returns:
        Newline-separated calibration prompts
    """
    return '\n'.join(MEDICAL_CALIBRATION)


def save_medical_calibration(filepath: Path) -> None:
    """Save medical calibration data to file.

    Args:
        filepath: Path to save the calibration data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(get_medical_calibration_text())

    print(f"Saved {len(MEDICAL_CALIBRATION)} medical calibration samples to {filepath}")


def get_medical_seed_examples() -> List[str]:
    """Get a subset of examples to use as seeds for synthetic generation.

    Returns:
        List of 5-7 representative examples
    """
    # Return a diverse subset for synthetic generation
    return [
        MEDICAL_CALIBRATION[0],  # Acute care / cardiology
        MEDICAL_CALIBRATION[1],  # Chronic disease management
        MEDICAL_CALIBRATION[3],  # Pharmacology
        MEDICAL_CALIBRATION[6],  # Emergency medicine
        MEDICAL_CALIBRATION[13], # Critical care
        MEDICAL_CALIBRATION[15], # Oncology
        MEDICAL_CALIBRATION[21], # Differential diagnosis
    ]


def get_medical_domain_description() -> str:
    """Get domain description for synthetic generation.

    Returns:
        Detailed description of the medical domain purpose
    """
    return """Medical and healthcare applications requiring expertise in:
- Clinical diagnosis and differential diagnosis
- Disease pathophysiology and mechanisms
- Pharmacology and medication management
- Acute and emergency medicine
- Chronic disease management
- Surgical and procedural knowledge
- Patient assessment and examination
- Evidence-based treatment protocols
- Medical terminology and documentation
- Special populations (pediatrics, geriatrics, pregnancy)
- Critical care and intensive care management
- Preventive medicine and screening

The model should provide clinically accurate information suitable for medical
professionals, while maintaining appropriate disclaimers about not replacing
professional medical judgment. Focus on educational and decision-support scenarios."""


# Example usage
if __name__ == "__main__":
    from synthetic_generator import generate_domain_calibration

    # Generate synthetic medical calibration data
    output = generate_domain_calibration(
        domain="medical",
        purpose=get_medical_domain_description(),
        examples=get_medical_seed_examples(),
        output_dir=Path("../../calibration_data"),
        num_samples=200,
    )
    print(f"Generated medical calibration data: {output}")
