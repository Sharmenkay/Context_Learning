"""
Synthetic Data Generator for Three-Curriculum Context-Enhanced Learning (3CEL).

Generates realistic clinical nursing scenarios to demonstrate the 3CEL architecture.
Each sample contains:
- cS (Static): Clinical protocols and guidelines
- cC (Case): Patient-specific context
- cU (User): Nurse practitioner profile
- x: Clinical question/scenario
- y: Expected clinical response/action

Usage:
    python generate_synthetic_data.py --output_dir ./data --train_samples 5000 --eval_samples 500
"""

import json
import pickle
import random
import argparse
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# Clinical Knowledge Base for Synthetic Generation
# ============================================================================

PROTOCOLS = {
    "medication_admin": [
        "Protocol: Verify the 5 Rights before medication administration - Right patient, Right drug, Right dose, Right route, Right time.",
        "Protocol: Double-check high-alert medications with a second nurse before administration.",
        "Protocol: Document medication administration within 30 minutes of giving the medication.",
        "Protocol: Hold medication and notify physician if vital signs are outside normal parameters.",
        "Protocol: Assess for allergies before administering any new medication.",
    ],
    "vital_signs": [
        "Guideline: Monitor vital signs every 4 hours for stable patients, every 1-2 hours for unstable patients.",
        "Guideline: Report immediately: BP >180/110 or <90/60, HR >120 or <50, RR >24 or <10, SpO2 <92%, Temp >38.5°C.",
        "Guideline: Document vital signs trends and notify physician of significant changes.",
        "Protocol: Use appropriate cuff size for accurate blood pressure measurement.",
        "Guideline: Assess pain level (0-10 scale) with each vital signs check.",
    ],
    "infection_control": [
        "Protocol: Perform hand hygiene before and after patient contact using soap/water or alcohol-based sanitizer.",
        "Protocol: Use appropriate PPE based on isolation precautions - contact, droplet, or airborne.",
        "Guideline: Change IV dressings every 72 hours or when soiled, loose, or damp.",
        "Protocol: Monitor surgical sites for signs of infection: redness, swelling, warmth, drainage, odor.",
        "Guideline: Obtain cultures before initiating antibiotic therapy when possible.",
    ],
    "fall_prevention": [
        "Protocol: Assess fall risk on admission and every shift using standardized fall risk assessment tool.",
        "Guideline: Implement fall precautions for high-risk patients: bed alarm, non-slip socks, call light within reach.",
        "Protocol: Assist high fall-risk patients with ambulation and toileting.",
        "Guideline: Keep bed in lowest position with brakes locked at all times.",
        "Protocol: Document all falls and complete incident report within 24 hours.",
    ],
    "pain_management": [
        "Protocol: Assess pain using appropriate scale (numeric, FACES, or behavioral) based on patient ability.",
        "Guideline: Reassess pain 30-60 minutes after intervention to evaluate effectiveness.",
        "Protocol: Document pain location, quality, intensity, duration, and aggravating/relieving factors.",
        "Guideline: Use multimodal pain management approach combining pharmacological and non-pharmacological methods.",
        "Protocol: Monitor for adverse effects of opioid medications: sedation, respiratory depression, constipation.",
    ],
    "patient_safety": [
        "Protocol: Verify patient identity using two identifiers before any procedure or medication.",
        "Guideline: Perform bedside shift report including patient in the handoff communication.",
        "Protocol: Use SBAR format for communication with physicians: Situation, Background, Assessment, Recommendation.",
        "Guideline: Document patient education and verify understanding using teach-back method.",
        "Protocol: Activate rapid response team for acute patient deterioration.",
    ],
    "wound_care": [
        "Protocol: Assess wounds for size, depth, tissue type, exudate, and surrounding skin condition.",
        "Guideline: Use aseptic technique for wound dressing changes.",
        "Protocol: Document wound measurements and characteristics with each dressing change.",
        "Guideline: Select appropriate wound dressing based on wound type and healing stage.",
        "Protocol: Monitor for signs of wound infection and report promptly.",
    ],
    "respiratory_care": [
        "Protocol: Assess respiratory status including rate, depth, pattern, and breath sounds.",
        "Guideline: Encourage incentive spirometry every 1-2 hours while awake for post-operative patients.",
        "Protocol: Position patient with head of bed elevated 30-45 degrees to optimize breathing.",
        "Guideline: Suction airway as needed based on assessment, not on a routine schedule.",
        "Protocol: Monitor oxygen saturation continuously for patients on supplemental oxygen.",
    ],
}

PATIENT_DEMOGRAPHICS = {
    "age_groups": ["32-year-old", "45-year-old", "58-year-old", "67-year-old", "72-year-old", "81-year-old", "89-year-old"],
    "genders": ["male", "female"],
    "admission_reasons": [
        "pneumonia", "COPD exacerbation", "heart failure exacerbation", "post-hip replacement surgery",
        "post-CABG surgery", "diabetic ketoacidosis", "acute kidney injury", "sepsis",
        "stroke", "GI bleeding", "chest pain observation", "syncope evaluation",
        "post-appendectomy", "cellulitis", "UTI with sepsis", "post-knee replacement surgery"
    ],
}

PATIENT_CONDITIONS = {
    "comorbidities": [
        "Type 2 diabetes", "hypertension", "COPD", "heart failure", "chronic kidney disease",
        "atrial fibrillation", "obesity", "hypothyroidism", "coronary artery disease", "dementia"
    ],
    "current_status": [
        "stable condition", "improving", "guarded condition", "critical but stable",
        "post-operative day 1", "post-operative day 2", "post-operative day 3"
    ],
    "recent_events": [
        "uneventful night", "episode of confusion overnight", "complained of increased pain",
        "low-grade fever this morning", "decreased urine output", "refused breakfast",
        "ambulated with assistance", "blood glucose elevated", "oxygen requirement increased"
    ],
}

VITAL_SIGNS_SCENARIOS = [
    "BP 142/88, HR 78, RR 18, SpO2 96% on room air, Temp 37.2°C",
    "BP 168/95, HR 92, RR 20, SpO2 94% on 2L NC, Temp 37.8°C",
    "BP 98/62, HR 110, RR 24, SpO2 91% on 4L NC, Temp 38.4°C",
    "BP 135/82, HR 68, RR 16, SpO2 98% on room air, Temp 36.8°C",
    "BP 155/90, HR 88 irregular, RR 22, SpO2 93% on 3L NC, Temp 37.5°C",
    "BP 118/72, HR 72, RR 14, SpO2 99% on room air, Temp 37.0°C",
]

NURSE_PROFILES = {
    "experience_levels": [
        ("New graduate RN", "6 months", "step-by-step guidance"),
        ("RN", "2 years", "clear rationales"),
        ("RN", "5 years", "efficient communication"),
        ("Senior RN", "10 years", "concise updates"),
        ("Charge RN", "15 years", "brief summaries"),
        ("Clinical nurse specialist", "20 years", "evidence-based approaches"),
    ],
    "workloads": [
        "3 patients (light)", "4 patients (moderate)", "5 patients (heavy)",
        "6 patients (very heavy)", "4 patients including 1 high-acuity"
    ],
    "specialties": [
        "medical-surgical", "ICU", "cardiac", "oncology", "orthopedic",
        "neurology", "emergency", "pediatric"
    ],
}

CLINICAL_QUESTIONS = [
    # Medication-related
    ("Patient's blood pressure is {bp}. They are due for their scheduled antihypertensive. What should you do?",
     "medication_admin", "vital_signs"),
    ("Patient reports pain level of {pain}/10 at surgical site. PRN pain medication was last given 5 hours ago. What action should be taken?",
     "pain_management", "medication_admin"),
    ("You notice the patient's IV site is red and swollen. The current IV antibiotic is due in 1 hour. What should you do?",
     "infection_control", "medication_admin"),

    # Assessment-related
    ("During your assessment, you notice the patient is more confused than at the start of your shift. What should you do?",
     "patient_safety", "vital_signs"),
    ("Patient's oxygen saturation has dropped from 96% to 91% over the past hour. What actions should you take?",
     "respiratory_care", "vital_signs"),
    ("You observe increased drainage from the patient's surgical wound that appears yellowish-green. What should you do?",
     "wound_care", "infection_control"),

    # Safety-related
    ("Patient is attempting to get out of bed unassisted. They are a high fall risk. What should you do?",
     "fall_prevention", "patient_safety"),
    ("Patient's family member is asking about the plan of care and seems frustrated with lack of information. How should you respond?",
     "patient_safety", "vital_signs"),
    ("You are preparing to administer a blood transfusion. What steps must you take before starting?",
     "patient_safety", "medication_admin"),

    # Deterioration-related
    ("Patient suddenly becomes diaphoretic and reports chest pressure. What immediate actions should you take?",
     "patient_safety", "vital_signs"),
    ("Patient's respiratory rate has increased to 28 and they appear to be using accessory muscles. What should you do?",
     "respiratory_care", "patient_safety"),
    ("Patient has not voided in 8 hours and reports lower abdominal discomfort. What assessments and actions are needed?",
     "patient_safety", "vital_signs"),
]

RESPONSE_TEMPLATES = {
    "medication_admin": [
        "1. Verify patient identity using two identifiers. 2. {action} 3. Document assessment findings and actions taken. 4. {follow_up}",
    ],
    "vital_signs": [
        "1. Perform complete vital signs assessment. 2. Compare to baseline and previous readings. 3. {action} 4. Document findings and notify physician if indicated using SBAR format.",
    ],
    "pain_management": [
        "1. Assess pain using numeric scale (location, quality, intensity). 2. {action} 3. Implement non-pharmacological comfort measures. 4. Reassess pain in 30-60 minutes. 5. Document assessment and interventions.",
    ],
    "infection_control": [
        "1. Perform hand hygiene. 2. Assess site thoroughly for signs of infection. 3. {action} 4. Document findings. 5. Monitor for systemic signs of infection.",
    ],
    "fall_prevention": [
        "1. Ensure immediate patient safety. 2. {action} 3. Review and reinforce fall precautions. 4. Document intervention and patient response.",
    ],
    "patient_safety": [
        "1. Ensure patient safety and stay with patient. 2. {action} 3. Communicate findings using SBAR format. 4. Document thoroughly. 5. {follow_up}",
    ],
    "respiratory_care": [
        "1. Assess respiratory status (rate, depth, pattern, breath sounds, SpO2). 2. {action} 3. Position patient to optimize breathing. 4. Notify physician of significant changes. 5. Document assessment and interventions.",
    ],
    "wound_care": [
        "1. Perform hand hygiene and gather supplies. 2. Assess wound characteristics. 3. {action} 4. Apply appropriate dressing using aseptic technique. 5. Document wound assessment and care provided.",
    ],
}


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_patient_context() -> str:
    """Generate a realistic patient case context."""
    age = random.choice(PATIENT_DEMOGRAPHICS["age_groups"])
    gender = random.choice(PATIENT_DEMOGRAPHICS["genders"])
    admission = random.choice(PATIENT_DEMOGRAPHICS["admission_reasons"])
    comorbidities = random.sample(PATIENT_CONDITIONS["comorbidities"], k=random.randint(1, 3))
    status = random.choice(PATIENT_CONDITIONS["current_status"])
    vitals = random.choice(VITAL_SIGNS_SCENARIOS)
    recent = random.choice(PATIENT_CONDITIONS["recent_events"])

    context = f"Patient: {age} {gender}, admitted for {admission}. "
    context += f"Medical history: {', '.join(comorbidities)}. "
    context += f"Current status: {status}. "
    context += f"Recent: {recent}. "
    context += f"Current vitals: {vitals}."

    return context


def generate_nurse_profile() -> str:
    """Generate a nurse practitioner profile."""
    exp = random.choice(NURSE_PROFILES["experience_levels"])
    workload = random.choice(NURSE_PROFILES["workloads"])
    specialty = random.choice(NURSE_PROFILES["specialties"])

    profile = f"Nurse: {exp[0]} with {exp[1]} experience in {specialty} unit. "
    profile += f"Current workload: {workload}. "
    profile += f"Communication preference: {exp[2]}."

    return profile


def generate_clinical_scenario() -> Tuple[str, str, str, str]:
    """Generate a clinical question and appropriate response."""
    question_template, primary_protocol, secondary_protocol = random.choice(CLINICAL_QUESTIONS)

    # Fill in template variables
    question = question_template.format(
        bp=random.choice(["158/94", "172/98", "145/88", "102/68"]),
        pain=random.randint(5, 9)
    )

    # Select relevant protocols
    protocol1 = random.choice(PROTOCOLS[primary_protocol])
    protocol2 = random.choice(PROTOCOLS[secondary_protocol])
    static_curriculum = f"{protocol1}\n{protocol2}"

    # Generate response based on scenario type
    actions = {
        "medication_admin": [
            "Check vital signs before administering medication",
            "Hold medication and reassess patient",
            "Administer medication as ordered after verification",
            "Contact physician for parameter clarification"
        ],
        "vital_signs": [
            "Continue monitoring and trending",
            "Increase monitoring frequency",
            "Notify physician of abnormal findings",
            "Initiate standing orders as appropriate"
        ],
        "pain_management": [
            "Administer PRN pain medication as ordered",
            "Assess for potential complications",
            "Implement positioning and comfort measures",
            "Request pain management consultation"
        ],
        "infection_control": [
            "Discontinue current IV and restart at new site",
            "Obtain cultures as ordered",
            "Apply appropriate wound care",
            "Implement isolation precautions"
        ],
        "fall_prevention": [
            "Assist patient back to bed safely",
            "Activate bed alarm and ensure call light is within reach",
            "Assess for injury and document",
            "Reinforce safety education with patient"
        ],
        "patient_safety": [
            "Stay with patient and call for help",
            "Activate rapid response if indicated",
            "Perform focused assessment",
            "Communicate urgently with healthcare team"
        ],
        "respiratory_care": [
            "Apply or increase supplemental oxygen",
            "Elevate head of bed to 45 degrees",
            "Prepare for possible respiratory support",
            "Notify respiratory therapy"
        ],
        "wound_care": [
            "Clean wound with appropriate solution",
            "Obtain wound culture if infection suspected",
            "Apply prescribed wound treatment",
            "Consult wound care specialist"
        ],
    }

    action = random.choice(actions.get(primary_protocol, actions["patient_safety"]))
    follow_up = random.choice([
        "Continue to monitor and reassess",
        "Follow up with physician for further orders",
        "Ensure documentation is complete",
        "Update care plan as needed"
    ])

    response_template = random.choice(RESPONSE_TEMPLATES.get(primary_protocol, RESPONSE_TEMPLATES["patient_safety"]))
    response = response_template.format(action=action, follow_up=follow_up)

    return static_curriculum, question, response, primary_protocol


def generate_sample() -> Dict[str, str]:
    """Generate a complete training sample with all three curricula."""
    c_static, question, response, _ = generate_clinical_scenario()
    c_case = generate_patient_context()
    c_user = generate_nurse_profile()

    return {
        "c_static": c_static,
        "c_case": c_case,
        "c_user": c_user,
        "x": question,
        "y": response
    }


def generate_dataset(n_samples: int, seed: int = None) -> List[Dict[str, str]]:
    """Generate a dataset of n_samples."""
    if seed is not None:
        random.seed(seed)

    return [generate_sample() for _ in range(n_samples)]


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical data for 3CEL training")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic", help="Output directory")
    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=500, help="Number of evaluation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", type=str, default="both", choices=["json", "pkl", "both"], help="Output format")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Synthetic Clinical Data Generator for 3CEL")
    print("=" * 60)

    # Generate training data
    print(f"\nGenerating {args.train_samples} training samples...")
    train_data = generate_dataset(args.train_samples, seed=args.seed)

    # Generate evaluation data (different seed)
    print(f"Generating {args.eval_samples} evaluation samples...")
    eval_data = generate_dataset(args.eval_samples, seed=args.seed + 1000)

    # Save data
    if args.format in ["json", "both"]:
        train_path = os.path.join(args.output_dir, "train_data.json")
        eval_path = os.path.join(args.output_dir, "eval_data.json")

        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)

        print(f"\nSaved JSON files:")
        print(f"  Training: {train_path}")
        print(f"  Evaluation: {eval_path}")

    if args.format in ["pkl", "both"]:
        train_path = os.path.join(args.output_dir, "train_data.pkl")
        eval_path = os.path.join(args.output_dir, "eval_data.pkl")

        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(eval_path, 'wb') as f:
            pickle.dump(eval_data, f)

        print(f"\nSaved Pickle files:")
        print(f"  Training: {train_path}")
        print(f"  Evaluation: {eval_path}")

    # Print sample
    print("\n" + "=" * 60)
    print("Sample Generated Data:")
    print("=" * 60)
    sample = train_data[0]
    print(f"\n[cS - Static Curriculum]:\n{sample['c_static']}")
    print(f"\n[cC - Case Curriculum]:\n{sample['c_case']}")
    print(f"\n[cU - User Curriculum]:\n{sample['c_user']}")
    print(f"\n[x - Task Input]:\n{sample['x']}")
    print(f"\n[y - Target Output]:\n{sample['y']}")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)

    # Print usage instructions
    print("\nTo train with this data:")
    print(f"""
python train_3CEL.py \\
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \\
    --train_path {os.path.join(args.output_dir, 'train_data.json')} \\
    --eval_path {os.path.join(args.output_dir, 'eval_data.json')} \\
    --output_dir ./outputs/clinical_3cel \\
    --dropout_static 0.5 \\
    --dropout_case 0.5 \\
    --dropout_user 0.5 \\
    --do_train --do_eval
""")


if __name__ == "__main__":
    main()
