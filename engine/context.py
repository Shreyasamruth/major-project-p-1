from typing import List, Dict, Any
import re

class ContextPreserver:
    def __init__(self):
        # Expanded regex patterns for clinical/medical values that must NOT be redacted
        self.medical_value_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|l|kg|cm|mm|units|iu|mmol/l|mg/dl|meq/l|ng/ml|u/l|bpm|%)(?!\w)', # Dosages, Units & SpO2
            r'\b\d{2,3}/\d{2,3}\s*(?:mmHg)?\b',                                                          # Blood Pressure (e.g. 120/80 mmHg)
            r'\b(?:pH|Hb|WBC|RBC|HCT|PLT|ALT|AST|BUN|Cr|SpO2|HR|BP|BMP|CBC|SaO2)\s*:?\s*\d+(?:\.\d+)?%?\b', # Vitals & Lab metrics
            r'\b\d+(?:\.\d+)?%?\s*(?:SpO2|SaO2|O2|oxygen)\b',                                            # SpO2 patterns (e.g. 98% SpO2)
            r'\b\d+(?:\.\d+)?\s*°\s*[CF]\b',                                                              # Temperatures (e.g. 98.6°F, 37.2 °C)
            r'\b(?:positive|negative|reactive|non-reactive|normal|abnormal|trace|elevated|borderline)\b', # Qualitative findings
        ]
        
        # Medical terminology whitelist
        self.whitelist = {
            "hypertension", "diabetes", "asthma", "pneumonia", "carcinoma", "ischemic",
            "ct scan", "mri", "x-ray", "ultrasound", "electrocardiogram", "ecg", "eeg",
            "blood pressure", "heart rate", "respiratory rate", "oxygen saturation", "spo2", "sao2",
            "findings", "impression", "history", "patient", "clinical", "evaluation",
            "diagnosis", "prognosis", "treatment", "medication", "prescription",
            "cardiac", "pulmonary", "oncology", "neurology", "creatinine", "hemoglobin",
            "glucose", "cholesterol", "troponin", "arrhythmia", "sepsis", "infection",
            "lesion", "tumor", "fracture", "edema", "effusion", "acute", "chronic",
            "metformin", "lisinopril", "atorvastatin", "albuterol", "aspirin", "paracetamol",
            "amoxicillin", "penicillin", "insulin", "ibuprofen", "omeprazole"
        }

    def is_medical_context(self, text: str, label: str = "") -> bool:
        """Returns True if text is a medical value, dosage, measurement, or whitelisted term."""
        if not text or not text.strip():
            return False
        clean_text = text.strip()
        
        # 1. Pattern checks
        for pattern in self.medical_value_patterns:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return True
        
        # 2. Whitelist checks
        if clean_text.lower() in self.whitelist:
            return True

        # 3. Special logic for relative dates (e.g., "3 days ago", "post-op day 2")
        relative_date_patterns = [
            r'\b\d+\s?(?:day|days|week|weeks|month|months|year|years)\s?ago\b',
            r'\btoday\b', r'\byesterday\b', r'\btomorrow\b', r'\bpost-op\s?day\s?\d+\b'
        ]
        for pattern in relative_date_patterns:
            if re.search(pattern, clean_text, re.IGNORECASE):
                return True

        return False

    def filter_phi(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out entities that are actually medical context."""
        filtered = []
        for ent in entities:
            if not self.is_medical_context(ent.get('text', ''), ent.get('label', 'PHI')):
                filtered.append(ent)
        return filtered


