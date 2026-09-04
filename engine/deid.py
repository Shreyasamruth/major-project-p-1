import spacy
from typing import List, Dict, Any, Tuple
import re

class DeidEngine:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            try:
                self.nlp = spacy.load(model_name)
            except Exception:
                self.nlp = None

        self.phi_labels = ["PERSON", "DATE", "GPE", "ORG", "PHONE", "EMAIL", "SSN", "MRN", "INSURANCE_ID"]
        self.id_patterns = {
            "MRN": r"\b(?:MRN|Medical Record Number)[:\-\s]*[A-Z0-9-]+\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "INSURANCE_ID": r"\b[A-Z]{3}\d{7,9}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "DATE": r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b",
            "PID": r"\b(?:PID|Patient ID|Account No)[:\-\s]*[A-Z0-9-]+\b",
            "AGE": r"\b\d{1,3}\s*(?:years|yrs|yo|y/o|year-old)\b"
        }

    def ensure_span_offsets(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures all entity dicts have valid start and end character offsets in text."""
        if not text or not entities:
            return []
        
        resolved = []
        for ent in entities:
            if 'start' in ent and 'end' in ent and ent['start'] >= 0 and ent['end'] <= len(text):
                resolved.append(ent.copy())
            else:
                ent_text = ent.get('text', '').strip()
                if not ent_text:
                    continue
                # Search for all occurrences of ent_text in text
                pattern = re.escape(ent_text)
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    new_ent = ent.copy()
                    new_ent['start'] = match.start()
                    new_ent['end'] = match.end()
                    new_ent['text'] = text[match.start():match.end()]
                    resolved.append(new_ent)
        return resolved

    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """Detects PHI using Regex patterns and SpaCy NER."""
        entities = []
        if not text or not text.strip():
            return entities

        # 1. Regex Detection
        for label, pattern in self.id_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "label": label,
                    "source": "Regex",
                    "confidence": 0.98,
                    "reasoning": f"Matched pattern for {label}."
                })

        # 2. SpaCy NER
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.phi_labels:
                    entities.append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_,
                        "source": "SpaCy NER",
                        "confidence": 0.90,
                        "reasoning": f"SpaCy entity type {ent.label_}."
                    })

        return self._merge_entities(text, entities)

    def _merge_entities(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges overlapping or duplicate entity spans cleanly."""
        if not entities:
            return []

        resolved_ents = self.ensure_span_offsets(text, entities)
        if not resolved_ents:
            return []
        
        # Sort entities by start index, then by end index descending
        sorted_ents = sorted(resolved_ents, key=lambda x: (x['start'], -x['end']))
        merged = []
        current = sorted_ents[0].copy()

        for next_ent in sorted_ents[1:]:
            if next_ent['start'] < current['end']:
                # Overlapping span
                if next_ent['end'] > current['end']:
                    current['end'] = next_ent['end']
                    current['text'] = text[current['start']:current['end']]
                
                # Update confidence/label if next_ent has higher confidence
                if next_ent.get('confidence', 0) > current.get('confidence', 0):
                    current['label'] = next_ent['label']
                    current['source'] = f"{current.get('source', '')}/{next_ent.get('source', '')}".strip('/')
            else:
                merged.append(current)
                current = next_ent.copy()

        merged.append(current)
        return merged

    def _generate_synthetic_label(self, label: str) -> str:
        clean = label.upper().strip()
        mapping = {
            "PERSON": "PATIENT_NAME",
            "GPE": "LOCATION",
            "ORG": "ORGANIZATION",
            "PID": "PATIENT_ID"
        }
        return mapping.get(clean, clean)

    def mask_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        reversible: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        """
        Replaces detected PHI entities with synthetic placeholders.
        Guarantees that identical original text strings map to the exact same placeholder throughout.
        """
        vault = {}
        if not text:
            return "", vault
        if not entities:
            return text, vault

        merged_entities = self._merge_entities(text, entities)
        
        # Build global memory mapping so "John Smith" always maps to [PATIENT_NAME_1]
        memory_map: Dict[str, str] = {}
        counts: Dict[str, int] = {}

        # First pass: assign consistent placeholders to each unique original text
        for ent in merged_entities:
            orig_val = ent.get('text', text[ent['start']:ent['end']])
            if orig_val not in memory_map:
                synth_type = self._generate_synthetic_label(ent.get('label', 'PHI'))
                counts[synth_type] = counts.get(synth_type, 0) + 1
                placeholder = f"[{synth_type}_{counts[synth_type]}]"
                memory_map[orig_val] = placeholder

        # Reverse sort by start index to apply substitutions in string
        masked_chars = list(text)
        for ent in sorted(merged_entities, key=lambda x: x['start'], reverse=True):
            orig_val = ent.get('text', text[ent['start']:ent['end']])
            placeholder = memory_map.get(orig_val, "[REDACTED]")
            if reversible:
                vault[placeholder] = orig_val
            masked_chars[ent['start']:ent['end']] = list(placeholder)

        return "".join(masked_chars), vault

if __name__ == "__main__":
    pass

