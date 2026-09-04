import unittest
import os
import tempfile
import json
import sqlite3
from engine.vault import PHIVault
from engine.deid import DeidEngine
from engine.context import ContextPreserver
from utils.db_handler import DBHandler
from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler
from agents.deid_agent import DeidAgent

class TestPHIVault(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_file.close()
        self.vault = PHIVault(storage_path=self.temp_file.name, secret_key="TestSecretKey123")

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_encryption_roundtrip(self):
        """Test Fernet encryption and decryption mapping roundtrip."""
        self.vault.add_mapping("doc1", {"[PATIENT_NAME_1]": "Alice Smith", "[SSN_1]": "123-45-6789"})
        vault2 = PHIVault(storage_path=self.temp_file.name, secret_key="TestSecretKey123")
        self.assertEqual(vault2.get_original("doc1", "[PATIENT_NAME_1]"), "Alice Smith")
        self.assertEqual(vault2.get_original("doc1", "[SSN_1]"), "123-45-6789")

    def test_demask_text(self):
        """Test placeholder de-masking back to original PHI."""
        self.vault.add_mapping("doc2", {"[PATIENT_NAME_1]": "Bob Jones"})
        demasked = self.vault.demask_text("Patient [PATIENT_NAME_1] diagnosed with hypertension.", "doc2")
        self.assertEqual(demasked, "Patient Bob Jones diagnosed with hypertension.")

    def test_unauthorized_demasking(self):
        """Test key validation for authorized demasking."""
        self.vault.add_mapping("doc3", {"[PATIENT_NAME_1]": "Secret Person"})
        res = self.vault.demask_text("Patient [PATIENT_NAME_1]", "doc3", provided_key="WrongKey")
        self.assertTrue(res.startswith("ERROR"))


class TestDeidEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DeidEngine()

    def test_regex_detection(self):
        """Test detection of MRN, SSN, Phone, Email, PID."""
        sample_text = "Patient SSN is 987-65-4321, MRN: MRN-884920, Phone: 555-123-4567, Email: doctor@hospital.com."
        entities = self.engine.detect_phi(sample_text)
        labels = [e["label"] for e in entities]
        self.assertIn("SSN", labels)
        self.assertIn("MRN", labels)
        self.assertIn("PHONE", labels)
        self.assertIn("EMAIL", labels)

    def test_mask_text_placeholder_consistency(self):
        """Test that identical text maps consistently to the exact same synthetic placeholder."""
        sample_text = "John Smith met John Smith on 2026-05-10."
        entities = [
            {"text": "John Smith", "start": 0, "end": 10, "label": "PERSON"},
            {"text": "John Smith", "start": 15, "end": 25, "label": "PERSON"},
            {"text": "2026-05-10", "start": 29, "end": 39, "label": "DATE"}
        ]
        masked, vault_map = self.engine.mask_text(sample_text, entities, reversible=True)
        self.assertEqual(masked.count("[PATIENT_NAME_1]"), 2)
        self.assertIn("[DATE_1]", masked)
        self.assertEqual(vault_map["[PATIENT_NAME_1]"], "John Smith")

    def test_entity_merging(self):
        """Test overlapping entity span resolution."""
        sample_text = "MRN-123456"
        entities = [
            {"text": "MRN-123456", "start": 0, "end": 10, "label": "MRN", "confidence": 0.98},
            {"text": "123456", "start": 4, "end": 10, "label": "PID", "confidence": 0.90}
        ]
        merged = self.engine._merge_entities(sample_text, entities)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "MRN-123456")


class TestContextPreserver(unittest.TestCase):
    def setUp(self):
        self.preserver = ContextPreserver()

    def test_medical_value_preservation(self):
        """Test that dosages, BP, SpO2, and temperatures are preserved."""
        self.assertTrue(self.preserver.is_medical_context("500 mg", "DOSAGE"))
        self.assertTrue(self.preserver.is_medical_context("120/80 mmHg", "BP"))
        self.assertTrue(self.preserver.is_medical_context("98.6°F", "TEMP"))
        self.assertTrue(self.preserver.is_medical_context("98% SpO2", "SPO2"))
        self.assertTrue(self.preserver.is_medical_context("diabetes", "DISEASE"))

    def test_relative_date_preservation(self):
        """Test relative dates preservation versus absolute dates."""
        self.assertTrue(self.preserver.is_medical_context("3 days ago", "DATE"))
        self.assertFalse(self.preserver.is_medical_context("2026-05-10", "DATE"))

    def test_filter_phi(self):
        """Test filtering out clinical measurements from PHI redaction list."""
        entities = [
            {"text": "John Doe", "label": "PERSON"},
            {"text": "500 mg", "label": "DOSAGE"},
            {"text": "diabetes", "label": "DISEASE"}
        ]
        filtered = self.preserver.filter_phi(entities)
        filtered_texts = [e["text"] for e in filtered]
        self.assertIn("John Doe", filtered_texts)
        self.assertNotIn("500 mg", filtered_texts)
        self.assertNotIn("diabetes", filtered_texts)


class TestDBHandler(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db = DBHandler(db_path=self.temp_db.name, legacy_json_path="non_existent.json")

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    def test_add_and_audit(self):
        """Test SQLite document insertion and audit trail logging."""
        self.db.add_document("doc100", "Test Report", "Masked content", [], status="SHARED")
        docs = self.db.get_all_documents()
        self.assertIn("doc100", docs)
        
        logs = self.db.get_audit_logs()
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["doc_id"], "doc100")
        self.assertEqual(logs[0]["action"], "DOCUMENT_CREATED")

    def test_doctor_update(self):
        """Test Doctor clinical notes update and status transition."""
        self.db.add_document("doc101", "Test Report 2", "Masked content", [], status="SHARED")
        self.db.update_doctor_notes("doc101", "Prescribed Metformin 500mg daily.")
        
        doc = self.db.get_document("doc101")
        self.assertEqual(doc["status"], "DOCTOR_REVIEWED")
        self.assertEqual(doc["doctor_notes"], "Prescribed Metformin 500mg daily.")


class TestOCRAndPDFHandlers(unittest.TestCase):
    def setUp(self):
        self.ocr = OCRHandler()
        self.pdf_handler = PDFHandler()
        self.temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        self.temp_out.close()

    def tearDown(self):
        if os.path.exists(self.temp_out.name):
            os.remove(self.temp_out.name)

    def test_generate_disease_image(self):
        """Test clinical report image generation."""
        disease_data = [{"label": "DISEASE", "text": "Type 2 Diabetes"}]
        res_path = self.ocr.generate_disease_image(disease_data, "testdoc1", self.temp_out.name)
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 0)


class TestEndToEndAcceptanceScenario(unittest.TestCase):
    """
    Executes Section 22 Acceptance Scenario:
    1. Upload healthcare document with John Smith, MRN123456, 9876543210, Diabetes, Metformin 500 mg, 120/80 mmHg.
    2. Verify PHI detection identifies PHI while preserving clinical information.
    3. Synthetic placeholders generated ([PATIENT_NAME_1], [MRN_1], [PHONE_1]).
    4. Doctor review submission.
    5. Authorized Admin demasking.
    6. Audit log recording.
    """
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.temp_vault = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_vault.close()

        self.engine = DeidEngine()
        self.preserver = ContextPreserver()
        self.vault = PHIVault(storage_path=self.temp_vault.name, secret_key="AcceptanceKey123")
        self.db = DBHandler(db_path=self.temp_db.name, legacy_json_path="non_existent.json")

    def tearDown(self):
        for path in [self.temp_db.name, self.temp_vault.name]:
            if os.path.exists(path):
                os.remove(path)

    def test_acceptance_flow(self):
        raw_text = """
        Healthcare Medical Report
        Patient: John Smith
        MRN: MRN123456
        Phone: 9876543210
        Diagnosis: Diabetes
        Medication: Metformin 500 mg
        BP: 120/80 mmHg
        """

        # 1. PHI Detection
        detected_entities = self.engine.detect_phi(raw_text)
        phi_filtered = self.preserver.filter_phi(detected_entities)
        phi_texts = [p['text'] for p in phi_filtered]

        # Verify PHI detected
        self.assertTrue(any("John Smith" in t for t in phi_texts))
        self.assertTrue(any("MRN123456" in t for t in phi_texts))
        self.assertTrue(any("9876543210" in t for t in phi_texts))

        # Verify clinical info preserved
        self.assertFalse(any("Diabetes" in t for t in phi_texts))
        self.assertFalse(any("500 mg" in t for t in phi_texts))
        self.assertFalse(any("120/80 mmHg" in t for t in phi_texts))

        # 2. Synthetic Token Generation
        doc_id = "ACC001"
        masked_text, doc_vault = self.engine.mask_text(raw_text, phi_filtered, reversible=True)
        self.vault.add_mapping(doc_id, doc_vault)

        self.assertIn("[PATIENT_NAME_1]", masked_text)
        self.assertIn("Diabetes", masked_text)
        self.assertIn("Metformin 500 mg", masked_text)
        self.assertIn("120/80 mmHg", masked_text)

        # 3. DB Registration & Shared Status
        self.db.add_document(
            doc_id=doc_id,
            title="Acceptance Test Report",
            masked_text=masked_text,
            disease_data=[{"text": "Diabetes", "label": "DISEASE"}],
            status="SHARED"
        )
        doc = self.db.get_document(doc_id)
        self.assertEqual(doc["status"], "SHARED")

        # 4. Doctor Review Submission
        self.db.update_doctor_notes(doc_id, "Patient advised to continue medication.")
        updated_doc = self.db.get_document(doc_id)
        self.assertEqual(updated_doc["status"], "DOCTOR_REVIEWED")
        self.assertEqual(updated_doc["doctor_notes"], "Patient advised to continue medication.")

        # 5. Authorized Demasking
        demasked = self.vault.demask_text(masked_text, doc_id, provided_key="AcceptanceKey123")
        self.assertIn("John Smith", demasked)
        self.assertIn("MRN123456", demasked)

        # 6. Audit Trail Verification
        logs = self.db.get_audit_logs()
        self.assertTrue(len(logs) >= 2)


if __name__ == "__main__":
    unittest.main()

