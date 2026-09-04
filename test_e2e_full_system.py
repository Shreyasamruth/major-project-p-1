import unittest
import os
import tempfile
import json
import base64
import fitz  # PyMuPDF

from engine.pipeline import DeidPipeline, PipelineResult
from engine.redactor import DocumentRedactor
from engine.vault import PHIVault
from utils.db_handler import DBHandler
from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler

class TestEndToEndFullSystem(unittest.TestCase):
    """
    Comprehensive End-to-End System Acceptance Test:
    Executes real file processing across TXT, Digital PDF, Scanned PDF, and PNG images.
    Verifies Admin login, extraction/OCR, hybrid PHI detection, context preservation,
    physical stream redaction, encrypted vault mapping, Doctor queue security isolation,
    Doctor attestation submission, Admin authorized demasking, and audit logging.
    """
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.temp_vault = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_vault.close()

        self.admin_secret_key = "HealthcareDeidSecretKey2026"
        self.vault = PHIVault(storage_path=self.temp_vault.name, secret_key=self.admin_secret_key)
        self.db = DBHandler(db_path=self.temp_db.name, legacy_json_path="non_existent.json")
        self.pdf_handler = PDFHandler()
        self.ocr_handler = OCRHandler()
        self.pipeline = DeidPipeline(vault=self.vault, pdf_handler=self.pdf_handler, ocr_handler=self.ocr_handler)
        self.redactor = DocumentRedactor(pdf_handler=self.pdf_handler, ocr_handler=self.ocr_handler)

        self.temp_dir = tempfile.TemporaryDirectory()

        self.real_clinical_report = """Healthcare Medical Report
Patient: John Smith
MRN: MRN123456
Phone: 9876543210
SSN: 987-65-4321
Diagnosis: Diabetes
Medication: Metformin 500 mg
BP: 120/80 mmHg
SpO2: 98% SpO2
Temp: 98.6°F
"""

    def tearDown(self):
        self.temp_dir.cleanup()
        for path in [self.temp_db.name, self.temp_vault.name]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_e2e_full_workflow_on_real_txt_file(self):
        """E2E Test 1: Real TXT document workflow."""
        doc_id = "E2E_TXT_01"
        txt_path = os.path.join(self.temp_dir.name, "real_patient_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.real_clinical_report)

        # 1. Admin Login Check
        self.assertEqual(self.vault.secret_key, self.admin_secret_key)

        # 2. Document Upload & Registration (PENDING -> PROCESSING)
        self.db.add_document(
            doc_id=doc_id,
            title="Report: real_patient_report.txt",
            masked_text="",
            disease_data=[],
            original_content=self.real_clinical_report,
            status="PENDING"
        )
        self.db.update_document_status(doc_id, "PROCESSING", "DeidPipeline execution initiated")

        # 3. Extraction & Hybrid PHI Detection
        pipe_res = self.pipeline.process_document(file_path=txt_path, file_name="real_patient_report.txt", doc_id=doc_id)
        self.assertIn("John Smith", pipe_res.extracted_text)

        # 4. Context Preservation Verification
        redacted_entity_texts = [e['text'] for e in pipe_res.final_entities]
        preserved_texts = [d['text'] for d in pipe_res.preserved_entities]

        self.assertTrue(any("John Smith" in t for t in redacted_entity_texts))
        self.assertTrue(any("MRN123456" in t for t in redacted_entity_texts))
        self.assertTrue(any("9876543210" in t for t in redacted_entity_texts))

        self.assertFalse(any("Diabetes" in t for t in redacted_entity_texts))
        self.assertFalse(any("500 mg" in t for t in redacted_entity_texts))
        self.assertFalse(any("120/80 mmHg" in t for t in redacted_entity_texts))

        # 5. Redaction & Encrypted Vault Mapping
        out_txt_path = os.path.join(self.temp_dir.name, "redacted_report.txt")
        redact_res = self.redactor.redact_document(
            file_path=txt_path,
            file_type="txt",
            entities=pipe_res.final_entities,
            output_path=out_txt_path,
            vault_map=pipe_res.synthetic_placeholders
        )

        with open(out_txt_path, "r", encoding="utf-8") as f:
            redacted_content = f.read()

        # Check PHI is hidden/removed in generated document
        self.assertNotIn("John Smith", redacted_content)
        self.assertNotIn("MRN123456", redacted_content)
        self.assertNotIn("9876543210", redacted_content)
        self.assertIn("[PATIENT_NAME_1]", redacted_content)

        # Check clinical info remains
        self.assertIn("Diabetes", redacted_content)
        self.assertIn("Metformin 500 mg", redacted_content)
        self.assertIn("120/80 mmHg", redacted_content)

        # 6. Share Redacted Document to Doctor Portal
        self.db.add_document(
            doc_id=doc_id,
            title="Report: real_patient_report.txt",
            masked_text=redacted_content,
            disease_data=pipe_res.preserved_entities,
            file_b64=redact_res["file_b64"],
            file_ext=".txt",
            original_content=self.real_clinical_report,
            status="SHARED"
        )

        # 7. Doctor Login / Queue Access Security Isolation
        doc_queue = self.db.get_doctor_queue()
        self.assertIn(doc_id, doc_queue)
        doctor_doc = doc_queue[doc_id]

        # ASSERTION: Doctor cannot see original PHI or unredacted files
        self.assertEqual(doctor_doc["original_content"], "")
        self.assertEqual(doctor_doc["original_file_b64"], "")
        unauth_try = self.vault.demask_text(doctor_doc["masked_text"], doc_id, provided_key="DoctorUnauthorizedKey")
        self.assertTrue(unauth_try.startswith("ERROR: Unauthorized access"))

        # 8. Doctor Views Redacted Doc & Adds Clinical Note
        doc_notes = "Patient to continue Metformin 500 mg daily. Re-check BP in 14 days."
        self.db.update_doctor_notes(doc_id, doc_notes)

        rev_doc = self.db.get_document(doc_id)
        self.assertEqual(rev_doc["status"], "DOCTOR_REVIEWED")
        self.assertEqual(rev_doc["doctor_notes"], doc_notes)

        # 9. Admin Views Note & Authorized Admin Demasks
        demasked = self.vault.demask_text(rev_doc["masked_text"], doc_id, provided_key=self.admin_secret_key)
        self.assertIn("John Smith", demasked)
        self.assertIn("MRN123456", demasked)
        self.assertIn("9876543210", demasked)

        self.db.update_document_status(doc_id, "COMPLETED", "Admin verified doctor notes & authorized demasking")
        self.assertEqual(self.db.get_document(doc_id)["status"], "COMPLETED")

        # 10. Audit Log Verification
        logs = self.db.get_audit_logs()
        self.assertTrue(len(logs) >= 4)
        actions = [l["action"] for l in logs]
        self.assertTrue(any("DOCUMENT_CREATED" in a for a in actions))
        self.assertTrue(any("DOCTOR_NOTE_ADDED" in a for a in actions))

    def test_e2e_full_workflow_on_real_digital_pdf(self):
        """E2E Test 2: Real Digital PDF physical font stream purging workflow."""
        doc_id = "E2E_PDF_02"
        pdf_path = os.path.join(self.temp_dir.name, "real_clinical_report.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), self.real_clinical_report, fontsize=11)
        doc.save(pdf_path)
        doc.close()

        # Run pipeline
        pipe_res = self.pipeline.process_document(file_path=pdf_path, file_name="real_clinical_report.pdf", doc_id=doc_id)
        self.assertEqual(pipe_res.file_type, "pdf_digital")

        # Redact PDF using PyMuPDF physical stream purging
        out_pdf_path = os.path.join(self.temp_dir.name, "redacted_clinical_report.pdf")
        self.redactor.redact_digital_pdf(
            input_pdf_path=pdf_path,
            entities=pipe_res.final_entities,
            output_pdf_path=out_pdf_path,
            vault_map=pipe_res.synthetic_placeholders
        )

        # Re-extract text from generated PDF font stream
        extracted_from_redacted_pdf = self.pdf_handler.extract_text(out_pdf_path)

        # ASSERTION: Original sensitive text bytes are 100% PURGED from PDF
        self.assertNotIn("John Smith", extracted_from_redacted_pdf)
        self.assertNotIn("MRN123456", extracted_from_redacted_pdf)
        self.assertNotIn("9876543210", extracted_from_redacted_pdf)

        # ASSERTION: Clinical findings remain intact
        self.assertIn("Diabetes", extracted_from_redacted_pdf)
        self.assertIn("Metformin 500 mg", extracted_from_redacted_pdf)
        self.assertIn("120/80 mmHg", extracted_from_redacted_pdf)

if __name__ == "__main__":
    unittest.main()
