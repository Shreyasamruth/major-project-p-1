import unittest
import os
import tempfile
import json
from engine.pipeline import DeidPipeline
from engine.redactor import DocumentRedactor
from engine.vault import PHIVault
from utils.db_handler import DBHandler

class TestFullApplicationWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.temp_vault = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_vault.close()

        self.secret_key = "AdminMasterSecret2026"
        self.vault = PHIVault(storage_path=self.temp_vault.name, secret_key=self.secret_key)
        self.db = DBHandler(db_path=self.temp_db.name, legacy_json_path="non_existent.json")
        self.pipeline = DeidPipeline(vault=self.vault)
        self.redactor = DocumentRedactor()

    def tearDown(self):
        for path in [self.temp_db.name, self.temp_vault.name]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_complete_admin_doctor_workflow(self):
        doc_id = "WORKFLOW_DOC_001"
        sample_report = """Healthcare Clinical Record
Patient: John Smith
MRN: MRN123456
Phone: 9876543210
Diagnosis: Diabetes
Medication: Metformin 500 mg
BP: 120/80 mmHg
"""

        # Step 1: Admin Document Registration & Status PENDING -> PROCESSING
        self.db.add_document(
            doc_id=doc_id,
            title="John Smith Medical Report",
            masked_text="",
            disease_data=[],
            original_content=sample_report,
            status="PENDING"
        )
        self.db.update_document_status(doc_id, "PROCESSING", "Core processing pipeline started")

        doc = self.db.get_document(doc_id)
        self.assertEqual(doc["status"], "PROCESSING")

        # Step 2: Core Processing Pipeline (PHI Detection & Context Preservation)
        pipe_res = self.pipeline.process_document(
            file_bytes=sample_report.encode('utf-8'),
            file_name="john_smith.txt",
            doc_id=doc_id
        )
        self.db.update_document_status(doc_id, "REVIEW_REQUIRED", "PHI entities detected")

        # Step 3: Redaction Confirmation & Document Generation
        txt_out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        txt_out.close()
        try:
            redact_res = self.redactor.redact_document(
                file_path=sample_report,
                file_type="txt",
                entities=pipe_res.final_entities,
                output_path=txt_out.name,
                vault_map=pipe_res.synthetic_placeholders
            )

            # Step 4: Share Redacted Document to Doctor Portal
            self.db.add_document(
                doc_id=doc_id,
                title="John Smith Medical Report",
                masked_text=redact_res["masked_text"],
                disease_data=pipe_res.preserved_entities,
                file_b64=redact_res["file_b64"],
                file_ext=".txt",
                original_content=sample_report,
                status="SHARED"
            )

            shared_doc = self.db.get_document(doc_id)
            self.assertEqual(shared_doc["status"], "SHARED")

            # Step 5: Doctor Portal Queue Access & Security Isolation Verification
            doctor_queue = self.db.get_doctor_queue()
            self.assertIn(doc_id, doctor_queue)
            doc_for_doctor = doctor_queue[doc_id]

            # SECURITY ASSERTION 1: Doctor role strictly sees empty original_content and original_file_b64
            self.assertEqual(doc_for_doctor["original_content"], "")
            self.assertEqual(doc_for_doctor["original_file_b64"], "")

            # SECURITY ASSERTION 2: Doctor/unauthorized attempt to demask fails
            unauth_demask = self.vault.demask_text(doc_for_doctor["masked_text"], doc_id, provided_key="DoctorUnauthorizedKey")
            self.assertTrue(unauth_demask.startswith("ERROR: Unauthorized access"))

            # Step 6: Doctor Submits Clinical Diagnosis & Updates Status to DOCTOR_REVIEWED
            doctor_notes = "Patient advised to continue Metformin 500 mg daily. Re-check BP in 2 weeks."
            self.db.update_doctor_notes(doc_id, doctor_notes)

            reviewed_doc = self.db.get_document(doc_id)
            self.assertEqual(reviewed_doc["status"], "DOCTOR_REVIEWED")
            self.assertEqual(reviewed_doc["doctor_notes"], doctor_notes)

            # Step 7 & 8: Authorized Admin Inspection & Demasking
            demasked_content = self.vault.demask_text(reviewed_doc["masked_text"], doc_id, provided_key=self.secret_key)
            
            self.assertIn("John Smith", demasked_content)
            self.assertIn("MRN123456", demasked_content)
            self.assertIn("9876543210", demasked_content)

            self.db.update_document_status(doc_id, "COMPLETED", "Admin completed review and authorized demasking")

            final_doc = self.db.get_document(doc_id)
            self.assertEqual(final_doc["status"], "COMPLETED")

            # Step 9: Verify HIPAA Audit Log Entries
            audit_logs = self.db.get_audit_logs()
            self.assertTrue(len(audit_logs) >= 5)
            log_actions = [l["action"] for l in audit_logs]
            self.assertTrue(any("DOCUMENT_CREATED" in a for a in log_actions))
            self.assertTrue(any("STATUS_UPDATED" in a for a in log_actions))
            self.assertTrue(any("DOCTOR_NOTE_ADDED" in a for a in log_actions))

        finally:
            if os.path.exists(txt_out.name):
                os.remove(txt_out.name)

if __name__ == "__main__":
    unittest.main()
