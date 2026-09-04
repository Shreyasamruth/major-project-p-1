import unittest
import os
import tempfile
from engine.pipeline import DeidPipeline, PipelineResult

class TestDeidPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DeidPipeline()

    def test_text_document_pipeline(self):
        """Test end-to-end pipeline processing on plain text medical report."""
        text_content = """
        Healthcare Medical Report
        Patient: Sarah Connor
        MRN: MRN-887766
        SSN: 123-45-6789
        Phone: 555-0199
        Diagnosis: Type 2 Diabetes
        Medication: Metformin 500 mg
        BP: 120/80 mmHg
        SpO2: 98% SpO2
        """
        result = self.pipeline.process_document(
            file_bytes=text_content.encode('utf-8'),
            file_name="sample_report.txt",
            doc_id="TEST_TXT_01"
        )
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.doc_id, "TEST_TXT_01")
        self.assertEqual(result.file_type, "txt")
        self.assertIn("Sarah Connor", result.extracted_text)
        
        # Verify detected entities & redaction spans
        redacted_texts = [span['text'] for span in result.redaction_spans]
        self.assertTrue(any("Sarah Connor" in t for t in redacted_texts))
        self.assertTrue(any("MRN-887766" in t for t in redacted_texts))
        self.assertTrue(any("123-45-6789" in t for t in redacted_texts))

        # Verify context preservation (Diabetes, 500 mg, 120/80 mmHg, 98% SpO2 NOT redacted)
        self.assertFalse(any("Diabetes" in t for t in redacted_texts))
        self.assertFalse(any("500 mg" in t for t in redacted_texts))
        self.assertFalse(any("120/80 mmHg" in t for t in redacted_texts))

        # Verify synthetic placeholders
        self.assertIn("[PATIENT_NAME_1]", result.masked_text)
        self.assertIn("[SSN_1]", result.masked_text)
        self.assertIn("Type 2 Diabetes", result.masked_text)
        self.assertIn("Metformin 500 mg", result.masked_text)

        # Verify structured outputs
        self.assertIsInstance(result.synthetic_placeholders, dict)
        self.assertIsInstance(result.confidence_sources, list)
        self.assertGreaterEqual(result.processing_time_sec, 0.0)

    def test_pipeline_result_to_dict(self):
        """Test serialization of PipelineResult to dictionary."""
        result = self.pipeline.process_document(
            file_bytes=b"Patient: John Doe",
            file_name="simple.txt"
        )
        res_dict = result.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertIn("doc_id", res_dict)
        self.assertIn("masked_text", res_dict)
        self.assertIn("redaction_spans", res_dict)

if __name__ == "__main__":
    unittest.main()
