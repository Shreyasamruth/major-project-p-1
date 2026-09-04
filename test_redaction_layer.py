import unittest
import os
import tempfile
import fitz  # PyMuPDF
from engine.pipeline import DeidPipeline
from engine.redactor import DocumentRedactor
from utils.pdf_handler import PDFHandler

class TestDocumentRedactionLayer(unittest.TestCase):
    def setUp(self):
        self.pipeline = DeidPipeline()
        self.redactor = DocumentRedactor()
        self.pdf_handler = PDFHandler()
        self.temp_dir = tempfile.TemporaryDirectory()

        self.sample_text = """Healthcare Medical Report
Patient: John Smith
MRN: MRN123456
Phone: 9876543210
Diagnosis: Diabetes
Medication: Metformin 500 mg
BP: 120/80 mmHg
"""

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_1_txt_redaction(self):
        """Test TXT file redaction on sample medical document."""
        txt_path = os.path.join(self.temp_dir.name, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.sample_text)

        pipe_res = self.pipeline.process_document(file_path=txt_path, file_name="sample.txt")
        
        out_txt_path = os.path.join(self.temp_dir.name, "redacted_sample.txt")
        res = self.redactor.redact_document(
            file_path=txt_path,
            file_type="txt",
            entities=pipe_res.final_entities,
            output_path=out_txt_path,
            vault_map=pipe_res.synthetic_placeholders
        )

        with open(res["output_path"], "r", encoding="utf-8") as f:
            redacted_content = f.read()

        # Assert PHI is redacted
        self.assertNotIn("John Smith", redacted_content)
        self.assertNotIn("MRN123456", redacted_content)
        self.assertNotIn("9876543210", redacted_content)

        # Assert medical concepts are preserved
        self.assertIn("Diabetes", redacted_content)
        self.assertIn("Metformin 500 mg", redacted_content)
        self.assertIn("120/80 mmHg", redacted_content)

    def test_2_digital_pdf_stream_purge_redaction(self):
        """Test PyMuPDF physical font stream purging on selectable digital PDF."""
        # 1. Create a digital PDF with sample text using PyMuPDF
        input_pdf = os.path.join(self.temp_dir.name, "input_digital.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), self.sample_text, fontsize=12)
        doc.save(input_pdf)
        doc.close()

        # 2. Run Pipeline to detect PHI
        pipe_res = self.pipeline.process_document(file_path=input_pdf, file_name="input_digital.pdf")

        # 3. Apply PyMuPDF physical stream redactions
        output_pdf = os.path.join(self.temp_dir.name, "redacted_digital.pdf")
        self.redactor.redact_digital_pdf(
            input_pdf_path=input_pdf,
            entities=pipe_res.final_entities,
            output_pdf_path=output_pdf,
            vault_map=pipe_res.synthetic_placeholders
        )

        # 4. Re-extract text from generated output PDF to verify physical byte purge
        extracted_redacted_text = self.pdf_handler.extract_text(output_pdf)

        # Assert original sensitive text bytes are 100% PURGED from PDF text stream
        self.assertNotIn("John Smith", extracted_redacted_text)
        self.assertNotIn("MRN123456", extracted_redacted_text)
        self.assertNotIn("9876543210", extracted_redacted_text)

        # Assert clinical metrics ARE preserved in PDF text stream
        self.assertIn("Diabetes", extracted_redacted_text)
        self.assertIn("Metformin 500 mg", extracted_redacted_text)
        self.assertIn("120/80 mmHg", extracted_redacted_text)

    def test_3_scanned_pdf_redaction(self):
        """Test scanned PDF frame rendering, bounding box overlay, and PDF re-compilation."""
        # Create an image-based scanned PDF page
        input_pdf = os.path.join(self.temp_dir.name, "input_scanned.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), self.sample_text, fontsize=12)
        pix = page.get_pixmap()
        
        # Save as single-page image PDF (simulated scanned scan)
        img_pdf = fitz.open()
        img_page = img_pdf.new_page(width=pix.width, height=pix.height)
        img_page.insert_image(img_page.rect, stream=pix.tobytes("png"))
        img_pdf.save(input_pdf)
        img_pdf.close()
        doc.close()

        pipe_res = self.pipeline.process_document(file_path=input_pdf, file_name="input_scanned.pdf")

        output_pdf = os.path.join(self.temp_dir.name, "redacted_scanned.pdf")
        self.redactor.redact_scanned_pdf(
            input_pdf_path=input_pdf,
            entities=pipe_res.final_entities,
            output_pdf_path=output_pdf,
            vault_map=pipe_res.synthetic_placeholders
        )

        self.assertTrue(os.path.exists(output_pdf))
        self.assertGreater(os.path.getsize(output_pdf), 1000)

    def test_4_image_redaction(self):
        """Test PNG / JPG image bounding box white-out overlay redaction."""
        # Generate image file from sample text using PyMuPDF pixmap
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), self.sample_text, fontsize=12)
        pix = page.get_pixmap()
        img_path = os.path.join(self.temp_dir.name, "sample_report.png")
        pix.save(img_path)
        doc.close()

        pipe_res = self.pipeline.process_document(file_path=img_path, file_name="sample_report.png")

        out_img_path = os.path.join(self.temp_dir.name, "redacted_report.png")
        self.redactor.redact_image(
            input_img_path=img_path,
            entities=pipe_res.final_entities,
            output_img_path=out_img_path,
            vault_map=pipe_res.synthetic_placeholders
        )

        self.assertTrue(os.path.exists(out_img_path))
        self.assertGreater(os.path.getsize(out_img_path), 1000)

if __name__ == "__main__":
    unittest.main()
