import os
import base64
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple

from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler
from engine.deid import DeidEngine

class DocumentRedactor:
    """
    Unified Document Redaction Layer supporting TXT, Digital PDF, Scanned PDF, PNG, and JPG/JPEG.
    Performs physical PyMuPDF stream purging for digital PDFs and pixel overlays for images.
    """
    def __init__(
        self,
        pdf_handler: Optional[PDFHandler] = None,
        ocr_handler: Optional[OCRHandler] = None,
        deid_engine: Optional[DeidEngine] = None
    ):
        self.pdf_handler = pdf_handler or PDFHandler()
        self.ocr_handler = ocr_handler or OCRHandler()
        self.deid_engine = deid_engine or DeidEngine()

    def redact_text(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        vault_map: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        """Replaces sensitive PHI text entities with synthetic placeholder tokens."""
        return self.deid_engine.mask_text(text, entities, reversible=True)

    def redact_digital_pdf(
        self,
        input_pdf_path: str,
        entities: List[Dict[str, Any]],
        output_pdf_path: str,
        vault_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Applies genuine PyMuPDF PDF redactions that physically purge original sensitive text
        from PDF font streams and replace them with synthetic placeholder annotations.
        """
        self.pdf_handler.apply_redactions(
            pdf_path=input_pdf_path,
            entities=entities,
            output_path=output_pdf_path,
            vault_map=vault_map
        )
        return output_pdf_path

    def redact_scanned_pdf(
        self,
        input_pdf_path: str,
        entities: List[Dict[str, Any]],
        output_pdf_path: str,
        vault_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Converts scanned PDF pages to high-DPI image frames, applies bounding box overlays,
        and re-assembles them into a clean output PDF.
        """
        page_images = self.pdf_handler.convert_pdf_to_images(input_pdf_path, dpi=150)
        out_doc = fitz.open()

        for idx, img_bytes in enumerate(page_images):
            tmp_in_img = f"tmp_scan_in_{idx}.png"
            tmp_out_img = f"tmp_scan_out_{idx}.png"
            try:
                with open(tmp_in_img, "wb") as f:
                    f.write(img_bytes)

                self.ocr_handler.apply_redactions_to_image(
                    image_path=tmp_in_img,
                    phi_entities=entities,
                    output_path=tmp_out_img,
                    vault_map=vault_map
                )

                img_doc = fitz.open(tmp_out_img)
                rect = img_doc[0].rect
                pdf_page = out_doc.new_page(width=rect.width, height=rect.height)
                pdf_page.insert_image(rect, filename=tmp_out_img)
                img_doc.close()
            finally:
                for path in [tmp_in_img, tmp_out_img]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass

        out_doc.save(output_pdf_path, garbage=4, deflate=True)
        out_doc.close()
        return output_pdf_path

    def redact_image(
        self,
        input_img_path: str,
        entities: List[Dict[str, Any]],
        output_img_path: str,
        vault_map: Optional[Dict[str, str]] = None
    ) -> str:
        """Applies pixel-level white-out overlays with synthetic placeholders on images."""
        return self.ocr_handler.apply_redactions_to_image(
            image_path=input_img_path,
            phi_entities=entities,
            output_path=output_img_path,
            vault_map=vault_map
        )

    def redact_document(
        self,
        file_path: str,
        file_type: str,
        entities: List[Dict[str, Any]],
        output_path: str,
        vault_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        High-level redaction dispatcher handling TXT, digital PDF, scanned PDF, and image formats.
        """
        file_ext = os.path.splitext(output_path)[1].lower()

        if file_ext == ".txt":
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            else:
                raw_text = file_path
            masked_text, _ = self.redact_text(raw_text, entities, vault_map)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(masked_text)
            
            with open(output_path, "rb") as f:
                file_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "output_path": output_path,
                "file_b64": file_b64,
                "masked_text": masked_text,
                "file_type": "txt"
            }

        elif file_ext == ".pdf":
            if file_type == "pdf_scanned":
                out_file = self.redact_scanned_pdf(file_path, entities, output_path, vault_map)
            else:
                out_file = self.redact_digital_pdf(file_path, entities, output_path, vault_map)

            with open(out_file, "rb") as f:
                file_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "output_path": out_file,
                "file_b64": file_b64,
                "masked_text": "",
                "file_type": file_type
            }

        elif file_ext in [".png", ".jpg", ".jpeg"]:
            out_file = self.redact_image(file_path, entities, output_path, vault_map)
            with open(out_file, "rb") as f:
                file_b64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "output_path": out_file,
                "file_b64": file_b64,
                "masked_text": "",
                "file_type": "image"
            }

        else:
            raise ValueError(f"Unsupported document format: {file_ext}")
