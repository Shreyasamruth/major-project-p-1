import fitz  # PyMuPDF
from typing import List, Dict, Any
import os

class PDFHandler:
    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> str:
        """Extracts text content from digital PDF pages using PyMuPDF."""
        if not os.path.exists(pdf_path):
            return ""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        return text.strip()

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """Determines if PDF is scanned (lacks selectable text)."""
        extracted = self.extract_text(pdf_path)
        return len(extracted.strip()) < 10

    def apply_redactions(
        self,
        pdf_path: str,
        entities: List[Dict[str, Any]],
        output_path: str,
        vault_map: Dict[str, str] = None
    ):
        """
        Applies genuine PyMuPDF PDF redactions that physically purge original sensitive text
        from PDF streams and replace them with synthetic placeholder annotations.
        """
        doc = fitz.open(pdf_path)

        reverse_map = {}
        if vault_map:
            for synth, orig in vault_map.items():
                if isinstance(orig, str):
                    reverse_map[orig.lower().strip()] = synth

        counts: Dict[str, int] = {}

        for page in doc:
            for ent in entities:
                orig_text = ent.get('text', '').strip()
                label = ent.get('label', 'PHI').upper()

                replacement = reverse_map.get(orig_text.lower(), None)
                if not replacement:
                    clean_label = label.replace("PERSON", "PATIENT_NAME").replace("GPE", "LOCATION")
                    counts[clean_label] = counts.get(clean_label, 0) + 1
                    replacement = f"[{clean_label}_{counts[clean_label]}]"

                # 1. Use box_2d coordinates if available
                if 'box_2d' in ent and isinstance(ent['box_2d'], list) and len(ent['box_2d']) == 4:
                    ymin, xmin, ymax, xmax = ent['box_2d']
                    rect = fitz.Rect(
                        (xmin / 1000.0) * page.rect.width,
                        (ymin / 1000.0) * page.rect.height,
                        (xmax / 1000.0) * page.rect.width,
                        (ymax / 1000.0) * page.rect.height
                    )
                    page.add_redact_annot(rect, text=replacement, fill=(1, 1, 1), text_color=(0.8, 0, 0), fontsize=7)
                elif orig_text:
                    # 2. Search for exact text instances on the page
                    text_instances = page.search_for(orig_text)
                    for inst in text_instances:
                        page.add_redact_annot(inst, text=replacement, fill=(1, 1, 1), text_color=(0.8, 0, 0), fontsize=7)

            # Apply redactions to physically strip text from page rendering stream
            page.apply_redactions(graphics=True)

        # Save with full garbage collection to physically remove erased bytes
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()

    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[bytes]:
        """Converts PDF pages into list of image bytes for rendering or OCR fallback."""
        doc = fitz.open(pdf_path)
        images = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            images.append(img_bytes)

        doc.close()
        return images

if __name__ == "__main__":
    pass

