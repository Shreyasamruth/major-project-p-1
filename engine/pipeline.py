import os
import time
import uuid
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from engine.deid import DeidEngine
from engine.context import ContextPreserver
from engine.vault import PHIVault
from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler
from agents.deid_agent import DeidAgent

@dataclass
class PipelineResult:
    doc_id: str
    file_name: str
    file_type: str  # "txt", "pdf_digital", "pdf_scanned", "image"
    extracted_text: str
    raw_detected_entities: List[Dict[str, Any]] = field(default_factory=list)
    preserved_entities: List[Dict[str, Any]] = field(default_factory=list)
    final_entities: List[Dict[str, Any]] = field(default_factory=list)
    redaction_spans: List[Dict[str, Any]] = field(default_factory=list)
    synthetic_placeholders: Dict[str, str] = field(default_factory=dict)
    masked_text: str = ""
    confidence_sources: List[Dict[str, Any]] = field(default_factory=list)
    warnings_errors: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DeidPipeline:
    """
    Unified Healthcare Data De-Identification Processing Pipeline.
    Executes: Document Upload → Extraction/OCR → Hybrid PHI Detection → Context Preservation
             → Entity Merging → Synthetic Placeholder & Vault Map Generation.
    """
    def __init__(
        self,
        engine: Optional[DeidEngine] = None,
        preserver: Optional[ContextPreserver] = None,
        vault: Optional[PHIVault] = None,
        pdf_handler: Optional[PDFHandler] = None,
        ocr_handler: Optional[OCRHandler] = None,
        agent: Optional[DeidAgent] = None
    ):
        self.engine = engine or DeidEngine()
        self.preserver = preserver or ContextPreserver()
        self.vault = vault or PHIVault()
        self.pdf_handler = pdf_handler or PDFHandler()
        self.ocr_handler = ocr_handler or OCRHandler()
        self.agent = agent or DeidAgent()

    def process_document(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_name: str = "document.txt",
        doc_id: Optional[str] = None
    ) -> PipelineResult:
        start_time = time.time()
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]

        warnings_errors = []
        file_ext = os.path.splitext(file_name)[1].lower()
        if not file_ext:
            file_ext = ".txt"

        # Temporary file management if given raw bytes
        tmp_path = file_path
        cleanup_tmp = False
        if not tmp_path and file_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
                cleanup_tmp = True

        extracted_text = ""
        file_type = "txt"
        ai_phi = []
        ai_disease = []

        try:
            # 1. Text Extraction & OCR
            if file_ext == ".txt":
                file_type = "txt"
                if tmp_path and os.path.exists(tmp_path):
                    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted_text = f.read()
                elif file_bytes:
                    extracted_text = file_bytes.decode("utf-8", errors="ignore")
            elif file_ext == ".pdf":
                if tmp_path and os.path.exists(tmp_path):
                    extracted_text = self.pdf_handler.extract_text(tmp_path)
                    if not extracted_text.strip():
                        file_type = "pdf_scanned"
                        warnings_errors.append("Scanned PDF detected without selectable text. Triggering OCR fallback.")
                        ocr_res = self.ocr_handler.run_ocr(tmp_path)
                        extracted_text = ocr_res.get("text", "")
                    else:
                        file_type = "pdf_digital"
            elif file_ext in [".png", ".jpg", ".jpeg"]:
                file_type = "image"
                if tmp_path and os.path.exists(tmp_path):
                    enhanced = self.ocr_handler.enhance_image(tmp_path)
                    ocr_res = self.ocr_handler.run_ocr(enhanced)
                    extracted_text = ocr_res.get("text", "")
                    if os.path.exists(enhanced) and enhanced != tmp_path:
                        try:
                            os.remove(enhanced)
                        except Exception:
                            pass
                
                if not extracted_text.strip():
                    if not file_bytes and tmp_path and os.path.exists(tmp_path):
                        with open(tmp_path, "rb") as f:
                            file_bytes = f.read()
                    if file_bytes:
                        warnings_errors.append("Local OCR returned empty text. Triggering Multimodal AI vision analysis.")
                        mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
                        mm_res = self.agent.run_multimodal_deid(file_bytes, mime_type=mime_type)
                        extracted_text = mm_res.get("full_text", "")
                        for p in mm_res.get("personal_data", []):
                            p_text = p.get('text', '').strip()
                            if p_text:
                                ai_phi.append({
                                    "text": p_text,
                                    "label": p.get('label', 'PHI'),
                                    "confidence": p.get('confidence', 0.95),
                                    "source": "Multimodal AI",
                                    "box_2d": p.get('box_2d', None)
                                })
                        for d in mm_res.get("disease_data", []):
                            d_text = d.get('text', '').strip()
                            if d_text:
                                ai_disease.append({
                                    "text": d_text,
                                    "label": d.get('label', 'DISEASE'),
                                    "confidence": d.get('confidence', 0.95),
                                    "source": "Multimodal AI",
                                    "box_2d": d.get('box_2d', None)
                                })

            if not extracted_text.strip():
                warnings_errors.append("Warning: Could not extract non-empty text content from document.")

            # 2. Hybrid PHI Detection (Regex + SpaCy NER)
            local_entities = self.engine.detect_phi(extracted_text) if extracted_text else []
            
            # AI Agent PHI & Clinical Detection if active
            ai_phi = []
            ai_disease = []
            if self.agent.llm and extracted_text:
                try:
                    ai_res = self.agent.run_deid(extracted_text)
                    for p in ai_res.get("personal_data", []):
                        p_text = p.get('text', '').strip()
                        if p_text:
                            ai_phi.append({
                                "text": p_text,
                                "label": p.get('label', 'PHI'),
                                "confidence": p.get('confidence', 0.95),
                                "source": "Gemini AI",
                                "reasoning": p.get('reasoning', "AI Detection")
                            })
                    for d in ai_res.get("disease_data", []):
                        d_text = d.get('text', '').strip()
                        if d_text:
                            ai_disease.append({
                                "text": d_text,
                                "label": d.get('label', 'DISEASE'),
                                "confidence": d.get('confidence', 0.95),
                                "source": "Gemini AI",
                                "reasoning": p.get('reasoning', "Clinical concept")
                            })
                except Exception as ex:
                    warnings_errors.append(f"AI Agent fallback active ({type(ex).__name__}). Using local detection.")

            raw_detected_entities = local_entities + [p for p in ai_phi if p['text'].lower() not in {x['text'].lower() for x in local_entities}]

            # 3. Context Preservation
            final_to_redact = self.preserver.filter_phi(raw_detected_entities)
            preserved_entities = [e for e in raw_detected_entities if self.preserver.is_medical_context(e.get('text', ''), e.get('label', ''))]

            # 4. Duplicate / Overlapping Entity Resolution
            final_merged_entities = self.engine._merge_entities(extracted_text, final_to_redact)

            # 5. Synthetic Placeholder Generation & Vault Map
            masked_text, vault_map = self.engine.mask_text(extracted_text, final_merged_entities, reversible=True)

            # Store in PHIVault
            if self.vault and vault_map:
                self.vault.add_mapping(doc_id, vault_map)

            # Build detailed redaction spans and confidence sources
            redaction_spans = []
            confidence_sources = []

            for ent in final_merged_entities:
                orig_text = ent.get('text', '')
                placeholder = vault_map.get(orig_text, "[REDACTED]")
                span_info = {
                    "start": ent.get('start', 0),
                    "end": ent.get('end', 0),
                    "text": orig_text,
                    "label": ent.get('label', 'PHI'),
                    "placeholder": placeholder
                }
                redaction_spans.append(span_info)
                confidence_sources.append({
                    "entity": orig_text,
                    "label": ent.get('label', 'PHI'),
                    "confidence": ent.get('confidence', 0.95),
                    "source": ent.get('source', 'Hybrid')
                })

            elapsed = round(time.time() - start_time, 4)

            return PipelineResult(
                doc_id=doc_id,
                file_name=file_name,
                file_type=file_type,
                extracted_text=extracted_text,
                raw_detected_entities=raw_detected_entities,
                preserved_entities=preserved_entities,
                final_entities=final_merged_entities,
                redaction_spans=redaction_spans,
                synthetic_placeholders=vault_map,
                masked_text=masked_text,
                confidence_sources=confidence_sources,
                warnings_errors=warnings_errors,
                processing_time_sec=elapsed
            )

        finally:
            if cleanup_tmp and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
