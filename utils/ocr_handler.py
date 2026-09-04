import cv2
import numpy as np
import os

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

class OCRHandler:
    def __init__(self):
        pass

    def run_ocr(self, image_path: str) -> dict:
        """Runs local Tesseract OCR on an image if installed, returning text and bounding boxes."""
        if not HAS_PYTESSERACT or not os.path.exists(image_path):
            return {"text": "", "boxes": []}
        try:
            img = cv2.imread(image_path)
            if img is None: return {"text": "", "boxes": []}
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            text_lines = []
            boxes = []
            h, w = img.shape[:2]
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30 and data['text'][i].strip():
                    text_lines.append(data['text'][i])
                    x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    boxes.append({
                        "text": data['text'][i],
                        "confidence": float(data['conf'][i]) / 100.0,
                        "box_2d": [
                            int((y / float(h)) * 1000),
                            int((x / float(w)) * 1000),
                            int(((y + bh) / float(h)) * 1000),
                            int(((x + bw) / float(w)) * 1000)
                        ]
                    })
            return {"text": " ".join(text_lines), "boxes": boxes}
        except Exception as e:
            print(f"PyTesseract OCR warning: {e}")
            return {"text": "", "boxes": []}

    def enhance_image(self, image_path: str) -> str:
        """Applies CLAHE adaptive contrast enhancement and deskewing for low-quality medical scans."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # 1. Convert to LAB color space for adaptive contrast enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # 2. Basic deskewing based on minAreaRect
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 50:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                elif angle > 45:
                    angle = 90 - angle
                if 0.5 < abs(angle) < 30:
                    (h, w) = enhanced.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    enhanced = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            out_path = f"enhanced_{os.path.basename(image_path)}"
            cv2.imwrite(out_path, enhanced)
            return out_path
        except Exception as e:
            print(f"Enhance image warning ({e}), falling back to original.")
            return image_path
        
    def apply_redactions_to_image(self, image_path: str, phi_entities: list, output_path: str, vault_map: dict = None) -> str:
        """Applies synthetic placeholder overlays directly onto redacted regions of an image."""
        try:
            img = cv2.imread(image_path)
            if img is None: return image_path
            h, w = img.shape[:2]
            
            reverse_map = {}
            if vault_map:
                for synth, orig in vault_map.items():
                    if isinstance(orig, str):
                        reverse_map[orig.lower().strip()] = synth
                        
            counts = {}
            # Run OCR once to get word coordinates for fallback matching
            ocr_boxes = []
            
            for ent in phi_entities:
                orig_text = ent.get('text', '').strip()
                label = ent.get('label', 'PHI').upper()
                
                replacement = reverse_map.get(orig_text.lower(), None)
                if not replacement:
                    clean_label = label.replace("PERSON", "PATIENT_NAME").replace("GPE", "LOCATION")
                    counts[clean_label] = counts.get(clean_label, 0) + 1
                    replacement = f"[{clean_label}_{counts[clean_label]}]"

                # Fallback matching if box_2d is missing
                if ('box_2d' not in ent or not ent['box_2d']) and orig_text:
                    if not ocr_boxes:
                        ocr_boxes = self.run_ocr(image_path).get("boxes", [])
                    matched_y1, matched_x1, matched_y2, matched_x2 = 1000, 1000, 0, 0
                    found_match = False
                    words_in_ent = [w.lower() for w in orig_text.split() if len(w) > 1]
                    for ob in ocr_boxes:
                        ob_text = ob.get("text", "").lower()
                        if any(w in ob_text or ob_text in w for w in words_in_ent):
                            box = ob.get("box_2d")
                            if box and len(box) == 4:
                                matched_y1 = min(matched_y1, box[0])
                                matched_x1 = min(matched_x1, box[1])
                                matched_y2 = max(matched_y2, box[2])
                                matched_x2 = max(matched_x2, box[3])
                                found_match = True
                    if found_match:
                        ent['box_2d'] = [matched_y1, matched_x1, matched_y2, matched_x2]

                if 'box_2d' in ent and isinstance(ent['box_2d'], list) and len(ent['box_2d']) == 4:
                    ymin, xmin, ymax, xmax = ent['box_2d']
                    y1 = int((ymin / 1000.0) * h)
                    x1 = int((xmin / 1000.0) * w)
                    y2 = int((ymax / 1000.0) * h)
                    x2 = int((xmax / 1000.0) * w)
                    
                    # 1. Draw crisp white background box over sensitive text
                    pad = 2
                    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
                    bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (180, 180, 180), 1)
                    
                    # 2. Calculate optimal font scale to render replacement text nicely inside box
                    box_w = max(10, bx2 - bx1)
                    box_h = max(10, by2 - by1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    thickness = 1
                    
                    (text_w, text_h), baseline = cv2.getTextSize(replacement, font, font_scale, thickness)
                    if text_w > box_w - 4 and box_w > 4:
                        font_scale = max(0.20, font_scale * ((box_w - 4) / float(text_w)))
                        (text_w, text_h), baseline = cv2.getTextSize(replacement, font, font_scale, thickness)

                    text_x = bx1 + max(2, (box_w - text_w) // 2)
                    text_y = by1 + max(text_h, (box_h + text_h) // 2) - 1
                    cv2.putText(img, replacement, (text_x, min(by2 - 2, text_y)), font, font_scale, (180, 0, 0), thickness, cv2.LINE_AA)
                    
            cv2.imwrite(output_path, img)
            return output_path
        except Exception as e:
            print(f"Error drawing redactions on image: {e}")
            return image_path
        
    def generate_disease_image(self, disease_data: list, doc_id: str, output_path: str) -> str:
        """Generates a clean clinical report image displaying only Disease/Clinical info under Reference ID."""
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        
        # Draw Dark Navy Banner
        cv2.rectangle(img, (0, 0), (800, 90), (120, 50, 15), -1)
        cv2.putText(img, f"CLINICAL RECORD | REF: {doc_id.upper()}", (30, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, "ALL PATIENT PHI & HEADER METADATA PHYSICALLY REMOVED", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
        
        y = 140
        if not disease_data:
            cv2.putText(img, "No specific disease data was extracted.", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            cv2.putText(img, "EXTRACTED CLINICAL FINDINGS:", (40, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
            y += 40
            for d in disease_data:
                label_name = d.get('label', 'DISEASE').upper()
                finding_text = d.get('text', '')
                text = f"* [{label_name}]: {finding_text}"
                words = text.split(" ")
                line = ""
                for word in words:
                    if len(line) + len(word) > 65:
                        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
                        y += 30
                        line = "   " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
                    y += 40
                if y > 940: break
                    
        cv2.imwrite(output_path, img)
        return output_path

    def extract_clinical_only_image(self, image_path: str, phi_entities: list, disease_data: list, doc_id: str, output_path: str) -> str:
        """Removes the entire PHI header box and footer from the document image, displaying only the Reference ID and Clinical/Disease content."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return self.generate_disease_image(disease_data, doc_id, output_path)
                
            h, w = img.shape[:2]
            
            # 1. Find maximum ymax among header PHI entities (Patient Name, ID, Date, Hospital Header)
            phi_ymax_list = []
            for ent in phi_entities:
                if 'box_2d' in ent and isinstance(ent['box_2d'], list) and len(ent['box_2d']) == 4:
                    if ent['box_2d'][2] < 700: # Exclude footer entities
                        phi_ymax_list.append(ent['box_2d'][2])
                    
            if phi_ymax_list:
                # Crop starting below highest header PHI entity
                header_cutoff_norm = min(max(phi_ymax_list) + 12, 450)
                crop_start_y = int((header_cutoff_norm / 1000.0) * h)
            else:
                crop_start_y = int(0.24 * h) # Default crop below header box (~24%)
                
            # 2. Crop ending above signature block if present
            footer_ymin_list = [ent['box_2d'][0] for ent in phi_entities if 'box_2d' in ent and isinstance(ent['box_2d'], list) and len(ent['box_2d']) == 4 and ent['box_2d'][0] > 750]
            if footer_ymin_list:
                crop_end_y = int((min(footer_ymin_list) / 1000.0) * h)
            else:
                crop_end_y = h

            cropped_clinical = img[crop_start_y:crop_end_y, 0:w]
            if cropped_clinical.size == 0 or cropped_clinical.shape[0] < 50:
                return self.generate_disease_image(disease_data, doc_id, output_path)
                
            # 3. Build canvas with sleek dark blue Reference ID banner
            ch, cw = cropped_clinical.shape[:2]
            banner_h = 85
            total_h = banner_h + ch + 20
            canvas = np.ones((total_h, cw, 3), dtype=np.uint8) * 255
            
            # Navy blue banner
            cv2.rectangle(canvas, (0, 0), (cw, banner_h), (120, 50, 15), -1)
            cv2.putText(canvas, f"CLINICAL RECORD | REFERENCE CODE: REF-{doc_id.upper()}", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(canvas, "HEADER & PATIENT IDENTIFIERS PHYSICALLY REMOVED", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
            
            # Place cropped clinical findings below header banner
            canvas[banner_h+10:banner_h+10+ch, 0:cw] = cropped_clinical
            
            cv2.imwrite(output_path, canvas)
            return output_path
        except Exception as e:
            print(f"Error in extract_clinical_only_image: {e}")
            return self.generate_disease_image(disease_data, doc_id, output_path)

    def append_doctor_notes_stamp(self, image_bytes: bytes, doctor_notes: str, doc_id: str) -> bytes:
        """Appends an official Doctor's Clinical Update & Attestation block onto the clinical report image cleanly without ugly black stamps."""
        try:
            from datetime import datetime
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return image_bytes

            h, w = img.shape[:2]
            
            # Calculate text line wrapping
            stamp_banner_h = 42
            lines = []
            note_text = f"CLINICAL DIAGNOSIS / UPDATE: {doctor_notes.strip()}"
            words = note_text.split(" ")
            curr_line = ""
            max_chars = max(38, int(w / 12))
            
            for word in words:
                if len(curr_line) + len(word) > max_chars:
                    lines.append(curr_line)
                    curr_line = word + " "
                else:
                    curr_line += word + " "
            if curr_line.strip():
                lines.append(curr_line)
                
            block_h = stamp_banner_h + (len(lines) * 28) + 35
            total_h = h + block_h
            
            canvas = np.ones((total_h, w, 3), dtype=np.uint8) * 255
            canvas[0:h, 0:w] = img
            
            # Draw clean forest green doctor attestation container
            stamp_top = h + 8
            cv2.rectangle(canvas, (10, stamp_top), (w - 10, total_h - 10), (245, 252, 245), -1)
            cv2.rectangle(canvas, (10, stamp_top), (w - 10, total_h - 10), (34, 139, 34), 2)
            
            # Header banner inside block
            cv2.rectangle(canvas, (10, stamp_top), (w - 10, stamp_top + stamp_banner_h), (34, 139, 34), -1)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(canvas, f"PHYSICIAN ATTESTATION & CLINICAL UPDATE | REF: {doc_id.upper()}", (20, stamp_top + 26), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)
            
            # Render doctor notes cleanly inside green container
            y_pos = stamp_top + stamp_banner_h + 28
            for line in lines:
                cv2.putText(canvas, line.strip(), (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 50, 20), 1, cv2.LINE_AA)
                y_pos += 26
                
            success, encoded_img = cv2.imencode(".png", canvas)
            if success:
                return encoded_img.tobytes()
            return image_bytes
        except Exception as e:
            print(f"Error appending doctor notes stamp: {e}")
            return image_bytes
