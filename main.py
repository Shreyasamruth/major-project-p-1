import os
import sys
import argparse
import tempfile
import base64
from typing import Optional

from engine.pipeline import DeidPipeline, PipelineResult
from engine.redactor import DocumentRedactor
from engine.vault import PHIVault
from utils.db_handler import DBHandler

# Workaround for Windows stdout UTF-8 encoding and PyTorch DLL issues
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["SPACY_USE_GPU"] = "0"
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class HealthcareDeidCLI:
    """
    Local CLI Interface for Healthcare Data De-Identification Engine.
    Executes core pipeline, redaction, vault demasking, and database operations without Streamlit.
    """
    def __init__(self, secret_key: Optional[str] = None):
        if not secret_key:
            secret_key = os.getenv("DEID_SECRET_KEY", "shreyas")
        self.secret_key = secret_key
        self.vault = PHIVault(secret_key=self.secret_key)
        self.db = DBHandler()
        self.pipeline = DeidPipeline(vault=self.vault)
        self.redactor = DocumentRedactor()

    def process_file(self, file_path: str, output_dir: str = ".") -> Optional[str]:
        """Processes and redacts a single document file locally."""
        if not os.path.exists(file_path):
            print(f"[!] Error: File not found at '{file_path}'")
            return None

        file_name = os.path.basename(file_path)
        doc_id = file_name.replace(".", "_")[:12]

        print(f"\n========================================================")
        print(f"[*] Processing: {file_name}")
        print(f"========================================================")

        # 1. Core Pipeline Execution
        print("[*] 1. Running Hybrid PHI Detection & Context Preservation...")
        pipe_res: PipelineResult = self.pipeline.process_document(
            file_path=file_path,
            file_name=file_name,
            doc_id=doc_id
        )

        print(f"   - Extracted Characters : {len(pipe_res.extracted_text)}")
        print(f"   - Document Type        : {pipe_res.file_type.upper()}")
        print(f"   - Processing Time      : {pipe_res.processing_time_sec:.2f}s")
        print(f"   - Raw PHI Detected     : {len(pipe_res.raw_detected_entities)}")
        print(f"   - Preserved Medical    : {len(pipe_res.preserved_entities)}")
        print(f"   - Final Redactions     : {len(pipe_res.final_entities)}")

        if pipe_res.final_entities:
            print("\n[!] Entities Marked for Redaction:")
            for ent in pipe_res.final_entities:
                print(f"   - [{ent.get('label', 'PHI')}] {ent.get('text')} (Src: {ent.get('source', 'Hybrid')})")

        if pipe_res.preserved_entities:
            print("\n[*] Preserved Clinical Information (NOT Redacted):")
            for d in pipe_res.preserved_entities:
                print(f"   - [{d.get('label', 'CLINICAL')}] {d.get('text')}")

        # 2. Redaction & Document Generation
        ext = os.path.splitext(file_name)[1].lower()
        out_file_name = f"redacted_{file_name}"
        out_path = os.path.join(output_dir, out_file_name)

        print(f"\n[*] 2. Applying Document Redactions -> {out_path}...")
        redact_res = self.redactor.redact_document(
            file_path=file_path,
            file_type=pipe_res.file_type,
            entities=pipe_res.final_entities,
            output_path=out_path,
            vault_map=pipe_res.synthetic_placeholders
        )

        # 3. Database Persistence
        print("[*] 3. Registering document state & encrypted vault map in DB...")
        with open(file_path, "rb") as f:
            orig_b64 = base64.b64encode(f.read()).decode("utf-8")

        self.db.add_document(
            doc_id=doc_id,
            title=f"Report: {file_name}",
            masked_text=pipe_res.masked_text,
            disease_data=pipe_res.preserved_entities,
            file_b64=redact_res["file_b64"],
            file_ext=ext,
            original_file_b64=orig_b64,
            original_content=pipe_res.extracted_text,
            status="SHARED"
        )

        print(f"[OK] Success! Redacted file saved: {out_path}")
        print(f"[OK] Vault Reference ID: REF-{doc_id.upper()}\n")
        return out_path

    def list_documents(self):
        """Displays stored documents and lifecycle statuses."""
        docs = self.db.get_all_documents()
        if not docs:
            print("\n[*] No documents found in database.")
            return

        print(f"\n========================================================")
        print(f"STORED DOCUMENTS LIST ({len(docs)} records)")
        print(f"========================================================")
        print(f"{'REF ID':<15} {'STATUS':<18} {'TITLE':<30} {'CREATED'}")
        print("-" * 75)
        for did, d in docs.items():
            print(f"{'REF-' + did.upper():<15} {d.get('status', 'PENDING'):<18} {d.get('title', '')[:28]:<30} {d.get('created_at', '')}")
        print("-" * 75)

    def doctor_review(self):
        """Doctor Portal review workflow to submit clinical updates."""
        queue = self.db.get_doctor_queue()
        if not queue:
            print("\n[*] Doctor Queue: No pending documents awaiting review.")
            return

        print(f"\n========================================================")
        print(f"DOCTOR CLINICAL REVIEW DESK ({len(queue)} pending)")
        print(f"========================================================")
        for idx, (did, d) in enumerate(queue.items(), 1):
            print(f"{idx}. REF-{did.upper()} | {d.get('title')} | Status: {d.get('status')}")

        try:
            choice = int(input("\nSelect document number to review (or 0 to cancel): "))
            if choice == 0 or choice > len(queue):
                return
            selected_did = list(queue.keys())[choice - 1]
            doc_data = queue[selected_did]

            print(f"\n--- Reviewing REF-{selected_did.upper()} ---")
            print("[*] Preserved Clinical Findings:")
            for item in doc_data.get("disease_data", []):
                print(f"   - [{item.get('label', 'CLINICAL')}] {item.get('text')}")

            if doc_data.get("masked_text"):
                print(f"\n[*] Masked Text Preview:\n{doc_data['masked_text'][:300]}...")

            notes = input("\nEnter Doctor's Clinical Diagnosis / Update notes: ")
            if notes.strip():
                self.db.update_doctor_notes(selected_did, notes.strip())
                print(f"[OK] Submitted clinical update for REF-{selected_did.upper()}!")
        except ValueError:
            print("Invalid selection.")

    def admin_demask(self):
        """Admin Authorized Demasking workflow."""
        docs = self.db.get_all_documents()
        reviewed = {did: d for did, d in docs.items() if d.get('status') == "DOCTOR_REVIEWED"}

        if not reviewed:
            print("\n[*] Admin Desk: No reviewed documents awaiting demasking.")
            return

        print(f"\n========================================================")
        print(f"ADMIN DEMASKING DESK ({len(reviewed)} ready)")
        print(f"========================================================")
        for idx, (did, d) in enumerate(reviewed.items(), 1):
            print(f"{idx}. REF-{did.upper()} | {d.get('title')} | Notes: {d.get('doctor_notes')[:40]}")

        try:
            choice = int(input("\nSelect document number to demask (or 0 to cancel): "))
            if choice == 0 or choice > len(reviewed):
                return
            selected_did = list(reviewed.keys())[choice - 1]
            doc_data = reviewed[selected_did]

            entered_key = input("Enter Admin De-ID Secret Key: ")
            demasked = self.vault.demask_text(doc_data.get("masked_text", ""), selected_did, provided_key=entered_key)

            if demasked.startswith("ERROR"):
                self.db.log_audit("ADMIN_DEMASK_FAILED", selected_did, role="Admin", details="Invalid secret key attempt")
                print(f"[!] {demasked}")
            else:
                self.db.log_audit("ADMIN_DEMASK_SUCCESS", selected_did, role="Admin", details="Restored original PHI text")
                self.db.update_document_status(selected_did, "COMPLETED", "Admin demasking completed")
                print("\n[OK] DEMASKED SUCCESSFULLY! Restored Original PHI Text:")
                print("--------------------------------------------------------")
                print(demasked)
                print("--------------------------------------------------------")
        except ValueError:
            print("Invalid selection.")

    def print_audit_logs(self):
        """Prints HIPAA audit trail events."""
        logs = self.db.get_audit_logs()
        if not logs:
            print("\n[*] No audit events recorded.")
            return

        print(f"\n========================================================")
        print(f"HIPAA AUDIT TRAIL LOGS ({len(logs)} events)")
        print(f"========================================================")
        print(f"{'TIMESTAMP':<20} {'ACTION':<24} {'DOC ID':<12} {'ROLE':<10} {'DETAILS'}")
        print("-" * 80)
        for l in logs[:20]:
            print(f"{l['timestamp']:<20} {l['action']:<24} {l['doc_id']:<12} {l['role']:<10} {l['details']}")
        print("-" * 80)

    def interactive_menu(self):
        """Interactive console menu loop."""
        while True:
            print("\n========================================================")
            print("HEALTHCARE DE-IDENTIFICATION ENGINE (LOCAL CLI)")
            print("========================================================")
            print("1. Process & Redact Document File (TXT, PDF, PNG, JPG)")
            print("2. List Stored Documents & Statuses")
            print("3. Doctor Clinical Review Desk")
            print("4. Admin Authorized Demasking Desk")
            print("5. View HIPAA Audit Logs")
            print("6. Exit")

            choice = input("\nSelect Option [1-6]: ").strip()
            if choice == "1":
                file_path = input("Enter path to input document file: ").strip().strip('"').strip("'")
                if file_path:
                    self.process_file(file_path)
            elif choice == "2":
                self.list_documents()
            elif choice == "3":
                self.doctor_review()
            elif choice == "4":
                self.admin_demask()
            elif choice == "5":
                self.print_audit_logs()
            elif choice == "6":
                print("\nGoodbye!")
                break
            else:
                print("Invalid option. Please enter 1-6.")

def main():
    parser = argparse.ArgumentParser(description="Healthcare Data De-Identification Local CLI Engine")
    parser.add_argument("--file", type=str, help="Path to input document (TXT, PDF, PNG, JPG)")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for redacted files")
    parser.add_argument("--list", action="store_true", help="List stored documents in database")
    parser.add_argument("--audit", action="store_true", help="Print HIPAA compliance audit logs")
    parser.add_argument("--demask", type=str, help="Document Ref ID to demask")
    parser.add_argument("--key", type=str, help="Admin De-ID secret key for demasking")

    args = parser.parse_args()
    cli = HealthcareDeidCLI()

    if args.file:
        cli.process_file(args.file, output_dir=args.outdir)
    elif args.list:
        cli.list_documents()
    elif args.audit:
        cli.print_audit_logs()
    elif args.demask:
        key = args.key or input("Enter Admin De-ID secret key: ")
        doc = cli.db.get_document(args.demask)
        if not doc:
            print(f"Document {args.demask} not found.")
        else:
            demasked = cli.vault.demask_text(doc.get("masked_text", ""), args.demask, provided_key=key)
            print(demasked)
    else:
        cli.interactive_menu()

if __name__ == "__main__":
    main()
