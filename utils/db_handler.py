import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List

from contextlib import contextmanager

class DBHandler:
    def __init__(self, db_path: str = "deid_system.db", legacy_json_path: str = "shared_db.json"):
        self.db_path = db_path
        self.legacy_json_path = legacy_json_path
        self._init_db()
        self._migrate_legacy_json()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initializes SQLite schema for documents, audit logs, and system state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    file_type TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    original_content TEXT,
                    masked_text TEXT,
                    disease_data TEXT,
                    file_b64 TEXT,
                    original_file_b64 TEXT,
                    doctor_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT DEFAULT ''
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'System',
                    details TEXT DEFAULT ''
                )
            """)
            conn.commit()

    def _migrate_legacy_json(self):
        """Migrates legacy shared_db.json records into SQLite if present."""
        if os.path.exists(self.legacy_json_path):
            try:
                with open(self.legacy_json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return
                    legacy_data = json.loads(content)

                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    # Migrate audit logs
                    for log in legacy_data.get("_audit_logs", []):
                        cursor.execute("""
                            INSERT INTO audit_logs (timestamp, action, doc_id, role, details)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            log.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            log.get("action", "UNKNOWN"),
                            log.get("doc_id", "N/A"),
                            log.get("role", "Admin"),
                            log.get("details", "")
                        ))

                    # Migrate documents
                    for doc_id, doc in legacy_data.items():
                        if doc_id.startswith("_"):
                            continue
                        disease_json = json.dumps(doc.get("disease_data", [])) if isinstance(doc.get("disease_data"), list) else "[]"
                        raw_status = doc.get("status", "PENDING")
                        status_map = {"Pending Review": "SHARED", "Reviewed": "DOCTOR_REVIEWED"}
                        norm_status = status_map.get(raw_status, raw_status.upper())

                        cursor.execute("""
                            INSERT OR IGNORE INTO documents (
                                doc_id, title, file_type, status, original_content, masked_text,
                                disease_data, file_b64, original_file_b64, doctor_notes, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc_id,
                            doc.get("title", f"Report: {doc_id}"),
                            doc.get("file_ext", ""),
                            norm_status,
                            "",
                            doc.get("masked_text", ""),
                            disease_json,
                            doc.get("file_b64", ""),
                            doc.get("original_file_b64", ""),
                            doc.get("doctor_notes", ""),
                            doc.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            doc.get("updated_at", "")
                        ))
                    conn.commit()
            except Exception as e:
                print(f"Warning during legacy DB migration: {e}")

    def log_audit(self, action: str, doc_id: str, role: str = "Admin", details: str = ""):
        """Records a HIPAA-compliant audit trail event."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (timestamp, action, doc_id, role, details)
                VALUES (?, ?, ?, ?, ?)
            """, (now, action, doc_id, role, details))
            conn.commit()

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """Retrieves all audit logs ordered by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, action, doc_id, role, details FROM audit_logs ORDER BY id DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_document(
        self,
        doc_id: str,
        title: str,
        masked_text: str,
        disease_data: list,
        file_b64: str = "",
        file_ext: str = "",
        original_file_b64: str = "",
        original_content: str = "",
        status: str = "SHARED"
    ):
        """Adds a new processed document into the database."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        disease_json = json.dumps(disease_data) if isinstance(disease_data, list) else "[]"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    doc_id, title, file_type, status, original_content, masked_text,
                    disease_data, file_b64, original_file_b64, doctor_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, title, file_ext, status, original_content, masked_text,
                disease_json, file_b64, original_file_b64, "", now, now
            ))
            conn.commit()

        self.log_audit("DOCUMENT_CREATED", doc_id, role="Admin", details=f"Title: {title} | Status: {status}")

    def update_document_status(self, doc_id: str, status: str, details: str = ""):
        """Updates document processing status."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?", (status, now, doc_id))
            conn.commit()
        self.log_audit(f"STATUS_UPDATED_{status}", doc_id, role="Admin", details=details)

    def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Retrieves all documents formatted as a dictionary keyed by doc_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            rows = cursor.fetchall()
            
            result = {}
            for row in rows:
                row_dict = dict(row)
                try:
                    row_dict["disease_data"] = json.loads(row_dict["disease_data"]) if row_dict["disease_data"] else []
                except Exception:
                    row_dict["disease_data"] = []
                result[row_dict["doc_id"]] = row_dict
            return result

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Retrieves a single document by doc_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                try:
                    res["disease_data"] = json.loads(res["disease_data"]) if res["disease_data"] else []
                except Exception:
                    res["disease_data"] = []
                return res
            return {}

    def update_doctor_notes(self, doc_id: str, notes: str, updated_file_b64: str = ""):
        """Updates clinical notes submitted by Doctor and sets status to DOCTOR_REVIEWED."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if updated_file_b64:
                cursor.execute("""
                    UPDATE documents
                    SET doctor_notes = ?, file_b64 = ?, status = 'DOCTOR_REVIEWED', updated_at = ?
                    WHERE doc_id = ?
                """, (notes, updated_file_b64, now, doc_id))
            else:
                cursor.execute("""
                    UPDATE documents
                    SET doctor_notes = ?, status = 'DOCTOR_REVIEWED', updated_at = ?
                    WHERE doc_id = ?
                """, (notes, now, doc_id))
            conn.commit()

        self.log_audit("DOCTOR_NOTE_ADDED", doc_id, role="Doctor", details="Doctor submitted clinical update")

    def get_doctor_queue(self) -> Dict[str, Dict[str, Any]]:
        """
        Role Security Isolation: Retrieves only documents with status 'SHARED' or 'PENDING_REVIEW'.
        Strictly strips original_content and original_file_b64 so Doctor role cannot access unredacted PHI.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, title, file_type, status, masked_text, disease_data, file_b64, doctor_notes, created_at, updated_at
                FROM documents
                WHERE status IN ('SHARED', 'PENDING_REVIEW', 'SHARED_TO_DOCTOR')
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            
            result = {}
            for row in rows:
                row_dict = dict(row)
                # Enforce zero original access
                row_dict["original_content"] = ""
                row_dict["original_file_b64"] = ""
                try:
                    row_dict["disease_data"] = json.loads(row_dict["disease_data"]) if row_dict["disease_data"] else []
                except Exception:
                    row_dict["disease_data"] = []
                result[row_dict["doc_id"]] = row_dict
            return result



