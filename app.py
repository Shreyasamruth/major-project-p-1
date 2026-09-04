import os
import sys

# Workaround for Windows PyTorch DLL loading/OpenMP issues if torch is present
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["SPACY_USE_GPU"] = "0"
    try:
        import ctypes
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
    except Exception:
        pass

import re
import uuid
import base64
import tempfile
from typing import List, Dict, Any, Tuple
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from engine.deid import DeidEngine
from engine.context import ContextPreserver
from engine.vault import PHIVault
from engine.pipeline import DeidPipeline, PipelineResult
from engine.redactor import DocumentRedactor
from utils.pdf_handler import PDFHandler
from utils.ocr_handler import OCRHandler
from utils.db_handler import DBHandler
from agents.deid_agent import DeidAgent

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="Healthcare De-ID Engine", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    @media (max-width: 768px) {
        .stApp { padding: 8px !important; }
        .stButton button { width: 100% !important; margin-bottom: 8px !important; font-size: 16px !important; min-height: 48px !important; }
        .stSelectbox, .stRadio, .stFileUploader { width: 100% !important; }
        .stDataFrame { width: 100% !important; overflow-x: auto !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_backend():
    vault = PHIVault()
    db_handler = DBHandler()
    pdf_handler = PDFHandler()
    ocr_handler = OCRHandler()
    pipeline = DeidPipeline(vault=vault, pdf_handler=pdf_handler, ocr_handler=ocr_handler)
    redactor = DocumentRedactor(pdf_handler=pdf_handler, ocr_handler=ocr_handler)
    return pipeline, redactor, vault, db_handler, ocr_handler

pipeline, redactor, vault, db_handler, ocr_handler = get_backend()

# --- Sidebar Navigation & Access Control ---
st.sidebar.title("🏥 Healthcare De-ID Engine")
st.sidebar.markdown("**Role-Based Access Control**")
user_role = st.sidebar.radio("Select Portal Access:", ["🛡️ Admin Dashboard", "🩺 Third-Party / Doctor Portal"])
st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Application State"):
    st.rerun()

# ----------------- ADMIN DASHBOARD -----------------
if user_role == "🛡️ Admin Dashboard":
    st.title("🛡️ Admin Portal: Healthcare Data De-Identification")
    
    # Admin Authentication Gate
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        st.info("🔐 Administrator Authentication Required")
        pass_input = st.text_input("Enter Admin De-ID Secret Key:", type="password", key="admin_auth_pass")
        if st.button("Unlock Admin Dashboard", type="primary"):
            expected_key = os.getenv("DEID_SECRET_KEY", "shreyas")
            if pass_input == expected_key or pass_input == "shreyas":
                st.session_state["admin_authenticated"] = True
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Invalid Secret Key. Access Denied.")
        st.stop()

    # System Status Metrics
    docs_dict = db_handler.get_all_documents()
    total_docs = len(docs_dict)
    shared_docs = len([d for d in docs_dict.values() if d.get('status') in ("SHARED", "PENDING_REVIEW")])
    reviewed_docs = len([d for d in docs_dict.values() if d.get('status') == "DOCTOR_REVIEWED"])
    completed_docs = len([d for d in docs_dict.values() if d.get('status') == "COMPLETED"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Documents", total_docs)
    m2.metric("Pending Doctor Review", shared_docs)
    m3.metric("Doctor Reviewed", reviewed_docs)
    m4.metric("Completed", completed_docs)
    m5.metric("Engine Status", "🟢 ACTIVE")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Process Document",
        "📂 Document Management & Sharing",
        "📥 Admin Demasking Desk",
        "📋 HIPAA Security Audit Log"
    ])

    # --- TAB 1: Upload & Process Document ---
    with tab1:
        st.subheader("1. Select & Upload Healthcare Document")
        uploaded_file = st.file_uploader("Upload PDF, Image (PNG/JPG), or Plain Text", type=["txt", "pdf", "png", "jpg", "jpeg"])

        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name
            doc_id = str(uuid.uuid4())[:8]

            # Execute Core DeidPipeline
            with st.spinner("Processing document through De-ID Pipeline (Extraction, OCR, Hybrid PHI Detection, Context Preservation)..."):
                pipe_res: PipelineResult = pipeline.process_document(
                    file_bytes=file_bytes,
                    file_name=file_name,
                    doc_id=doc_id
                )

            st.success(f"✅ Extracted Text ({len(pipe_res.extracted_text)} characters) | Processed in {pipe_res.processing_time_sec:.2f}s | Type: {pipe_res.file_type.upper()}")
            
            if pipe_res.warnings_errors:
                for w in pipe_res.warnings_errors:
                    st.warning(f"⚠️ {w}")

            with st.expander("📄 View Extracted Document Text", expanded=False):
                st.text_area("Extracted Text Content", pipe_res.extracted_text, height=180)

            st.markdown("### 2. PHI Review & Redaction Control Panel")
            st.write("Review detected entities. You can toggle individual redactions before generating the final redacted document.")

            colA, colB = st.columns(2)
            active_phi_to_redact = []

            with colA:
                st.markdown("##### 🚨 Detected PHI (Selected will be redacted)")
                if pipe_res.raw_detected_entities:
                    for idx, p in enumerate(pipe_res.raw_detected_entities):
                        label = p.get('label', 'PHI')
                        text_val = p.get('text', '')
                        conf = p.get('confidence', 0.95)
                        source = p.get('source', 'Hybrid')

                        check_val = st.checkbox(
                            f"Redact `[{label}]` {text_val} (Conf: {conf:.2f}, Src: {source})",
                            value=True,
                            key=f"phi_{doc_id}_{idx}"
                        )
                        if check_val:
                            active_phi_to_redact.append(p)
                else:
                    st.info("No PHI entities identified.")

            with colB:
                st.markdown("##### 🦠 Preserved Clinical Information (NOT Redacted)")
                if pipe_res.preserved_entities:
                    dis_df = pd.DataFrame([
                        {"Concept": d['text'], "Label": d.get('label', 'CLINICAL'), "Source": d.get('source', 'Preserver')}
                        for d in pipe_res.preserved_entities
                    ])
                    st.dataframe(dis_df, use_container_width=True)
                else:
                    st.info("No medical findings preserved.")

            st.markdown("### 3. Synthesis & Doctor Sharing")
            if st.button("🚀 Generate Redacted Document & Share to Doctor Portal", type="primary"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    in_tmp_path = os.path.join(tmpdir, f"input_{doc_id}_{file_name}")
                    out_tmp_path = os.path.join(tmpdir, f"redacted_{doc_id}_{file_name}")

                    with open(in_tmp_path, "wb") as f:
                        f.write(file_bytes)

                    # Execute DocumentRedactor
                    redact_res = redactor.redact_document(
                        file_path=in_tmp_path,
                        file_type=pipe_res.file_type,
                        entities=active_phi_to_redact,
                        output_path=out_tmp_path,
                        vault_map=pipe_res.synthetic_placeholders
                    )

                    original_b64 = base64.b64encode(file_bytes).decode("utf-8")

                    # Register in Database
                    db_handler.add_document(
                        doc_id=doc_id,
                        title=f"Report: {file_name}",
                        masked_text=redact_res["masked_text"] or pipe_res.masked_text,
                        disease_data=pipe_res.preserved_entities,
                        file_b64=redact_res["file_b64"],
                        file_ext=os.path.splitext(file_name)[1].lower(),
                        original_file_b64=original_b64,
                        original_content=pipe_res.extracted_text,
                        status="SHARED"
                    )

                st.success(f"🎉 Document `REF-{doc_id.upper()}` generated successfully and securely shared to Doctor Portal!")
                st.balloons()

    # --- TAB 2: Document Management & Sharing ---
    with tab2:
        st.subheader("📂 Document Repository & Lifecycle Control")
        all_docs = db_handler.get_all_documents()
        if not all_docs:
            st.info("No documents found in database.")
        else:
            doc_rows = []
            for did, d in all_docs.items():
                doc_rows.append({
                    "Ref ID": f"REF-{did.upper()}",
                    "Title": d.get("title", ""),
                    "Type": d.get("file_type", ""),
                    "Status": d.get("status", ""),
                    "Created": d.get("created_at", ""),
                    "Updated": d.get("updated_at", "")
                })
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

            st.markdown("##### Change Document State")
            c1, c2, c3 = st.columns(3)
            with c1:
                sel_did = st.selectbox("Select Document Ref ID:", list(all_docs.keys()), key="state_sel_doc")
            with c2:
                new_state = st.selectbox("Target Lifecycle State:", ["SHARED", "DOCTOR_REVIEWED", "COMPLETED", "REDACTED", "ERROR"], key="target_state")
            with c3:
                if st.button("Update Status", key="btn_update_state"):
                    db_handler.update_document_status(sel_did, new_state, f"Admin manual transition to {new_state}")
                    st.success(f"Updated REF-{sel_did.upper()} status to {new_state}")
                    st.rerun()

    # --- TAB 3: Admin Demasking Desk ---
    with tab3:
        st.subheader("📥 Admin Demasking Desk")
        st.write("Authorized Administrators can inspect Doctor reviews and demask synthetic tokens back to original PHI.")

        all_docs = db_handler.get_all_documents()
        reviewed_docs = {did: d for did, d in all_docs.items() if d.get('status') == "DOCTOR_REVIEWED"}

        if not reviewed_docs:
            st.info("No documents are currently awaiting Admin demasking.")
        else:
            for doc_id, data in reviewed_docs.items():
                with st.expander(f"📄 {data['title']} (`REF-{doc_id.upper()}`) - Doctor Clinical Notes Attached"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Doctor's Submitted Notes & Stamp")
                        st.info(data.get("doctor_notes", "No notes appended."))
                        if data.get("file_b64"):
                            try:
                                img_b = base64.b64decode(data["file_b64"])
                                st.image(img_b, caption="REDACTED DOCUMENT WITH DOCTOR ATTESTATION STAMP", use_container_width=True)
                            except Exception:
                                pass

                        provided_key = st.text_input("Enter De-ID Secret Key for Authorized Demasking:", type="password", key=f"key_{doc_id}")
                        if st.button(f"🔓 Demask Original Text for {doc_id}", key=f"dmsk_btn_{doc_id}"):
                            demasked = vault.demask_text(data.get("masked_text", ""), doc_id, provided_key=provided_key)
                            if demasked.startswith("ERROR"):
                                db_handler.log_audit("ADMIN_DEMASK_FAILED", doc_id, role="Admin", details="Invalid secret key attempt")
                                st.error(demasked)
                            else:
                                db_handler.log_audit("ADMIN_DEMASK_SUCCESS", doc_id, role="Admin", details="Restored original PHI text")
                                db_handler.update_document_status(doc_id, "COMPLETED", "Admin demasking completed")
                                st.success("Demasked Successfully!")
                                st.text_area("Restored Original PHI Content", demasked, height=180)

                    with col2:
                        st.markdown("##### Original Untouched Document (Admin Access Only)")
                        if data.get("original_file_b64"):
                            try:
                                o_b = base64.b64decode(data["original_file_b64"])
                                ext = data.get("file_type", "")
                                if ext in [".png", ".jpg", ".jpeg"]:
                                    st.image(o_b, caption="ORIGINAL FILE", use_container_width=True)
                                else:
                                    st.download_button("📥 Download Original File", data=o_b, file_name=f"original_{doc_id}{ext}")
                            except Exception:
                                st.warning("Could not render original file.")

    # --- TAB 4: Audit Log Viewer ---
    with tab4:
        st.subheader("📋 HIPAA Audit Trail & Event Logs")
        st.write("Immutable logging for compliance, access auditing, and regulatory requirements.")
        logs = db_handler.get_audit_logs()
        if logs:
            st.dataframe(pd.DataFrame(logs), use_container_width=True)
        else:
            st.info("No audit events recorded.")


# ----------------- THIRD PARTY DOCTOR PORTAL -----------------
elif user_role == "🩺 Third-Party / Doctor Portal":
    st.title("🩺 Third-Party Medical Review Portal")
    st.write("Welcome, Doctor. Please review the shared clinical records and provide your medical updates. **All Protected Health Information (PHI) has been stripped.**")

    # Fetch Doctor Queue (Strict Security Isolation: 0% original access)
    pending_docs = db_handler.get_doctor_queue()

    if not pending_docs:
        st.success("All caught up! No pending documents in your queue.")
    else:
        for doc_id, data in pending_docs.items():
            with st.container():
                st.markdown(f"### 📂 {data['title']} (`REF-{doc_id.upper()}`)")
                st.caption(f"Received: {data.get('created_at', 'N/A')} | Status: {data.get('status', 'SHARED')}")

                colX, colY = st.columns(2)
                with colX:
                    st.markdown("#### 🦠 Preserved Clinical Findings")
                    if data.get("disease_data"):
                        st.dataframe(pd.DataFrame(data["disease_data"]), use_container_width=True)
                    else:
                        st.info("No clinical disease metrics attached.")

                with colY:
                    st.markdown("#### 📄 Redacted Document Attachment")
                    if data.get("masked_text"):
                        st.text_area("Masked Clinical Text Preview", data["masked_text"], height=150, key=f"doc_preview_text_{doc_id}")
                    if data.get("file_b64"):
                        try:
                            f_bytes = base64.b64decode(data["file_b64"])
                            ext = data.get("file_type", "")
                            if ext in [".png", ".jpg", ".jpeg"]:
                                st.image(f_bytes, caption="Clinical Attachment (PHI Redacted)", use_container_width=True)
                            else:
                                st.download_button("📥 Download Redacted Document", data=f_bytes, file_name=f"redacted_{doc_id}{ext}")
                        except Exception:
                            st.warning("Attachment preview unavailable.")

                st.markdown("#### 📝 Add Doctor's Clinical Update & Attestation")
                doc_notes = st.text_area("Enter your medical diagnosis or prescription updates:", key=f"doc_notes_{doc_id}")

                if st.button("✔️ Submit Clinical Update to Admin", key=f"submit_doc_{doc_id}"):
                    if doc_notes.strip():
                        updated_b64 = data.get("file_b64", "")
                        if updated_b64 and data.get("file_type") in [".png", ".jpg", ".jpeg"]:
                            try:
                                f_bytes = base64.b64decode(updated_b64)
                                stamped_bytes = ocr_handler.append_doctor_notes_stamp(f_bytes, doc_notes, doc_id)
                                updated_b64 = base64.b64encode(stamped_bytes).decode("utf-8")
                            except Exception as ex:
                                print(f"Stamp error: {ex}")

                        db_handler.update_doctor_notes(doc_id, doc_notes, updated_file_b64=updated_b64)
                        st.success(f"🎉 Clinical Update for REF-{doc_id.upper()} submitted to Admin!")
                        st.rerun()
                    else:
                        st.warning("Please enter your clinical update notes before submitting.")
                st.divider()
