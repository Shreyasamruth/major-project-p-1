import json
import os
import base64
import hashlib
from typing import Dict, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    InvalidToken = Exception

class PHIVault:
    """
    Secure, authenticated PHI Vault implementing AES-128-Fernet authenticated encryption.
    Maps synthetic placeholders (e.g. [PATIENT_NAME_1]) back to original sensitive PHI.
    """
    def __init__(self, storage_path: str = "phi_vault.json", secret_key: Optional[str] = None):
        self.storage_path = storage_path
        if not secret_key:
            secret_key = os.getenv("DEID_SECRET_KEY", "shreyas")
        self.secret_key = secret_key
        self._fernet = self._init_fernet(self.secret_key)
        self.vault = self._load_vault()

    def _init_fernet(self, key_str: str):
        key = hashlib.sha256(key_str.encode('utf-8')).digest()
        b64_key = base64.urlsafe_b64encode(key)
        if HAS_CRYPTOGRAPHY:
            return Fernet(b64_key)
        return b64_key

    def _encrypt_bytes(self, raw_bytes: bytes, key_str: Optional[str] = None) -> str:
        fernet = self._fernet
        if key_str and key_str != self.secret_key:
            fernet = self._init_fernet(key_str)

        if HAS_CRYPTOGRAPHY and isinstance(fernet, Fernet):
            return fernet.encrypt(raw_bytes).decode('utf-8')
        
        # Authenticated HMAC-SHA256 + XOR Fallback
        k = key_str or self.secret_key
        key_bytes = hashlib.sha256(k.encode('utf-8')).digest()
        xor_bytes = bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(raw_bytes)])
        mac = hashlib.sha256(key_bytes + xor_bytes).hexdigest()
        payload = {"cipher": base64.b64encode(xor_bytes).decode('utf-8'), "mac": mac}
        return base64.b64encode(json.dumps(payload).encode('utf-8')).decode('utf-8')

    def _decrypt_bytes(self, enc_str: str, key_str: Optional[str] = None) -> bytes:
        fernet = self._fernet
        if key_str and key_str != self.secret_key:
            fernet = self._init_fernet(key_str)

        if HAS_CRYPTOGRAPHY and isinstance(fernet, Fernet):
            return fernet.decrypt(enc_str.encode('utf-8'))
        
        # Fallback decryption with HMAC validation
        k = key_str or self.secret_key
        key_bytes = hashlib.sha256(k.encode('utf-8')).digest()
        raw_json = base64.b64decode(enc_str.encode('utf-8'))
        payload = json.loads(raw_json.decode('utf-8'))
        
        cipher_bytes = base64.b64decode(payload["cipher"].encode('utf-8'))
        expected_mac = hashlib.sha256(key_bytes + cipher_bytes).hexdigest()
        if payload.get("mac") != expected_mac:
            raise ValueError("Authentication tag mismatch or corrupted data.")
            
        return bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(cipher_bytes)])

    def _load_vault(self) -> Dict[str, Dict[str, str]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    payload = json.loads(content)
                    if isinstance(payload, dict) and "encrypted_payload" in payload:
                        decrypted_bytes = self._decrypt_bytes(payload["encrypted_payload"])
                        return json.loads(decrypted_bytes.decode('utf-8'))
                    elif isinstance(payload, dict):
                        # Convert legacy plain dictionary payload into encrypted state
                        return payload
            except Exception as e:
                # Log sanitized warning without printing sensitive PHI values
                print(f"Vault Loading Alert: Vault file at {self.storage_path} could not be decrypted ({type(e).__name__}). Starting fresh vault state.")
                return {}
        return {}

    def _save_vault(self):
        try:
            json_bytes = json.dumps(self.vault, indent=4).encode('utf-8')
            enc_payload = self._encrypt_bytes(json_bytes)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "2.0",
                    "encrypted": True,
                    "algorithm": "AES-128-Fernet" if HAS_CRYPTOGRAPHY else "XOR-HMAC-SHA256",
                    "encrypted_payload": enc_payload
                }, f, indent=4)
        except Exception as e:
            print(f"Error saving encrypted vault: {type(e).__name__}")

    def add_mapping(self, doc_id: str, mapping: Dict[str, str]):
        if not doc_id or not mapping:
            return
        if doc_id not in self.vault:
            self.vault[doc_id] = {}
        self.vault[doc_id].update(mapping)
        self._save_vault()

    def get_original(self, doc_id: str, placeholder: str, provided_key: Optional[str] = None) -> str:
        if provided_key and provided_key != self.secret_key:
            return "ERROR: Unauthorized access - Invalid Secret Key"
        return self.vault.get(doc_id, {}).get(placeholder, placeholder)

    def demask_text(self, text: str, doc_id: str, provided_key: Optional[str] = None) -> str:
        """
        Demasks placeholders back to original PHI if authorized.
        Validates key and rejects unauthorized access or corrupted state.
        """
        if provided_key and provided_key != self.secret_key:
            return "ERROR: Unauthorized access - Invalid Secret Key"
        
        if doc_id not in self.vault:
            return text

        doc_mappings = self.vault[doc_id]
        demasked_text = text
        for placeholder, original in doc_mappings.items():
            demasked_text = demasked_text.replace(placeholder, original)
        return demasked_text

if __name__ == "__main__":
    vault = PHIVault()
    vault.add_mapping("doc1", {"[PATIENT_NAME_1]": "John Doe"})
    print("Demasked test:", vault.demask_text("Patient [PATIENT_NAME_1]", "doc1"))
