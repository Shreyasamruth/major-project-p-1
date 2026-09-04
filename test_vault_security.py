import unittest
import os
import tempfile
import json
from engine.vault import PHIVault

class TestPHIVaultSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_vault_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_vault_file.close()
        self.secret_key = "TopSecretMasterKey2026"
        self.vault = PHIVault(storage_path=self.temp_vault_file.name, secret_key=self.secret_key)

    def tearDown(self):
        if os.path.exists(self.temp_vault_file.name):
            os.remove(self.temp_vault_file.name)

    def test_1_encrypt_and_store(self):
        """Test encrypt and store sensitive PHI mappings into authenticated ciphertext payload."""
        doc_id = "DOC_STORE_001"
        mapping = {"[PATIENT_NAME_1]": "Jane Doe", "[MRN_1]": "MRN998877"}
        self.vault.add_mapping(doc_id, mapping)
        
        # Verify file on disk is encrypted (no plaintext PHI strings)
        with open(self.temp_vault_file.name, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        self.assertNotIn("Jane Doe", raw_content)
        self.assertNotIn("MRN998877", raw_content)
        self.assertIn("encrypted_payload", raw_content)

    def test_2_retrieve_and_decrypt(self):
        """Test authorized retrieval and decryption of stored PHI mapping."""
        doc_id = "DOC_RETR_002"
        mapping = {"[PATIENT_NAME_1]": "Robert Johnson", "[SSN_1]": "123-45-6789"}
        self.vault.add_mapping(doc_id, mapping)

        # Reload vault from disk with correct key
        loaded_vault = PHIVault(storage_path=self.temp_vault_file.name, secret_key=self.secret_key)
        self.assertEqual(loaded_vault.get_original(doc_id, "[PATIENT_NAME_1]"), "Robert Johnson")
        self.assertEqual(loaded_vault.get_original(doc_id, "[SSN_1]"), "123-45-6789")

        # Demask text
        masked_text = "Report for [PATIENT_NAME_1] SSN: [SSN_1]"
        demasked = loaded_vault.demask_text(masked_text, doc_id, provided_key=self.secret_key)
        self.assertEqual(demasked, "Report for Robert Johnson SSN: 123-45-6789")

    def test_3_wrong_key(self):
        """Test that supplying an incorrect secret key fails decryption or returns authorization error."""
        doc_id = "DOC_KEY_003"
        self.vault.add_mapping(doc_id, {"[PATIENT_NAME_1]": "Confidential Person"})

        # Reload vault with WRONG secret key
        wrong_vault = PHIVault(storage_path=self.temp_vault_file.name, secret_key="WrongSecretKey999")
        # Encryption payload cannot be decrypted with wrong key
        self.assertEqual(wrong_vault.vault, {})

        # Attempting demasking with wrong key
        res = self.vault.demask_text("Patient [PATIENT_NAME_1]", doc_id, provided_key="WrongKey123")
        self.assertTrue(res.startswith("ERROR: Unauthorized access"))

    def test_4_corrupted_data(self):
        """Test vault loading resilience when ciphertext or json file on disk is corrupted."""
        doc_id = "DOC_CORRUPT_004"
        self.vault.add_mapping(doc_id, {"[PATIENT_NAME_1]": "Alice Smith"})

        # Corrupt file payload on disk
        with open(self.temp_vault_file.name, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "version": "2.0",
                "encrypted": True,
                "encrypted_payload": "INVALID_CORRUPTED_BASE64_PAYLOAD_%%%!!!"
            }))

        # Load vault from corrupted file
        corrupted_vault = PHIVault(storage_path=self.temp_vault_file.name, secret_key=self.secret_key)
        self.assertEqual(corrupted_vault.vault, {}) # Gracefully resets state, zero unhandled crash

    def test_5_unauthorized_access(self):
        """Test rejection of unauthorized access when key is omitted or invalid."""
        doc_id = "DOC_UNAUTH_005"
        self.vault.add_mapping(doc_id, {"[PATIENT_NAME_1]": "Secret Patient"})

        # Check get_original with invalid key
        orig = self.vault.get_original(doc_id, "[PATIENT_NAME_1]", provided_key="BadKey")
        self.assertTrue(orig.startswith("ERROR: Unauthorized access"))

        # Check demask_text with invalid key
        demasked = self.vault.demask_text("Patient [PATIENT_NAME_1]", doc_id, provided_key="BadKey")
        self.assertTrue(demasked.startswith("ERROR: Unauthorized access"))

    def test_6_repeated_placeholder_lookup(self):
        """Test consistent mapping for repeated placeholder lookups."""
        doc_id = "DOC_REPEAT_006"
        placeholder = "[PATIENT_NAME_1]"
        original_name = "Dr. Alexander Fleming"
        self.vault.add_mapping(doc_id, {placeholder: original_name})

        for _ in range(50):
            res = self.vault.get_original(doc_id, placeholder, provided_key=self.secret_key)
            self.assertEqual(res, original_name)

        text = "Patient [PATIENT_NAME_1] visited [PATIENT_NAME_1] again with [PATIENT_NAME_1]."
        demasked = self.vault.demask_text(text, doc_id, provided_key=self.secret_key)
        expected = "Patient Dr. Alexander Fleming visited Dr. Alexander Fleming again with Dr. Alexander Fleming."
        self.assertEqual(demasked, expected)

if __name__ == "__main__":
    unittest.main()
