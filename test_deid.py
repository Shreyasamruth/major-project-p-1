import pandas as pd
from engine.deid import DeidEngine
from engine.context import ContextPreserver
import os

def test_deid_pipeline():
    # Initialize engine and context preserver
    engine = DeidEngine()
    preserver = ContextPreserver()
    
    # Path to metadata
    csv_path = "d:/major project/dataset/dataset_metadata.csv"
    if not os.path.exists(csv_path):
        print(f"Metadata not found at {csv_path}")
        return

    # Load samples
    df = pd.read_csv(csv_path)
    samples = df.head(5)
    
    for idx, row in samples.iterrows():
        print(f"\n--- Testing Sample {idx} ---")
        # Construct a report text from metadata
        text = f"""
        Hospital: {row['hospital_name']}
        Patient Name: {row['patient_name']}
        Patient ID: {row['patient_id']}
        Age/Gender: {row['age']} / {row['gender']}
        Indication: {row['indication']}
        Findings: {row['findings']}
        Impression: {row['impression']}
        """
        
        # 1. Detect PHI
        all_ents = engine.detect_phi(text)
        
        # 2. Filter using Context Preservation
        phi_only = preserver.filter_phi(all_ents)
        
        # 3. Mask
        masked, vault = engine.mask_text(text, phi_only, reversible=True)
        
        print(f"Detected entities: {[e['text'] for e in all_ents]}")
        print(f"PHI after filtering: {[e['text'] for e in phi_only]}")
        # print(f"Masked Text:\n{masked}")
        print(f"Vault size: {len(vault)}")

if __name__ == "__main__":
    test_deid_pipeline()
