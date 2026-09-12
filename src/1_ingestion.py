import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

# Use absolute paths based on the script's location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = DATA_DIR / "vector_store"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Expanded LactMed Dataset
lactmed_data = {
    "Sertraline": {
        "text": "Sertraline is a Selective Serotonin Reuptake Inhibitor (SSRI). Because of the low levels of sertraline in breastmilk, amounts ingested by the infant are small and is usually not detected in the serum of the infant. It is considered a first-line agent for postpartum depression.",
        "RID": "0.5% to 3.0%",
        "half_life": "26 hours"
    },
    "Zuranolone": {
        "text": "Zuranolone is a neuroactive steroid GABA-A receptor positive modulator indicated for the treatment of postpartum depression. It is taken orally once daily for 14 days. Caution is advised as long-term infant effects are still being studied, and sedation can occur.",
        "RID": "Less than 1%",
        "half_life": "19.7 to 24.6 hours"
    },
    "Brexanolone": {
        "text": "Brexanolone is an intravenous formulation of allopregnanolone, used for severe postpartum depression. Because of its poor oral bioavailability, any drug in breastmilk is unlikely to reach the infant's systemic circulation in significant amounts.",
        "RID": "1.3% to 2.8%",
        "half_life": "9 hours"
    },
    "Fluoxetine": {
        "text": "Fluoxetine is an SSRI. It has an active metabolite, norfluoxetine, with a very long half-life. Because of this, it can accumulate in nursing infants, especially neonates. Other agents are preferred if starting treatment during lactation.",
        "RID": "1.6% to 14.6%",
        "half_life": "1 to 4 days (Norfluoxetine: 7 to 15 days)"
    },
    "Escitalopram": {
        "text": "Escitalopram is an SSRI. Levels in breastmilk are generally low, but slightly higher than sertraline. It is generally considered acceptable during breastfeeding, but infants should be monitored for excess sedation or poor feeding.",
        "RID": "3.9% to 7.9%",
        "half_life": "27 to 32 hours"
    },
    "Paroxetine": {
        "text": "Paroxetine is an SSRI. It produces very low levels in breastmilk and infant serum. It is considered a preferred agent during breastfeeding. However, it has a short half-life which can cause withdrawal symptoms if the mother misses a dose.",
        "RID": "1.2% to 2.8%",
        "half_life": "21 hours"
    },
    "Venlafaxine": {
        "text": "Venlafaxine is a Serotonin and Norepinephrine Reuptake Inhibitor (SNRI). The active metabolite, O-desmethylvenlafaxine (ODV), is excreted in breastmilk in relatively high amounts compared to SSRIs. Monitoring of the infant is recommended.",
        "RID": "6.8% to 8.1%",
        "half_life": "5 hours (ODV: 11 hours)"
    }
}

def build_aura_databases():
    print("Initializing AURA Knowledge Base...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    embedding_dim = 384
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    chunk_mapping = {}
    guardrail_db = {}
    vector_id = 0

    print(f"Ingesting {len(lactmed_data)} specialized PPD drugs...")
    for drug_name, data in lactmed_data.items():
        guardrail_db[drug_name] = {
            "RID": data["RID"],
            "half_life": data["half_life"]
        }

        text_chunk = f"Drug: {drug_name}. Pharmacokinetics and Safety: {data['text']}"
        vector = model.encode([text_chunk])
        faiss_index.add(np.array(vector).astype('float32'))
        
        chunk_mapping[vector_id] = text_chunk
        vector_id += 1

    faiss.write_index(faiss_index, str(VECTOR_DIR / "lactmed_faiss.index"))
    with open(VECTOR_DIR / "chunk_mapping.json", "w") as f:
        json.dump(chunk_mapping, f, indent=4)
    with open(PROCESSED_DIR / "lactmed_exact.json", "w") as f:
        json.dump(guardrail_db, f, indent=4)

    print("Data Ingestion Complete! Vector DB and Guardrail JSON successfully built.")

if __name__ == '__main__':
    build_aura_databases()
