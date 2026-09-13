import pandas as pd
import random
import os
from pathlib import Path

# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
random.seed(42)

# Parameters
num_vignettes = 150
drugs = ["Sertraline", "Zuranolone", "Brexanolone", "Fluoxetine", "Escitalopram", "Paroxetine", "Venlafaxine", "Lithium", "Aspirin"] # Included negative controls

ages = range(18, 45)
infant_status = ["healthy newborn", "premature neonate", "3-month-old infant", "6-month-old infant"]
comorbidities = ["none", "history of seizures", "hypertension", "bipolar disorder", "anxiety", "insomnia"]
symptoms = ["severe sadness", "suicidal ideation", "lethargy", "panic attacks", "detachment from infant"]

vignettes = []

for i in range(num_vignettes):
    drug = random.choice(drugs)
    age = random.choice(ages)
    infant = random.choice(infant_status)
    comorb = random.choice(comorbidities)
    symp = random.choice(symptoms)
    
    text = f"A {age}-year-old breastfeeding mother with a {infant} presents with {symp}. Medical history is notable for {comorb}. The care team is considering initiating {drug}."
    
    vignettes.append({
        "vignette_id": i + 1,
        "patient_age": age,
        "infant_status": infant,
        "comorbidity": comorb,
        "target_drug": drug,
        "vignette_text": text
    })

df = pd.DataFrame(vignettes)
output_path = DATA_DIR / '150_clinical_vignettes.csv'
df.to_csv(output_path, index=False)
print(f"Successfully generated {len(df)} clinical vignettes at {output_path}")
