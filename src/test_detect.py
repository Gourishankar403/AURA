import sys
sys.stdout.reconfigure(encoding='utf-8')

SUPPORTED_DRUGS = ["Sertraline","Zuranolone","Brexanolone","Fluoxetine","Escitalopram","Paroxetine","Venlafaxine"]
OUT_OF_SCOPE_DRUGS = ["Lithium","Aspirin","Ibuprofen","Paracetamol","Acetaminophen","Lorazepam","Diazepam","Olanzapine","Quetiapine","Risperidone","Haloperidol","Valproate","Lamotrigine","Clomipramine","Duloxetine","Mirtazapine","Bupropion","Citalopram"]

def detect_drug(text):
    lower = text.lower()
    for drug in SUPPORTED_DRUGS:
        if drug.lower() in lower:
            return drug, True
    for drug in OUT_OF_SCOPE_DRUGS:
        if drug.lower() in lower:
            return drug, False
    return None, None

# Test 3 cases
tests = [
    ("A mother wants to take Sertraline for PPD.", "SUPPORTED"),
    ("A breastfeeding mother with a severe headache wants to take Aspirin.", "OUT-OF-SCOPE DEFERRAL"),
    ("A patient with no medicine mentioned in the text.", "NO DRUG FOUND"),
]

print("SMOKE TEST RESULTS:")
print("="*60)
for vignette, expected in tests:
    drug, in_scope = detect_drug(vignette)
    if drug is None:
        result = "NO DRUG FOUND"
    elif in_scope:
        result = "SUPPORTED"
    else:
        result = "OUT-OF-SCOPE DEFERRAL"

    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] Vignette: '{vignette[:45]}...'")
    print(f"       Detected: {drug} | In-scope: {in_scope} | Result: {result}")
    print()
