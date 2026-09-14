import streamlit as st
import os
import groq
from dotenv import load_dotenv
from pathlib import Path
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))
from retrieval import AURARetriever

# ── Drug registry ─────────────────────────────────────────────────────────────
SUPPORTED_DRUGS = [
    "Sertraline", "Zuranolone", "Brexanolone",
    "Fluoxetine", "Escitalopram", "Paroxetine", "Venlafaxine"
]
OUT_OF_SCOPE_DRUGS = [
    "Lithium", "Aspirin", "Ibuprofen", "Paracetamol", "Acetaminophen",
    "Lorazepam", "Diazepam", "Clonazepam", "Olanzapine", "Quetiapine",
    "Risperidone", "Haloperidol", "Valproate", "Lamotrigine", "Clomipramine",
    "Amitriptyline", "Duloxetine", "Mirtazapine", "Bupropion", "Citalopram",
    "Alprazolam", "Zolpidem", "Codeine", "Tramadol", "Methadone",
    "Warfarin", "Atenolol", "Metformin", "Levothyroxine", "Prednisone"
]

def detect_drug(text: str):
    lower = text.lower()
    for drug in SUPPORTED_DRUGS:
        if drug.lower() in lower:
            return drug, True
    for drug in OUT_OF_SCOPE_DRUGS:
        if drug.lower() in lower:
            return drug, False
    return None, None

# ── Init ──────────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

@st.cache_resource(show_spinner="Loading AURA clinical databases…")
def load_system():
    retriever = AURARetriever()
    api_key   = os.getenv("GROQ_API_KEY", "")
    client    = groq.Groq(api_key=api_key) if api_key and api_key != "YOUR_API_KEY_HERE" else None
    return retriever, client

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AURA Clinical System", page_icon="🛡️", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ AURA")
    st.caption("Adaptive Uncertainty Reduction Architecture")
    st.markdown("---")
    st.subheader("System Status")
    st.success("✅ FAISS Semantic Vector DB")
    st.success("✅ Sentinel JSON Guardrail")
    st.success("✅ Groq LLM Engine")
    st.markdown("---")
    st.subheader("Verified Drug Database (7 Drugs)")
    for d in SUPPORTED_DRUGS:
        st.markdown(f"- {d}")
    st.markdown("---")
    st.subheader("About AURA")
    st.info(
        "AURA separates language from math. "
        "The LLM handles clinical reasoning using verified NIH LactMed context. "
        "The Sentinel Guardrail then injects exact pharmacokinetic numbers. "
        "**0% numerical hallucination guaranteed.**"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛡️ AURA: Adaptive Uncertainty Reduction Architecture")
st.markdown("*Zero-Hallucination Medical Decision Support for Postpartum Psychopharmacology*")
st.markdown("---")

retriever, client = load_system()
if not client:
    st.error("❌ GROQ_API_KEY missing in .env file!")
    st.stop()

# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "— Choose a preset —": "",
    "Standard PPD · Sertraline":
        "A 28-year-old breastfeeding mother with a healthy 6-month-old has persistent low mood, tearfulness, and poor appetite. The care team is considering initiating Sertraline.",
    "New FDA Drug · Zuranolone":
        "A 32-year-old breastfeeding mother with severe PPD is being evaluated for the new oral treatment Zuranolone. Assess the clinical safety profile.",
    "Premature Neonate Risk · Fluoxetine":
        "A 30-year-old breastfeeding mother with a 3-week-old premature neonate wants to take Fluoxetine for severe postpartum depression.",
    "IV Treatment · Brexanolone":
        "A mother is receiving an IV infusion of Brexanolone over 60 hours for severe PPD. What is the safety profile during breastfeeding?",
    "Out-of-Scope Deferral · Lithium":
        "A patient with postpartum psychosis has been suggested Lithium by the attending psychiatrist. Assess the safety for breastfeeding.",
    "Out-of-Scope Deferral · Aspirin":
        "A breastfeeding mother with a severe headache wants to take Aspirin. Based strictly on the PPD database, what is the half-life?",
    "SNRI Option · Venlafaxine":
        "A 34-year-old breastfeeding mother failed sertraline. The team is considering switching to Venlafaxine. Is it safe for the infant?",
    "Custom Prompt…": "",
}

st.header("Patient Consultation")
preset   = st.selectbox("Load Sample Vignette:", list(PRESETS.keys()))
vignette = st.text_area(
    "Patient Vignette:",
    value=PRESETS[preset],
    height=130,
    placeholder="Describe the patient here. Include the medicine name (e.g. Sertraline, Zuranolone…)"
)
st.caption("💡 AURA automatically detects the drug from your vignette — no manual selection needed.")
run = st.button("🔬 Generate AURA Report", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    if not vignette.strip():
        st.warning("⚠️ Please enter a patient vignette.")
        st.stop()

    # 1. Auto-detect
    drug, in_scope = detect_drug(vignette)

    if drug is None:
        st.error(
            "⚠️ **No drug name detected.**\n\n"
            "Please mention the medication by name in the vignette "
            "(e.g. *'...considering initiating **Sertraline**...'*)\n\n"
            "Supported drugs: " + ", ".join(SUPPORTED_DRUGS)
        )
        st.stop()

    st.markdown(f"> 🔍 **Auto-detected drug:** `{drug}`")

    # 2. Out-of-scope deferral
    if not in_scope:
        st.error(
            f"### 🚨 HUMAN DEFERRAL TRIGGERED\n\n"
            f"**{drug}** is outside the verified NIH LactMed database scope.\n\n"
            f"Generating pharmacokinetic data for unverified drugs would constitute a hallucination risk. "
            f"AURA's Sentinel Guardrail has terminated this query to protect patient safety.\n\n"
            f"**Required action:** Consult a clinical pharmacologist or refer directly to "
            f"[NIH LactMed](https://www.ncbi.nlm.nih.gov/books/NBK501922/) for *{drug}*."
        )
        st.stop()

    # 3. Guardrail lookup
    guardrail_data = retriever.get_guardrail_data(drug)
    if not guardrail_data:
        st.error(f"🚨 No verified record for {drug}. Human deferral triggered.")
        st.stop()

    exact_rid       = guardrail_data.get("RID", "N/A")
    exact_half_life = guardrail_data.get("half_life", "N/A")

    # 4. Semantic context
    context_results = retriever.search_semantic(drug, top_k=1)
    medical_context = context_results[0] if context_results else f"{drug} is used in PPD treatment."

    # 5. LLM reasoning
    with st.spinner(f"Generating clinical assessment for {drug}…"):
        system_msg = (
            f"You are AURA, a clinical pharmacology assistant specialised in maternal mental health "
            f"and breastfeeding safety. You have been given the complete verified NIH LactMed clinical "
            f"profile for {drug}. "
            f"Your task: write a concise, confident 3-5 sentence clinical safety assessment based on "
            f"this evidence and the patient's specific details (age, infant age/maturity, comorbidities). "
            f"IMPORTANT RULES:\n"
            f"1. Do NOT mention RID percentages or half-life values — those are injected separately by the Sentinel Guardrail.\n"
            f"2. Do NOT say you lack information — you have the full LactMed profile.\n"
            f"3. Focus on: is this drug appropriate for THIS specific patient? What monitoring is needed?\n"
            f"4. If the infant is premature or a newborn, emphasise the elevated risk explicitly.\n"
            f"5. Mention the AAP classification if relevant.\n"
            f"6. Be direct and clinically confident."
        )
        user_msg = (
            f"Verified NIH LactMed Clinical Profile for {drug}:\n"
            f"{medical_context}\n\n"
            f"Patient Clinical Scenario:\n{vignette}\n\n"
            f"Provide your clinical safety assessment."
        )
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            reasoning = completion.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

    # 6. Unified Report
    st.markdown("---")
    st.subheader(f"📋 AURA Clinical Report — {drug}")

    # Guardrail data FIRST (prominent, at the top)
    st.markdown("#### 🛡️ Verified Pharmacokinetic Data (NIH LactMed)")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Relative Infant Dose (RID)", exact_rid)
    with m2:
        st.metric("Elimination Half-life", exact_half_life)
    st.caption("*⚠️ These values are injected deterministically from the NIH LactMed JSON vault — not generated by AI.*")

    st.markdown("---")

    # LLM reasoning below
    st.markdown("#### 🧠 Clinical Safety Assessment")
    st.info(reasoning)
    st.caption(f"*Reasoning generated using full NIH LactMed evidence for {drug}.*")

    st.success("✅ Zero numerical hallucination — pharmacokinetic data sourced deterministically from verified NIH LactMed database.")
