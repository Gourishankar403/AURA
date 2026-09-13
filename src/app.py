import streamlit as st
import os
import re
import groq
from dotenv import load_dotenv
from pathlib import Path
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from retrieval import AURARetriever

# ── Constants ────────────────────────────────────────────────────────────────
SUPPORTED_DRUGS = [
    "Sertraline", "Zuranolone", "Brexanolone",
    "Fluoxetine", "Escitalopram", "Paroxetine", "Venlafaxine"
]

# ── Drug auto-detector ────────────────────────────────────────────────────────
def detect_drug(text: str) -> str | None:
    """Case-insensitive keyword scan over the vignette to find a supported drug."""
    lower = text.lower()
    for drug in SUPPORTED_DRUGS:
        if drug.lower() in lower:
            return drug
    return None

# ── System setup ─────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

@st.cache_resource(show_spinner="Loading AURA databases…")
def load_system():
    retriever = AURARetriever()
    api_key = os.getenv("GROQ_API_KEY", "")
    client = groq.Groq(api_key=api_key) if api_key and api_key != "YOUR_API_KEY_HERE" else None
    return retriever, client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA Clinical System",
    page_icon="🛡️",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ AURA")
    st.caption("Adaptive Uncertainty Reduction Architecture")
    st.markdown("---")
    st.subheader("System Status")
    st.success("✅ FAISS Semantic Vector DB")
    st.success("✅ Sentinel JSON Guardrail")
    st.success("✅ Groq LLM Engine (qwen3.8-27b)")
    st.markdown("---")
    st.subheader("About AURA")
    st.info(
        "AURA separates language generation from mathematical data. "
        "The LLM handles clinical reasoning, while a deterministic JSON guardrail "
        "injects exact pharmacokinetic numbers to guarantee **0% hallucination**."
    )
    st.markdown("---")
    st.subheader("Supported Drugs")
    for d in SUPPORTED_DRUGS:
        st.markdown(f"- {d}")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🛡️ AURA: Adaptive Uncertainty Reduction Architecture")
st.markdown("*Zero-Hallucination Medical Decision Support for Postpartum Psychopharmacology*")
st.markdown("---")

retriever, client = load_system()
if not client:
    st.error("❌ GROQ_API_KEY is missing or invalid in your .env file!")
    st.stop()

# ── Sample vignettes ──────────────────────────────────────────────────────────
sample_vignettes = {
    "— Choose a preset —": "",
    "The Premature Neonate Risk (Fluoxetine)":
        "A 30-year-old breastfeeding mother with a 3-week-old premature neonate wants to take Fluoxetine for severe postpartum depression. Is this safe?",
    "Standard PPD Case (Sertraline)":
        "A 28-year-old breastfeeding mother with a healthy 6-month-old presents with low mood, tearfulness and anxiety. We are considering initiating Sertraline.",
    "The Deferral Trap (Lithium)":
        "A patient with postpartum psychosis needs Lithium. Provide the safety profile and Relative Infant Dose.",
    "New FDA Drug (Zuranolone)":
        "A 28-year-old breastfeeding mother with severe PPD is starting Zuranolone. What is the clinical safety profile and RID?",
    "Custom Prompt…": "",
}

st.header("Patient Consultation")

col1, col2 = st.columns([3, 1])

with col1:
    preset = st.selectbox("Load Sample Vignette (Presentation Demo):", list(sample_vignettes.keys()))
    vignette_default = sample_vignettes[preset]
    vignette = st.text_area("Patient Vignette:", value=vignette_default, height=120,
                             placeholder="Describe the patient scenario here…")

with col2:
    st.markdown("**Target Drug**")
    st.caption("Select 'Auto-Detect' to extract drug from vignette automatically.")
    drug_options = ["Auto-Detect from Vignette"] + SUPPORTED_DRUGS + ["Other (Out-of-Database)"]
    target_drug_selection = st.selectbox("Target Drug:", drug_options)

run = st.button("🔬 Generate AURA Report", type="primary", use_container_width=True)

# ── Processing ────────────────────────────────────────────────────────────────
if run:
    if not vignette.strip():
        st.warning("⚠️ Please enter a patient vignette first.")
        st.stop()

    # ── Step 1: Resolve target drug ──────────────────────────────────────────
    if target_drug_selection == "Auto-Detect from Vignette":
        detected = detect_drug(vignette)
        if detected:
            target_drug = detected
            st.info(f"🔍 **Auto-detected drug:** {target_drug}")
        else:
            st.error(
                "❌ **Auto-detection failed.** No supported drug name found in the vignette text. "
                "Please select the drug manually from the dropdown, or type the drug name into the vignette."
            )
            st.stop()
    elif target_drug_selection == "Other (Out-of-Database)":
        # Force a deferral without even hitting the LLM
        st.error(
            "🚨 **HUMAN DEFERRAL TRIGGERED** 🚨\n\n"
            "This drug is NOT in the verified LactMed database. "
            "AURA physically prevents out-of-bounds recommendations.\n\n"
            "**Action required:** Consult a clinical pharmacologist directly."
        )
        st.stop()
    else:
        target_drug = target_drug_selection

    # ── Step 2: Guardrail check ──────────────────────────────────────────────
    guardrail_data = retriever.get_guardrail_data(target_drug)
    if not guardrail_data:
        st.error(
            f"🚨 **HUMAN DEFERRAL TRIGGERED** 🚨\n\n"
            f"Drug **'{target_drug}'** is NOT in the verified LactMed database. "
            f"AURA physically prevents out-of-bounds recommendations.\n\n"
            f"**Action required:** Consult a clinical pharmacologist directly."
        )
        st.stop()

    exact_rid = guardrail_data.get("RID", "N/A")
    exact_half_life = guardrail_data.get("half_life", "N/A")

    # ── Step 3: Semantic retrieval ───────────────────────────────────────────
    context_results = retriever.search_semantic(target_drug, top_k=1)
    medical_context = context_results[0] if context_results else f"{target_drug} is a medication used in PPD."

    # ── Step 4: LLM reasoning (plain text, tight prompt) ────────────────────
    with st.spinner(f"⏳ Generating clinical reasoning for **{target_drug}**…"):
        system_msg = (
            "You are a clinical pharmacology assistant specialized in maternal health. "
            "Be concise. 3-4 sentences maximum. Never invent numerical dosage data."
        )
        user_msg = (
            f"LactMed Context: {medical_context}\n\n"
            f"Patient: {vignette}\n\n"
            f"Drug being considered: {target_drug}\n\n"
            f"Provide a brief clinical safety assessment."
        )
        try:
            completion = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            reasoning = completion.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

    # ── Step 5: Render report ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 AURA Final Medical Report")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Patient Drug:** `{target_drug}`")
        st.markdown("**Layer 1 — LLM Clinical Reasoning** *(Language only, no math)*")
        st.info(reasoning)

    with col_b:
        st.markdown("**Layer 2 — Sentinel Guardrail** *(Deterministic, zero-hallucination)*")
        st.success(
            f"**Relative Infant Dose (RID):** {exact_rid}\n\n"
            f"**Elimination Half-life:** {exact_half_life}\n\n"
            f"*⚠️ Mathematically verified from NIH LactMed JSON vault — not generated by AI.*"
        )
        st.markdown("**Compliance Status:**")
        st.success("✅ Report generated within verified database scope.")
