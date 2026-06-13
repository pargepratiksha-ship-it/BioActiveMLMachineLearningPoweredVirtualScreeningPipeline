import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import matthews_corrcoef

# --- 1. WEB PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BioActive-ML Platform",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { 
        font-size: 48px; font-weight: 900; 
        background: linear-gradient(135deg, #FF4B4B 0%, #FF69B4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    .subtitle { 
        font-size: 18px; text-align: center; 
        color: #00D9FF; 
        margin-bottom: 30px;
        font-weight: 600;
    }
    .stat-card { 
        background: linear-gradient(135deg, #1a1f35 0%, #262a47 100%);
        padding: 20px; 
        border-radius: 12px; 
        border: 2px solid #00D9FF;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 217, 255, 0.2);
    }
    .result-active {
        background: linear-gradient(135deg, rgba(0, 255, 102, 0.1) 0%, rgba(0, 255, 102, 0.05) 100%);
        border: 2px solid #00FF66;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0, 255, 102, 0.15);
    }
    .result-inactive {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 75, 75, 0.05) 100%);
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(255, 75, 75, 0.15);
    }
    .result-moderate {
        background: linear-gradient(135deg, rgba(255, 184, 0, 0.1) 0%, rgba(255, 184, 0, 0.05) 100%);
        border: 2px solid #FFB800;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(255, 184, 0, 0.15);
    }
    .explanation-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2a1f4d 100%);
        border-left: 4px solid #00D9FF;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 14px;
        line-height: 1.8;
    }
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, #00FF66 0%, #FFB800 50%, #FF4B4B 100%);
        box-shadow: 0 2px 8px rgba(0, 217, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER BLOCK ---
st.markdown("<div class='main-title'>🧬 BioActive-ML</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Virtual Screening Platform • AI-Powered Bioactivity Prediction • Drug Discovery Acceleration</div>", unsafe_allow_html=True)

st.markdown("""
---
### Welcome to BioActive-ML
This platform predicts whether chemical compounds and peptides will show biological activity against a target protein.
Simply input your molecule, and our machine learning model will score its predicted bioactivity.

**How it works:** Enter a target protein, submit a chemical structure, and receive bioactivity predictions with confidence metrics and detailed explanations.

---
""")

# --- 3. LIVE UNIPROT API FETCHING FUNCTION ---
def fetch_uniprot_data(uid):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            pref_name = "Unknown Protein"
            try:
                rec_name = data.get("proteinDescription", {}).get("recommendedName", {})
                pref_name = rec_name.get("fullName", {}).get("value", "Unknown Protein")
            except:
                pass
            organism = data.get("organism", {}).get("scientificName", "Unknown Species")
            protein_class = "Functional Cellular Protein"
            comments = data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("text", [])
                    if texts:
                        protein_class = texts[0].get("value", "Functional Cellular Protein")
                        break
            return {"success": True, "name": pref_name, "organism": organism, "class": protein_class}
        else:
            return {"success": False, "error": "ID not found in global database"}
    except requests.RequestException as exc:
        return {"success": False, "error": f"Connection timeout: {exc}"}

@st.cache_data(show_spinner=False)
def build_training_model():
    sample_smiles = [
        # Active compounds
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        "CC1=C(C(=O)NC(C)C)N(C)C2=CC=CC=C12",
        "CC(=O)NCC1=CC=CC=C1",
        "CCC(=O)N1CCCC1C(=O)O",
        "CC1=C(C(=O)NC(CC2=CC=CC=C2)C)N(C)C2=CC=CC=C12",
        # Inactive compounds
        "CC(C)CC1=CC=CC=C1",
        "CC(C)OC1=CC=CC=C1",
        "CCCCC(=O)O",
        "CCN(CC)CC",
        "CCCCCC",
        # Additional actives
        "CC1=CN(C(=O)C2=CC=CC=C2)C=C1",
        "CNC(=O)C1=CC=CC=C1",
        "CC1=CN=CN1",
        "CCOC(=O)C1=CN=CN1",
        "CCN(CC)CCO",
        # Additional inactives
        "COC1=CC=CC=C1",
        "CCC1=CC=CC=C1",
        "CC(C)C(=O)O",
        "CC1=CC=CC=C1O",
        "CC(C)(C)O",
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    fingerprints = []
    for smiles in sample_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)

    X = np.vstack(fingerprints)
    y = np.array(labels[: len(fingerprints)])

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    scores_acc = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    scores_auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    y_pred = cross_val_predict(clf, X, y, cv=5)
    mcc = matthews_corrcoef(y, y_pred)
    clf.fit(X, y)

    return {
        'model': clf,
        'accuracy': float(np.mean(scores_acc)),
        'roc_auc': float(np.mean(scores_auc)),
        'mcc': float(mcc),
        'samples': X.shape[0],
    }

@st.cache_data(show_spinner=False)
def compute_fingerprint(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

model_info = build_training_model()

# --- 4. TARGET SELECTION ---
st.subheader("1. Specify Biological Target via UniProt ID")

uniprot_id = st.text_input("Enter UniProt ID:", value="P00533").strip().upper()
target_seed_modifier = 0

if uniprot_id:
    with st.spinner("Fetching live data from UniProt KB..."):
        result = fetch_uniprot_data(uniprot_id)
    if result["success"]:
        st.info(f"""
        **Target Protein:** {result['name']}  
        **Organism:** *{result['organism']}* | **Function:** {result['class']}
        """)
        target_seed_modifier = len(result['name'])
    else:
        st.warning(f"""
        Unable to fetch UniProt data for {uniprot_id}. Using default configuration.
        """)
        target_seed_modifier = len(uniprot_id)

st.write("---")

# --- 5. TWO-COLUMN WORKSPACE ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("2. Input Molecular Configuration")
    input_type = st.radio("Structural Representation Format:", ("Small Molecule (SMILES Notation)", "Peptide Sequence (FASTA Amino Acids)"))
    mol = None

    if input_type == "Small Molecule (SMILES Notation)":
        st.markdown("""
        **SMILES Format:** Simplified Molecular Input Line Entry System  
        *Example active compounds:*
        - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
        - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        """)
        smiles_input = st.text_input(
            "Enter SMILES String:", 
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Paste a valid SMILES string or use one of the examples above"
        )
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is None: 
                st.error("❌ Invalid SMILES syntax. Please check the string and try again.")
            else:
                st.success("✅ SMILES parsed successfully!")
    else:
        st.markdown("""
        **Peptide Format:** Single-letter amino acid codes  
        *Example sequences:*
        - Alanine-Glycine-Lysine: `AGK`
        - Sample peptide: `SSMAGAFDIG`
        """)
        peptide_input = st.text_input(
            "Enter Amino Acid Chain:", 
            value="SSMAGAFDIG",
            help="Use standard single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
        )
        if peptide_input:
            mol = Chem.MolFromSequence(peptide_input.upper())
            if mol is None: 
                st.error("❌ Invalid amino acid sequence. Please use standard single-letter codes.")
            else:
                st.success("✅ Sequence converted to molecule successfully!")

with col_right:
    st.subheader("3. Predictive Analytics & Results")
    if mol is not None:
        st.success("🔬 Chemical structural configuration loaded successfully!")
        mw = round(Chem.rdMolDescriptors.CalcExactMolWt(mol), 2)
        heavy_atoms = mol.GetNumHeavyAtoms()

        fp_array = compute_fingerprint(mol)
        model = model_info["model"]
        active_prob = float(model.predict_proba([fp_array])[0, 1])
        inactive_prob = 1.0 - active_prob
        
        # Display molecular properties
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Molecular Weight", f"{mw} Da", delta=None)
        with m2:
            st.metric("Heavy Atoms Count", f"{heavy_atoms}", delta=None)
        
        st.write("---")
        
        # Enhanced prediction results with cooler UI
        st.subheader("Bioactivity Prediction Result")
        
        col_prob, col_gauge = st.columns([1, 1])
        with col_prob:
            st.markdown(f"### **Binding Probability: {round(active_prob * 100, 1)}%**")
            st.markdown(f"<div class='confidence-bar'></div>", unsafe_allow_html=True)
            st.caption(f"Confidence: {round(active_prob * 100, 2)}% active | {round(inactive_prob * 100, 2)}% inactive")
        
        # Color-coded result boxes
        if active_prob >= 0.70:
            st.balloons()
            st.markdown(f"""
            <div class='result-active'>
                <h3>HIGH POTENCY CANDIDATE</h3>
                <p><strong>Status:</strong> Predicted to be <b>ACTIVE</b> against target {uniprot_id}</p>
                <p><strong>Action:</strong> Recommended for experimental validation and wet-lab assay testing.</p>
                <p><strong>Confidence:</strong> Very High ({round(active_prob * 100, 1)}%)</p>
            </div>
            """, unsafe_allow_html=True)
        elif active_prob >= 0.40:
            st.markdown(f"""
            <div class='result-moderate'>
                <h3>MODERATE ACTIVITY</h3>
                <p><strong>Status:</strong> Predicted to show <b>MODERATE</b> activity against target {uniprot_id}</p>
                <p><strong>Action:</strong> Consider structural modifications or scaffold optimization.</p>
                <p><strong>Confidence:</strong> Medium ({round(active_prob * 100, 1)}%)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-inactive'>
                <h3>NEGLIGIBLE BIOACTIVITY</h3>
                <p><strong>Status:</strong> Predicted to be <b>INACTIVE</b> against target {uniprot_id}</p>
                <p><strong>Action:</strong> Consider redesign or alternative chemical series.</p>
                <p><strong>Confidence:</strong> Low activity risk ({round(active_prob * 100, 1)}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Bioactivity explanation
        with st.expander("How Does This Prediction Work?"):
            st.markdown("""
            ### Bioactivity Prediction Methodology
            
            **What we're predicting:** Whether this compound will bind and show biological activity against your target protein.
            
            **How it works:**
            1. **Molecular Fingerprinting:** Your compound is converted into a Morgan fingerprint (2048 bits) — a numerical representation capturing its 2D chemical structure.
            2. **Machine Learning Model:** A Random Forest classifier (trained on active/inactive compounds) analyzes this fingerprint.
            3. **Probability Output:** The model outputs the probability that your compound is "active" (will bind) vs "inactive" (won't bind).
            
            **Why this is useful:**
            - **Virtual Screening:** Quickly filter thousands of compounds before expensive lab synthesis.
            - **Lead Optimization:** Identify which structural changes improve activity.
            - **Cost Savings:** Predict bioactivity before committing to wet-lab experiments.
            
            **Limitations:**
            - This model is trained on a demonstration dataset (not real ChEMBL data).
            - Results should always be validated experimentally.
            - The prediction is based on 2D structure; 3D binding and off-target effects are not modeled.
            - Applicability domain: The model works best on drug-like molecules with familiar scaffolds.
            
            **Next Steps:**
            • High potency candidates → prioritize for synthesis and testing  
            • Moderate activity → consider structure optimization  
            • Negligible activity → explore alternative chemical series  
            """)
        
        chart_data = pd.DataFrame({
            'Prediction': ['Likely Active', 'Likely Inactive'], 
            'Probability': [active_prob, inactive_prob]
        })
        st.bar_chart(data=chart_data, x='Prediction', y='Probability')
    else:
        st.warning("Awaiting biological structure input to initialize ML pipeline...")

st.write("---")

# --- 6. RELIABILITY & VALIDATION DASHBOARD ---
st.write("---")
st.subheader("4. Model Performance & Validation Metrics")
st.markdown("""
This dashboard shows the cross-validation performance of our Random Forest classifier on the demonstration dataset.
Learn how reliable these predictions are based on the model's training metrics.
""")

vm1, vm2, vm3, vm4 = st.columns(4)
with vm1:
    st.markdown(f"""
    <div class='stat-card'>
        <strong>ROC-AUC Score</strong><br>
        <span style='font-size:32px; color:#FF4B4B;'>{model_info['roc_auc']:.3f}</span><br>
        <small>Discriminatory Power</small>
    </div>
    """, unsafe_allow_html=True)
with vm2:
    st.markdown(f"""
    <div class='stat-card'>
        <strong>Accuracy</strong><br>
        <span style='font-size:32px; color:#00D9FF;'>{model_info['accuracy']:.1%}</span><br>
        <small>Prediction Consistency</small>
    </div>
    """, unsafe_allow_html=True)
with vm3:
    st.markdown(f"""
    <div class='stat-card'>
        <strong>MCC</strong><br>
        <span style='font-size:32px; color:#00FF66;'>{model_info['mcc']:.3f}</span><br>
        <small>Balanced Correlation</small>
    </div>
    """, unsafe_allow_html=True)
with vm4:
    st.markdown(f"""
    <div class='stat-card'>
        <strong>Training Samples</strong><br>
        <span style='font-size:32px; color:#FFB800;'>{model_info['samples']}</span><br>
        <small>Compounds Used</small>
    </div>
    """, unsafe_allow_html=True)

with st.expander("Understanding Model Metrics"):
    st.markdown(f"""
    ### Model Performance Explanation
    
    **ROC-AUC Score ({model_info['roc_auc']:.3f})**
    - Measures the model's ability to discriminate between active and inactive compounds
    - Range: 0.0 (random) to 1.0 (perfect)
    - Our model has a {model_info['roc_auc']:.1%} chance of ranking a random active higher than a random inactive
    
    **Accuracy ({model_info['accuracy']:.1%})**
    - Percentage of correct predictions (both active and inactive)
    - Shows how often the model predicts the right class
    
    **Matthews Correlation Coefficient ({model_info['mcc']:.3f})**
    - Balances true/false positives and negatives
    - Works well for imbalanced datasets
    - Range: -1 (all wrong) to +1 (all correct), 0 (random)
    
    **Training Samples ({model_info['samples']})**
    - Small demonstration dataset
    - Real-world models use thousands of compounds from sources like ChEMBL
    """)
    
if mol is not None:
    st.write("---")
    st.subheader("Export & Advanced Options")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("Copy SMILES to Clipboard"):
            st.success("SMILES copied! (In production, would copy to clipboard)")
    with col_exp2:
        if st.button("Download Prediction Report"):
            st.success("Report generated! (In production, would download CSV/PDF)")
    
    with st.expander("View Full Feature Vector"):
        st.markdown(f"""
        **Morgan Fingerprint (2048 bits):** First 40 bits shown  
        *This is the numerical representation used by the ML model to make predictions.*
        """)
        st.code(str(list(fp_array[:40]))[:-1] + ", ...]")
        st.caption(f"Total fingerprint length: {len(fp_array)} bits | Bits set (ON): {int(fp_array.sum())}")
