import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Advanced Petrophysical AI Platform", layout="wide", page_icon="🛢️")

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    footer {
        visibility: hidden;
    }
    .signature {
        text-align: center;
        padding: 20px;
        font-family: 'Courier New', Courier, monospace;
        color: #555;
        border-top: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Loading Models
@st.cache_resource
def load_models():
    # Ensure these files are in the same directory
    classifier = joblib.load('lithology_model.pkl')
    regressor = joblib.load('lithology_model.pkl')
    return classifier, regressor

try:
    classifier, regressor = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure .pkl files are present.")

# 3. Sidebar Branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3011/3011689.png", width=100)
    st.title("Control Panel")
    st.markdown("---")
    st.info("Developed for high-precision Well Log analysis and data recovery.")
    st.write("---")
    st.markdown("👤 **Project Lead:**")
    st.markdown("### Eng. Sulaiman Kudaimi")
    st.markdown("*Petroleum Data Scientist*")

# 4. Main Interface Header
st.title("🛢️ Advanced Petrophysical Analysis Platform")
st.subheader("AI-Driven Well Logging & Synthetic Data Generation")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Well Log Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.upper().strip() for c in df.columns]
    
    # 5. Smart Data Recovery Logic
    if 'RHOB' not in df.columns:
        st.warning("⚠️ RHOB (Density) log missing! AI is generating synthetic values...")
        X_reg = df[['DEPTH', 'GR']]
        X_reg.columns = ['Depth', 'GR'] 
        df['RHOB'] = regressor.predict(X_reg)
        st.success("✅ Synthetic RHOB log generated successfully.")
    
    # 6. AI Lithology Classification
    X_cls = df[['DEPTH', 'GR']]
    X_cls.columns = ['Depth', 'GR']
    df['Lithology_Predicted'] = ["Sandstone" if p == 1 else "Shale" for p in classifier.predict(X_cls)]

    # 7. Professional Visualization
    tab1, tab2 = st.tabs(["📊 Data Preview", "📈 Graphical Log Plots"])

    with tab1:
        st.write("### Processed Well Data")
        cols_to_show = ['DEPTH', 'GR', 'RHOB', 'Lithology_Predicted']
        st.dataframe(df[cols_to_show].head(500), use_container_width=True)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Export Final Report (CSV)", csv, "Sulaiman_Well_Report.csv", "text/csv")

    with tab2:
        col_gr, col_rhob = st.columns(2)
        
        with col_gr:
            st.write("#### Gamma Ray (GR) Log")
            fig1, ax1 = plt.subplots(figsize=(4, 10))
            ax1.plot(df['GR'], df['DEPTH'], color='green', linewidth=1)
            ax1.set_ylim(df['DEPTH'].max(), df['DEPTH'].min())
            ax1.set_ylabel("Depth (m)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with col_rhob:
            st.write("#### Density (RHOB) Log")
            fig2, ax2 = plt.subplots(figsize=(4, 10))
            ax2.plot(df['RHOB'], df['DEPTH'], color='red', linewidth=1)
            ax2.set_ylim(df['DEPTH'].max(), df['DEPTH'].min())
            ax2.set_ylabel("Depth (m)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

# Footer Signature
st.markdown("""
    <div class="signature">
        Designed & Developed by <b>Eng. Sulaiman Kudaimi</b> | 2026 AI Petroleum Initiative
    </div>
    """, unsafe_allow_html=True)
