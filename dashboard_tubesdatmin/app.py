import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e8eaf0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d3148;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b92b8 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #c8d0ff !important; font-size: 1.8rem !important;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7b88ff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3148;
    }

    /* Info banner */
    .info-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2840 100%);
        border-left: 3px solid #7b88ff;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #b0b8d8;
    }

    /* Cluster badges */
    .badge-red   { background:#ff4b6e22; color:#ff4b6e; border:1px solid #ff4b6e55;
                   border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
    .badge-green { background:#39d98a22; color:#39d98a; border:1px solid #39d98a55;
                   border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }
    .badge-blue  { background:#7b88ff22; color:#7b88ff; border:1px solid #7b88ff55;
                   border-radius:20px; padding:4px 12px; font-size:0.8rem; font-weight:600; }

    /* Tab styling */
    button[data-baseweb="tab"] { color: #8b92b8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #7b88ff !important; }

    h1, h2, h3 { color: #c8d0ff !important; }
    p, li { color: #b0b8d8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1d2e',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#2d3148',
    'axes.labelcolor': '#8b92b8',
    'xtick.color': '#8b92b8',
    'ytick.color': '#8b92b8',
    'text.color': '#c8d0ff',
    'grid.color': '#2d3148',
    'grid.alpha': 0.5,
})

PALETTE = ['#7b88ff', '#39d98a', '#ff4b6e', '#ffb547', '#a78bfa']

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    file_id = "1ObvkV5s1oNVnm4XgW4nTO-3-dYonHb_S"
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    df_raw = pd.read_excel(url)

    df = df_raw.copy()

    # Fix Total Charges
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop Churn Reason
    df = df.drop(columns=['Churn Reason'], errors='ignore')

    # Outlier capping
    for col in ['Tenure Months', 'Monthly Charges', 'Total Charges']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # Drop kolom tidak relevan
    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City',
                 'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
                 'Churn Score', 'CLTV', 'Churn Label']
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df['Churn Value']

    # Encoding
    le = LabelEncoder()
    binary_cols = ['Partner', 'Dependents', 'Phone Service', 'Paperless Billing',
                   'Online Security', 'Online Backup', 'Device Protection',
                   'Tech Support', 'Streaming TV', 'Streaming Movies']
    for col in binary_cols:
        if col in df_model.columns:
            df_model[col] = le.fit_transform(df_model[col])

    if 'Gender' in df_model.columns:
        df_model['Gender'] = le.fit_transform(df_model['Gender'])
    for col in ['Multiple Lines', 'Internet Service', 'Payment Method']:
        if col in df_model.columns:
            df_model[col] = le.fit_transform(df_model[col])
    if 'Contract' in df_model.columns:
        df_model['Contract'] = df['Contract'].map(
            {'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    if 'Senior Citizen' in df_model.columns:
        df_model['Senior Citizen'] = df_model['Senior Citizen'].map(
            {'No': 0, 'Yes': 1}).fillna(df_model['Senior Citizen'])

    # Normalisasi
    scaler = MinMaxScaler()
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    df_model[num_cols] = scaler.fit_transform(df_model[num_cols])

    df_final = df_model.copy()
    df_final['Churn Value'] = y.values

    return df_raw, df_final, scaler

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_kmeans(df_final):
    X = df_final.drop('Churn Value', axis=1)

    # Elbow
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    # Final model K=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # PCA for viz
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Silhouette
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X, clusters)

    # Profile
    df_cl = df_final.copy()
    df_cl['Cluster'] = clusters
    profile = df_cl.groupby('Cluster').mean().round(3)

    return wcss, clusters, X_pca, sil, profile, df_cl

@st.cache_data
def run_logistic(df_final):
    X = df_final.drop('Churn Value', axis=1)
    y = df_final['Churn Value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                    target_names=['No Churn', 'Churn'],
                                    output_dict=True)

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)

    return acc, auc, cm, fpr, tpr, report, coef_df

@st.cache_data
def run_naive_bayes(df_final):
    X = df_final.drop('Churn Value', axis=1)
    y = df_final['Churn Value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                    target_names=['Tidak Churn', 'Churn'],
                                    output_dict=True)

    log_ratio = np.log(nb.theta_[1] + 1e-9) - np.log(nb.theta_[0] + 1e-9)
    feat_imp = pd.DataFrame({
        'Feature': X.columns,
        'Log_Ratio': log_ratio
    }).sort_values('Log_Ratio', ascending=False)

    return acc, auc, cm, fpr, tpr, report, feat_imp

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 ChurnScope")
    st.markdown("<p style='color:#8b92b8;font-size:0.85rem;'>IBM Telco Customer Churn Analysis</p>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigasi",
        ["🏠 Overview", "🔍 Eksplorasi Data", "📦 K-Means Clustering",
         "📈 Logistic Regression", "🧮 Naïve Bayes", "⚖️ Perbandingan Model",
         "🎯 Prediksi Pelanggan"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("<p style='color:#8b92b8;font-size:0.8rem;'>Tugas Besar Data Mining<br>SI-48-10 | Telkom University<br>2025/2026</p>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Memuat dataset dan melatih model..."):
    df_raw, df_final, scaler = load_and_preprocess()
    wcss, clusters, X_pca, sil_score, cluster_profile, df_cl = run_kmeans(df_final)
    lr_acc, lr_auc, lr_cm, lr_fpr, lr_tpr, lr_report, lr_coef = run_logistic(df_final)
    nb_acc, nb_auc, nb_cm, nb_fpr, nb_tpr, nb_report, nb_feat = run_naive_bayes(df_final)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("# 📡 Prediksi Customer Churn")
    st.markdown("<p style='color:#8b92b8;'>Layanan Berbasis Langganan — IBM Telco Dataset</p>",
                unsafe_allow_html=True)
    st.divider()

    # KPI row
    total = len(df_raw)
    churn_n = df_raw['Churn Value'].sum()
    churn_pct = churn_n / total * 100
    no_churn = total - churn_n

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pelanggan", f"{total:,}")
    c2.metric("Pelanggan Churn", f"{int(churn_n):,}", f"{churn_pct:.1f}%")
    c3.metric("Pelanggan Aktif", f"{int(no_churn):,}")
    c4.metric("Rata-rata Tenure", f"{df_raw['Tenure Months'].mean():.0f} bln")

    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Distribusi Churn</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes = [no_churn, churn_n]
        labels = ['Tidak Churn', 'Churn']
        colors = ['#39d98a', '#ff4b6e']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.6),
            textprops={'color': '#c8d0ff', 'fontsize': 11}
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight('bold')
        ax.set_facecolor('#1a1d2e')
        fig.patch.set_facecolor('#1a1d2e')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Performa Model</div>', unsafe_allow_html=True)
        model_data = {
            'Model': ['Logistic Regression', 'Naïve Bayes'],
            'Accuracy': [lr_acc, nb_acc],
            'AUC-ROC': [lr_auc, nb_auc]
        }
        mdf = pd.DataFrame(model_data)

        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(mdf))
        width = 0.35
        bars1 = ax.bar(x - width/2, mdf['Accuracy'], width, label='Accuracy',
                       color='#7b88ff', alpha=0.85, edgecolor='none')
        bars2 = ax.bar(x + width/2, mdf['AUC-ROC'], width, label='AUC-ROC',
                       color='#39d98a', alpha=0.85, edgecolor='none')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(mdf['Model'], fontsize=9)
        ax.set_ylabel('Score')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#c8d0ff')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#c8d0ff')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Tentang Penelitian</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    Penelitian ini menggunakan dataset <b>IBM Telco Customer Churn</b> (7.043 pelanggan, 33 atribut)
    untuk menganalisis dan memprediksi churn pelanggan pada layanan berbasis langganan.
    Tiga metode analisis diterapkan: <b>K-Means Clustering</b> (segmentasi pelanggan),
    <b>Logistic Regression</b>, dan <b>Naïve Bayes</b> (prediksi churn).
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EKSPLORASI DATA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Eksplorasi Data":
    st.markdown("# 🔍 Eksplorasi Data")
    st.markdown(f"<p style='color:#8b92b8;'>{len(df_raw):,} pelanggan · {df_raw.shape[1]} atribut</p>",
                unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Numerik", "📋 Distribusi Kategorikal", "🔗 Korelasi"])

    with tab1:
        st.markdown('<div class="section-title">Distribusi Variabel Numerik</div>',
                    unsafe_allow_html=True)
        
        # --- TAMBAHKAN PEMERIKSAAN DI SINI ---
        # Pastikan data numerik bersih dari spasi sebelum plotting
        df_plot = df_raw.copy()
        num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
        for col in num_cols:
            # Paksa konversi ke numerik, ganti string kosong/spasi menjadi NaN
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
            # Isi NaN dengan median (atau 0, tergantung preferensi Anda)
            df_plot[col] = df_plot[col].fillna(df_plot[col].median())
        # ------------------------------------

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        colors_hist = ['#7b88ff', '#39d98a', '#ffb547']

        for i, col in enumerate(num_cols):
            # Gunakan df_plot yang sudah bersih
            sns.histplot(df_plot[col], kde=True, ax=axes[0, i],
                         color=colors_hist[i], alpha=0.7)
            axes[0, i].set_title(f'Distribusi {col}')
            axes[0, i].grid(alpha=0.3)

            sns.boxplot(x='Churn Label', y=col, data=df_plot, ax=axes[1, i],
                        hue='Churn Label', palette={'No': '#39d98a', 'Yes': '#ff4b6e'},
                        legend=False)
            axes[1, i].set_title(f'{col} vs Churn')
            axes[1, i].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown('<div class="section-title">Variabel Kategorikal vs Churn</div>',
                    unsafe_allow_html=True)
        cat_cols = ['Gender', 'Contract', 'Internet Service', 'Payment Method', 'Senior Citizen']
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            sns.countplot(x=col, hue='Churn Label', data=df_raw, ax=axes[i],
                          palette={'No': '#39d98a', 'Yes': '#ff4b6e'})
            axes[i].set_title(f'{col} vs Churn')
            axes[i].tick_params(axis='x', rotation=20)
            axes[i].grid(axis='y', alpha=0.3)

        axes[-1].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown('<div class="section-title">Heatmap Korelasi Antar Fitur</div>',
                    unsafe_allow_html=True)
        corr_df = df_final.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.zeros_like(corr_df, dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(corr_df, cmap='coolwarm', center=0, ax=ax,
                    linewidths=0.3, linecolor='#2d3148',
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Heatmap Korelasi', fontsize=13, pad=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-title">Korelasi Fitur terhadap Churn Value</div>',
                    unsafe_allow_html=True)
        corr_target = df_final.corr(numeric_only=True)['Churn Value'] \
            .drop('Churn Value').sort_values()

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ['#ff4b6e' if v > 0 else '#39d98a' for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=colors, alpha=0.85)
        ax.axvline(0, color='#8b92b8', linewidth=0.8)
        ax.set_xlabel('Nilai Korelasi')
        ax.set_title('Korelasi Fitur vs Churn Value')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📦 K-Means Clustering":
    st.markdown("# 📦 K-Means Clustering")
    st.markdown("<p style='color:#8b92b8;'>Segmentasi pelanggan berdasarkan karakteristik</p>",
                unsafe_allow_html=True)
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Cluster", "3")
    c2.metric("Silhouette Score", f"{sil_score:.4f}")
    c3.metric("Total Pelanggan", f"{len(df_cl):,}")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Metode Elbow</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, 11), wcss, marker='o', color='#7b88ff',
                markerfacecolor='#39d98a', markersize=8, linewidth=2)
        ax.axvline(3, color='#ff4b6e', linestyle='--', alpha=0.7, label='K optimal = 3')
        ax.set_xlabel('Jumlah Cluster (K)')
        ax.set_ylabel('WCSS')
        ax.set_title('Elbow Method')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Visualisasi Cluster (PCA)</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        cluster_colors = ['#7b88ff', '#39d98a', '#ff4b6e']
        cluster_labels_viz = ['Cluster 0 (Loyal)', 'Cluster 1 (Biaya Rendah)', 'Cluster 2 (Berisiko)']
        for ci in range(3):
            mask = clusters == ci
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=cluster_colors[ci], alpha=0.5, s=10,
                       label=cluster_labels_viz[ci])
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title('Hasil Clustering K-Means')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # Cluster profile
    st.markdown('<div class="section-title">Profil Cluster</div>', unsafe_allow_html=True)

    cluster_info = [
        ("Cluster 0", "Pelanggan Loyal", "#7b88ff",
         "Tenure tertinggi (0.727), layanan lengkap, churn rate 17.4%"),
        ("Cluster 1", "Biaya Rendah", "#39d98a",
         "Monthly Charges terendah (0.204), beban ringan, churn rate 12.2%"),
        ("Cluster 2", "Berisiko Tinggi", "#ff4b6e",
         "Tenure pendek (0.232), minim layanan tambahan, churn rate 45.3%"),
    ]

    cols = st.columns(3)
    churn_rates = [0.174, 0.122, 0.453]
    counts = df_cl['Cluster'].value_counts().sort_index()

    for i, (col, (cid, label, color, desc)) in enumerate(zip(cols, cluster_info)):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1d2e, #1e2235);
                        border: 1px solid {color}44; border-top: 3px solid {color};
                        border-radius: 10px; padding: 16px; margin-bottom: 8px;">
                <p style="color:{color}; font-weight:700; font-size:1rem; margin:0">{cid}</p>
                <p style="color:#c8d0ff; font-size:0.95rem; margin:4px 0">{label}</p>
                <p style="color:#8b92b8; font-size:0.82rem; margin:0">{desc}</p>
                <p style="color:#c8d0ff; font-size:1.4rem; font-weight:700; margin:8px 0 0 0">
                    {counts[i]:,} <span style="font-size:0.75rem; color:#8b92b8;">pelanggan</span>
                </p>
                <p style="color:{color}; font-size:0.85rem; font-weight:600; margin:4px 0 0 0">
                    Churn Rate: {churn_rates[i]*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Distribution chart
    st.markdown('<div class="section-title">Distribusi Cluster</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie
    axes[0].pie(counts.values, labels=[f'Cluster {i}' for i in range(3)],
                colors=cluster_colors, autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.6),
                textprops={'color': '#c8d0ff', 'fontsize': 10})
    axes[0].set_title('Distribusi Jumlah Pelanggan per Cluster')

    # Churn rate bar
    axes[1].bar(['Cluster 0\nLoyal', 'Cluster 1\nBiaya Rendah', 'Cluster 2\nBerisiko'],
                churn_rates, color=cluster_colors, alpha=0.85, edgecolor='none')
    axes[1].set_ylabel('Churn Rate')
    axes[1].set_title('Tingkat Churn per Cluster')
    axes[1].grid(axis='y', alpha=0.3)
    for j, v in enumerate(churn_rates):
        axes[1].text(j, v + 0.01, f'{v*100:.1f}%', ha='center', fontsize=10,
                     color='#c8d0ff', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Logistic Regression":
    st.markdown("# 📈 Logistic Regression")
    st.markdown("<p style='color:#8b92b8;'>Model klasifikasi dengan class_weight=balanced</p>",
                unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{lr_acc:.4f}")
    c2.metric("AUC-ROC", f"{lr_auc:.4f}")
    c3.metric("Recall (Churn)", f"{lr_report['Churn']['recall']:.4f}")
    c4.metric("Precision (Churn)", f"{lr_report['Churn']['precision']:.4f}")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'], ax=ax,
                    linewidths=0.5, linecolor='#2d3148')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — Logistic Regression')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(lr_fpr, lr_tpr, color='#7b88ff', lw=2,
                label=f'LR (AUC = {lr_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.fill_between(lr_fpr, lr_tpr, alpha=0.1, color='#7b88ff')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Logistic Regression')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown('<div class="section-title">Feature Importance (Koefisien)</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = ['#ff4b6e' if c > 0 else '#39d98a' for c in lr_coef['Coefficient']]
    ax.barh(lr_coef['Feature'], lr_coef['Coefficient'], color=colors_fi, alpha=0.85)
    ax.axvline(0, color='#8b92b8', linewidth=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance — Logistic Regression')
    ax.grid(axis='x', alpha=0.3)

    red_patch = mpatches.Patch(color='#ff4b6e', label='Mendorong Churn')
    green_patch = mpatches.Patch(color='#39d98a', label='Mencegah Churn')
    ax.legend(handles=[red_patch, green_patch])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NAIVE BAYES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧮 Naïve Bayes":
    st.markdown("# 🧮 Naïve Bayes Classifier")
    st.markdown("<p style='color:#8b92b8;'>Gaussian Naïve Bayes — pendekatan probabilistik</p>",
                unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{nb_acc:.4f}")
    c2.metric("AUC-ROC", f"{nb_auc:.4f}")
    c3.metric("Recall (Churn)", f"{nb_report['Churn']['recall']:.4f}")
    c4.metric("Precision (Churn)", f"{nb_report['Churn']['precision']:.4f}")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Tidak Churn', 'Churn'],
                    yticklabels=['Tidak Churn', 'Churn'], ax=ax,
                    linewidths=0.5, linecolor='#2d3148')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — Naïve Bayes')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(nb_fpr, nb_tpr, color='#ffb547', lw=2,
                label=f'NB (AUC = {nb_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.fill_between(nb_fpr, nb_tpr, alpha=0.1, color='#ffb547')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve — Naïve Bayes')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance NB
    st.markdown('<div class="section-title">Feature Importance (Log Ratio)</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_nb = ['#ff4b6e' if v > 0 else '#39d98a' for v in nb_feat['Log_Ratio']]
    ax.barh(nb_feat['Feature'], nb_feat['Log_Ratio'], color=colors_nb, alpha=0.85)
    ax.axvline(0, color='#8b92b8', linewidth=0.8)
    ax.set_xlabel('Log Ratio')
    ax.set_title('Feature Importance — Naïve Bayes (Log Ratio)')
    ax.grid(axis='x', alpha=0.3)

    red_patch = mpatches.Patch(color='#ff4b6e', label='Mendorong Churn')
    green_patch = mpatches.Patch(color='#39d98a', label='Mencegah Churn')
    ax.legend(handles=[red_patch, green_patch])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ Perbandingan Model":
    st.markdown("# ⚖️ Perbandingan Model")
    st.markdown("<p style='color:#8b92b8;'>Logistic Regression vs Naïve Bayes</p>",
                unsafe_allow_html=True)
    st.divider()

    # Comparison table
    comp_data = {
        'Metrik': ['Accuracy', 'Precision (Churn)', 'Recall (Churn)', 'F1-Score (Churn)', 'AUC-ROC'],
        'Logistic Regression': [
            lr_acc,
            lr_report['Churn']['precision'],
            lr_report['Churn']['recall'],
            lr_report['Churn']['f1-score'],
            lr_auc
        ],
        'Naïve Bayes': [
            nb_acc,
            nb_report['Churn']['precision'],
            nb_report['Churn']['recall'],
            nb_report['Churn']['f1-score'],
            nb_auc
        ]
    }
    comp_df = pd.DataFrame(comp_data)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Tabel Perbandingan</div>', unsafe_allow_html=True)
        styled = comp_df.copy()
        styled['Logistic Regression'] = styled['Logistic Regression'].map('{:.4f}'.format)
        styled['Naïve Bayes'] = styled['Naïve Bayes'].map('{:.4f}'.format)
        st.dataframe(styled, hide_index=True, use_container_width=True)

        winner = "Logistic Regression" if lr_auc > nb_auc else "Naïve Bayes"
        st.markdown(f"""
        <div class="info-card">
        🏆 <b>{winner}</b> unggul secara keseluruhan dengan AUC-ROC lebih tinggi
        ({max(lr_auc, nb_auc):.4f} vs {min(lr_auc, nb_auc):.4f}).
        Untuk prediksi churn, <b>Recall</b> menjadi metrik prioritas karena
        False Negative (churn tidak terdeteksi) memiliki dampak bisnis lebih besar.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Radar / Bar Perbandingan</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        metrics = comp_data['Metrik']
        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, comp_data['Logistic Regression'], w,
               label='Logistic Regression', color='#7b88ff', alpha=0.85)
        ax.bar(x + w/2, comp_data['Naïve Bayes'], w,
               label='Naïve Bayes', color='#ffb547', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score')
        ax.set_title('Perbandingan Metrik Evaluasi')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ROC comparison
    st.markdown('<div class="section-title">ROC Curve — Perbandingan</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lr_fpr, lr_tpr, color='#7b88ff', lw=2,
            label=f'Logistic Regression (AUC={lr_auc:.4f})')
    ax.plot(nb_fpr, nb_tpr, color='#ffb547', lw=2, linestyle='--',
            label=f'Naïve Bayes (AUC={nb_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.fill_between(lr_fpr, lr_tpr, alpha=0.08, color='#7b88ff')
    ax.fill_between(nb_fpr, nb_tpr, alpha=0.08, color='#ffb547')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDIKSI PELANGGAN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Prediksi Pelanggan":
    st.markdown("# 🎯 Prediksi Churn Pelanggan Baru")
    st.markdown("<p style='color:#8b92b8;'>Masukkan data pelanggan untuk mendapatkan prediksi churn</p>",
                unsafe_allow_html=True)
    st.divider()

    # Retrain model for prediction (use LR as primary)
    @st.cache_resource
    def get_trained_model(df_final):
        X = df_final.drop('Churn Value', axis=1)
        y = df_final['Churn Value']
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X, y)
        nb = GaussianNB()
        nb.fit(X, y)
        return lr, nb, X.columns.tolist()

    lr_model, nb_model, feature_cols = get_trained_model(df_final)

    st.markdown('<div class="section-title">Data Pelanggan</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informasi Demografis**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.markdown("**Informasi Layanan**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Informasi Akun**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        tenure = st.slider("Tenure Months", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges = st.slider("Total Charges ($)", 18.0, 8700.0,
                                   float(tenure * monthly_charges), 10.0)

    st.markdown("")
    if st.button("🔮 Prediksi Churn", type="primary", use_container_width=False):
        # Encode input
        le = LabelEncoder()

        def encode_binary(val, yes_val='Yes'):
            return 1 if val == yes_val else 0

        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        le_ml = LabelEncoder()

        ml_map = {
            'Multiple Lines': {'No': 0, 'No phone service': 1, 'Yes': 2},
            'Internet Service': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'Payment Method': {
                'Bank transfer (automatic)': 0,
                'Credit card (automatic)': 1,
                'Electronic check': 2,
                'Mailed check': 3
            }
        }

        # Normalize numeric
        tenure_n, monthly_n, total_n = scaler.transform(
            [[tenure, monthly_charges, total_charges]])[0]

        input_dict = {
            'Senior Citizen': 1 if senior == 'Yes' else 0,
            'Partner': encode_binary(partner),
            'Dependents': encode_binary(dependents),
            'Phone Service': encode_binary(phone_service),
            'Multiple Lines': ml_map['Multiple Lines'][multiple_lines],
            'Internet Service': ml_map['Internet Service'][internet_service],
            'Online Security': encode_binary(online_security),
            'Online Backup': encode_binary(online_backup),
            'Device Protection': encode_binary(device_protection),
            'Tech Support': encode_binary(tech_support),
            'Streaming TV': encode_binary(streaming_tv),
            'Streaming Movies': encode_binary(streaming_movies),
            'Contract': contract_map[contract],
            'Paperless Billing': encode_binary(paperless_billing),
            'Payment Method': ml_map['Payment Method'][payment_method],
            'Gender': 1 if gender == 'Male' else 0,
            'Tenure Months': tenure_n,
            'Monthly Charges': monthly_n,
            'Total Charges': total_n,
        }

        # Reorder to match feature_cols
        X_input = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_cols}])

        lr_prob = lr_model.predict_proba(X_input)[0][1]
        nb_prob = nb_model.predict_proba(X_input)[0][1]
        avg_prob = (lr_prob + nb_prob) / 2

        st.divider()
        st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Logistic Regression", f"{lr_prob*100:.1f}%",
                  "⚠️ Berisiko Churn" if lr_prob > 0.5 else "✅ Aman")
        r2.metric("Naïve Bayes", f"{nb_prob*100:.1f}%",
                  "⚠️ Berisiko Churn" if nb_prob > 0.5 else "✅ Aman")
        r3.metric("Rata-rata Probabilitas", f"{avg_prob*100:.1f}%",
                  "⚠️ Berisiko Churn" if avg_prob > 0.5 else "✅ Aman")

        if avg_prob > 0.5:
            st.error(f"""
            ⚠️ **Pelanggan ini diprediksi BERISIKO CHURN** dengan probabilitas rata-rata **{avg_prob*100:.1f}%**.

            Rekomendasi: Tawarkan diskon kontrak jangka panjang, aktifkan layanan tambahan (Online Security,
            Tech Support), dan lakukan outreach proaktif sebelum pelanggan memutuskan berhenti.
            """)
        else:
            st.success(f"""
            ✅ **Pelanggan ini diprediksi TIDAK CHURN** dengan probabilitas churn rata-rata **{avg_prob*100:.1f}%**.

            Pelanggan ini tergolong loyal. Pertahankan kualitas layanan dan pertimbangkan program reward
            untuk meningkatkan loyalitas jangka panjang.
            """)

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 2))
        bar_color = '#ff4b6e' if avg_prob > 0.5 else '#39d98a'
        ax.barh(['Probabilitas Churn'], [avg_prob], color=bar_color, alpha=0.85)
        ax.barh(['Probabilitas Churn'], [1 - avg_prob], left=[avg_prob],
                color='#2d3148', alpha=0.5)
        ax.axvline(0.5, color='#ffb547', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilitas')
        ax.text(avg_prob / 2, 0, f'{avg_prob*100:.1f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=12)
        ax.set_title('Probabilitas Churn (Rata-rata)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
