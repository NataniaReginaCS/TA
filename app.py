import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🎮 Roblox Game Success Predictor",
    page_icon="🎮",
    layout="wide"
)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# PREMIUM CUSTOM CSS (Mobile-First)
# =====================================================
st.markdown("""
<style>
/* Header */
.main-header {
    font-size: clamp(2rem, 5vw, 3rem) !important;
    font-weight: 800 !important;
    text-align: center;
    margin: 1rem 0 2rem 0 !important;
    color: #1f77b4 !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg,#ffffff,#f8f9fa);
    border-radius:12px;
    padding:18px;
    text-align:center;
    box-shadow:0px 3px 8px rgba(0,0,0,0.08);
    border:1px solid #e6e6e6;
}

.metric-card h3{
    margin:0;
    font-size:0.9rem;
    color:#555;
}

.metric-card h2{
    font-size:1.8rem;
    font-weight:700;
    color:#1f77b4;
}

/* Result Boxes */
.success-box {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(86,171,47,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}
.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(255,107,107,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}

/* Stats */
.stats-metric {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: white !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(31,119,180,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(31,119,180,0.4);
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .metric-card { padding: 1rem 0.75rem !important; margin: 0.5rem 0; }
    .success-box, .error-box { padding: 1.5rem 1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL 
# =====================================================
@st.cache_resource
def load_model_and_artifacts():
    try:
        final_model = joblib.load("final_model.pkl")
        df = joblib.load("processed_df.pkl")
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        y_pred = joblib.load("y_pred.pkl")
        y_prob = joblib.load("y_prob.pkl")
        metrics = joblib.load("metrics.pkl")

        df_imp = None
        try:
            df_imp = joblib.load("feature_importance_df.pkl")
        except:
            pass

        return final_model, df, X_test, y_test, y_pred, y_prob, metrics, df_imp
    except Exception as e:
        st.error(f"🚨 **Model loading failed:** {str(e)}")
        st.stop()

final_model, df, X_test, y_test, y_pred, y_prob, metrics, df_imp = load_model_and_artifacts()
baseline_prob = float(np.mean(y_test))

# Data prep
unique_genres = sorted(df.get("Genre", pd.Series()).dropna().unique().tolist())
unique_ages = sorted(df.get("AgeRecommendation", pd.Series()).dropna().unique().tolist())

# =====================================================
# HEADER
# =====================================================
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Classification</h1>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:#666; font-size:1.2rem; margin-bottom:2rem;'>
    Klasifikasi game populer dalam platform Roblox menggunakan <strong>Random Forest</strong><br>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PERFORMANCE METRICS
# =====================================================
st.markdown("### 📊 **Model Performance Overview**")
col1, col2, col3, col4, col5, col6, col7= st.columns(7)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Accuracy</h3>
        <h2>{metrics['accuracy']:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)



with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>F1 Score</h3>
        <h2>{metrics['f1_score']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Precision</h3>
        <h2>{metrics['precision']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Recall</h3>
        <h2>{metrics['recall']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <h3>F1 Score macro</h3>
        <h2>{metrics['f1_score_macro']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Precision macro</h3>
        <h2>{metrics['precision_macro']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Recall macro</h3>
        <h2>{metrics['recall_macro']:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs([
    "🎮 Game Predictor", 
    "📊 Model Analytics", 
    "📈 Data Explorer"
])
# =====================================================
# TAB 1: PREDICTOR 
# =====================================================
with tab1:
    st.markdown("---")
    st.markdown("#### 🔮 **Input Game Data**")
    
    with st.form("predict_form", clear_on_submit=True):
        col1, col2 = st.columns([1,1])
        with col1:
            genre = st.selectbox("🎨 **Genre**", options=unique_genres)
            age_rec = st.selectbox("👶 **Age Rating**", options=unique_ages)
        with col2:
            game_age = st.number_input("📅 **Game Age** (days)", min_value=0, value=300)
            update_gap = st.number_input("🔄 **Update Gap** (days)", min_value=0, value=30)
        
        col3, col4 = st.columns([1,1])
        with col3:
            visits = st.number_input("👥 **Total Visits**", min_value=0, value=50000)
            favorites = st.number_input("❤️ **Favorites**", min_value=0, value=2500)
        with col4:
            likes = st.number_input("👍 **Likes**", min_value=0, value=5000)
            dislikes = st.number_input("👎 **Dislikes**", min_value=0, value=500)
        
        submitted = st.form_submit_button("🚀 **ANALYZE GAME**", use_container_width=True)
    
    if submitted:
        with st.spinner("🔬 Analyzing game potential..."):
            # Feature engineering 
            favorite_rate = np.log1p(favorites / max(visits, 1))
            engagement_rate = np.log1p((likes + dislikes) / max(visits, 1))
            like_ratio = likes / max(likes + dislikes, 1)
            update_gap_days = np.log1p(update_gap)

            input_df = pd.DataFrame({
                "game_age": [game_age], "update_gap_days": [update_gap_days],
                "favorite_rate": [favorite_rate],
                "engagement_rate": [engagement_rate], "like_ratio": [like_ratio],
                "Genre": [genre], "AgeRecommendation": [age_rec]
            })

            pred = final_model.predict(input_df)[0]
            prob = final_model.predict_proba(input_df)[:, 1][0]

            # =====================================================
            # GENERATE EXPLANATION
            # =====================================================

            feature_cols = [
                "game_age",
                "update_gap_days",
                "favorite_rate",
                "engagement_rate",
                "like_ratio",
                "Genre",
                "AgeRecommendation"
            ]

        # Display result
        if pred == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2>🎉 HIGH SUCCESS POTENTIAL!</h2>
                <h3 style='color:#2d5a2a; font-size:2rem;'>
                    <span class="stats-metric">Probability of success: {prob*100:.1f}%</span>
                </h3>
                <p>🚀 This game has viral potential on Roblox!</p>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>⚠️ Needs Optimization </h2>
                <h3 style='color:#8b1a1a; font-size:2rem;'>
                    <span class="stats-metric">Probability of success: {prob*100:.1f}%</span>
                </h3>
                <p>💡 Improve marketing & engagement strategies</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

        st.markdown("### 💡 **Rekomendasi Peningkatan**")
        st.markdown("Berikut fitur-fitur yang paling berpengaruh terhadap prediksi popularitas game kamu:")

        # Ambil feature importance langsung dari RF model
        preprocessor_step = final_model.named_steps['preprocessor']
        rf_model = final_model.named_steps['model']

        all_feature_names = preprocessor_step.get_feature_names_out()
        importances = rf_model.feature_importances_

        # Buat DataFrame feature importance
        fi_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        transformed_input = preprocessor_step.transform(input_df)
        transformed_input_series = pd.Series(
            transformed_input[0], index=all_feature_names
        )

        actionable_features = {
            "num__update_gap_days": "🔄 Update Gap Days: Perpendek jarak antar update agar pemain tetap tertarik dan kembali bermain.",
            "num__like_ratio": "👍 Like Ratio: Perbaiki kualitas gameplay, visual, dan cerita agar rasio like lebih tinggi dari dislike.",
            "num__favorite_rate": "❤️ Favorite Rate: Buat fitur unik dan konten menarik supaya lebih banyak pemain memfavoritkan game.",
            "num__engagement_rate": "💬 Engagement Rate: Dorong interaksi pemain melalui event, daily quest, atau fitur komunitas.",
            "num__game_age": "📅 Game Age: Pertahankan konsistensi update dan promosi agar game tetap relevan seiring waktu.",
            "cat__Genre_Simulation": "🏗️ Genre Simulation: Tingkatkan realisme dan tambahkan variasi gameplay yang lebih kaya.",
            "cat__Genre_Survival": "🗡️ Genre Survival: Tambahkan tantangan baru, item, dan mekanisme bertahan hidup yang lebih beragam.",
            "cat__Genre_Action": "⚔️ Genre Action: Fokus pada gameplay responsif, kontrol yang mulus, dan mode kompetitif.",
            "cat__Genre_Adventure": "🗺️ Genre Adventure: Tambahkan area baru, cerita menarik, dan misi yang beragam untuk dieksplorasi.",
            "cat__Genre_RPG": "🧙 Genre RPG: Kembangkan sistem karakter, storyline, dan elemen progression yang lebih dalam.",
            "cat__Genre_Obby & Platformer": "🏃 Genre Obby/Platformer: Desain level yang kreatif, menantang, dan bervariasi.",
            "cat__Genre_Party & Casual": "🎉 Genre Party/Casual: Buat game mudah dimainkan bersama teman dan tambahkan mode multiplayer.",
            "cat__Genre_Puzzle": "🧩 Genre Puzzle: Tambahkan teka-teki baru yang inovatif dan sistem hints yang baik.",
            "cat__Genre_Shooter": "🔫 Genre Shooter: Seimbangkan senjata, tambahkan mode kompetitif, dan perbaiki sistem matchmaking.",
            "cat__Genre_Roleplay & Avatar Sim": "👗 Genre Roleplay: Tambahkan opsi kustomisasi yang lebih banyak dan fitur sosial yang menarik.",
            "cat__AgeRecommendation_All Ages": "👶 All Ages: Pastikan konten aman, ramah, dan menarik untuk semua kelompok usia.",
            "cat__AgeRecommendation_Ages 9+": "🧒 Ages 9+: Sesuaikan tingkat kesulitan dan konten agar cocok untuk anak usia 9 tahun ke atas.",
            "cat__AgeRecommendation_Ages 13+": "🧑 Ages 13+: Tambahkan fitur yang lebih kompleks dan menarik untuk remaja usia 13 tahun ke atas.",
        }
        top5_features = fi_df.head(5)

        shown = 0
        for _, row in top5_features.iterrows():
            feat = row['feature']
            importance = row['importance']
            recommendation = actionable_features.get(
                feat,
                f"Optimalkan aspek **{feat.replace('num__','').replace('cat__','').replace('_',' ').title()}** untuk meningkatkan performa game."
            )

            feature_val = transformed_input_series.get(feat, 0)
            if feature_val >= 0:
                status_icon = "✅"
                status_text = "Sudah cukup baik"
            else:
                status_icon = "⚠️"
                status_text = "Perlu ditingkatkan"
            st.markdown(f"""
            <div style='background:#f8f9fa; border-left:4px solid #1f77b4; 
                        padding:12px 16px; border-radius:8px; margin-bottom:10px;'>
                <div style='font-size:0.85rem; color:#888; margin-bottom:4px;'>
                    Importance: {importance:.4f} &nbsp;|&nbsp; {status_icon} {status_text}
                </div>
                <div style='font-size:1rem;'>{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
            shown += 1
        if shown == 0:
            st.info("Semua fitur utama sudah berkontribusi positif. Pertahankan kualitas game!")

        # Saran tambahan berdasarkan hasil prediksi
        st.markdown("---")
        if pred == 1:
            st.success("🚀 **Game kamu sudah berpotensi populer!** Pertahankan kualitas update dan terus tingkatkan engagement pemain.")
        else:
            st.warning("💡 **Tips untuk meningkatkan popularitas:**\n\n"
                    "1. Rutin update game minimal 1–2 minggu sekali\n"
                    "2. Aktif di komunitas Roblox dan media sosial\n"
                    "3. Minta feedback pemain dan perbaiki gameplay\n"
                    "4. Tambahkan event atau konten musiman")


# =====================================================
# TAB 2: ANALYTICS 
# =====================================================
with tab2:
    st.markdown("---")
    
    # Target Distribution
    st.subheader("📊 **Target Distribution**")
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        fig, ax = plt.subplots(figsize=(5, 3))
        target_counts = pd.Series(y_test).value_counts().sort_index()
        colors = ['#ff9999', '#66b3ff']
        bars = ax.bar(target_counts.index, target_counts.values, 
                        color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Success', 'Success'], fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Game Success Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col_desc:
        st.markdown("""
        <div style='font-size:1.1rem;'>
        <b>Distribusi Target</b> menunjukkan jumlah game Roblox yang sukses dan tidak sukses dalam dataset.<br>
        <ul>
            <li><b>Success:</b> Game yang memenuhi kriteria sukses</li>
            <li><b>Not Success:</b> Game yang belum memenuhi kriteria sukses</li>
        </ul>
        Analisis distribusi ini membantu memahami proporsi target sebelum model dilatih.
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix
    st.subheader("🔍 **Confusion Matrix**")
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Success', 'Success'],
                    yticklabels=['Actual Not', 'Actual Yes'],
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontweight='bold', pad=20)
        st.pyplot(fig)
    with col_desc:
        st.markdown("""
        <div style='font-size:1.1rem;'>
        <b>Confusion Matrix</b> menunjukkan jumlah prediksi benar dan salah dari model.<br>
        <ul>
            <li><b>True Positive:</b> Game sukses yang diprediksi sukses</li>
            <li><b>True Negative:</b> Game tidak sukses yang diprediksi tidak sukses</li>
            <li><b>False Positive:</b> Game tidak sukses yang diprediksi sukses</li>
            <li><b>False Negative:</b> Game sukses yang diprediksi tidak sukses</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# TAB 3: DATA EXPLORER
# =====================================================
with tab3:
    st.markdown("---")
    
    # Dataset Cards
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Total Samples", f"{len(df):,}")
    with col2: st.metric("🔧 Features", df.shape[1])
    with col3: st.metric("🗂️ Dataset", "[Roblox Games](https://www.kaggle.com/datasets/jansenccruz/roblox-dataset)")
    
    st.markdown("---")
    
    # Top Age Recommendations
    st.subheader("👶 **Top Age Recommendations**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    <b>Top Age Recommendations</b> menunjukkan usia pengguna yang paling banyak memainkan game Roblox.<br>
    Informasi ini dapat digunakan untuk menyesuaikan konten dan fitur game agar lebih menarik bagi kelompok usia tertentu.
    </div>
    """, unsafe_allow_html=True)

    try:
        test_df = df.loc[X_test.index].copy()  
        test_df['Predicted_Success'] = y_pred

        age_stats = test_df.groupby('AgeRecommendation')['Predicted_Success'].agg([
            'count',           
            'mean'             
        ]).round(3)
        
        age_stats.columns = ['Total Games', 'Success Rate']
        age_stats['Success Rate %'] = (age_stats['Success Rate'] * 100).round(1)
        age_df = age_stats.sort_values('Success Rate', ascending=False).head(10).reset_index()
        
        # Format display
        display_df = age_df[['AgeRecommendation', 'Total Games', 'Success Rate %']]
        display_df.columns = ['Age Recommendation', 'Total Games', 'Success Rate']
        
        st.dataframe(display_df.style.format({
            'Total Games': '{:.0f}', 
            'Success Rate': '{:.1f}%'
        }).background_gradient(subset=['Success Rate'], cmap='plasma'), 
        use_container_width=True, height=200)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Pastikan df, X_test, y_pred sudah selaras")

    # Top Genres 
    st.subheader("🎨 **Top 10 Genres**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    <b>Top Genre</b> menunjukkan genre game Roblox yang paling populer.<br>
    Dengan melihat genre teratas, Anda dapat memahami tren kategori game yang paling diminati dan merencanakan pengembangan konten yang sesuai.
    </div>
    """, unsafe_allow_html=True) 

    try:
        test_df = df.loc[X_test.index].copy()
        test_df['Predicted_Success'] = y_pred
        
        genre_stats = test_df.groupby('Genre')['Predicted_Success'].agg([
            'count', 'mean'
        ]).round(3)
        
        genre_stats.columns = ['Total Games', 'Success Rate']
        genre_stats['Success Rate %'] = (genre_stats['Success Rate'] * 100).round(1)
        genre_df = genre_stats.sort_values('Success Rate', ascending=False).head(10).reset_index()
        
        display_df = genre_df[['Genre', 'Total Games', 'Success Rate %']]
        display_df.columns = ['Genre', 'Total Games', 'Success Rate']
        
        st.dataframe(display_df.style.format({
            'Total Games': '{:.0f}', 
            'Success Rate': '{:.1f}%'
        }).background_gradient(subset=['Success Rate'], cmap='viridis'), 
        use_container_width=True, height=300)
        
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Dataset Statistics 
    st.subheader("📈 **Dataset Statistics**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    <b>Dataset Statistics</b> memberikan ringkasan statistik dari seluruh fitur dalam dataset.<br>
    Anda dapat melihat nilai rata-rata, standar deviasi, minimum, maksimum, dan jumlah unik untuk setiap fitur.<br>
    Informasi ini membantu memahami karakteristik data sebelum melakukan analisis lebih lanjut.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        desc = df.describe(include='all').T.round(2)
        st.dataframe(desc, use_container_width=True, height=400)
    except Exception as e:
        st.warning("📊 Dataset summary temporarily unavailable")
    
    # Feature Importance 
    if df_imp is not None and len(df_imp) > 0:
        st.subheader("🏆 **Top 10 Feature Importance**")
        top_imp = df_imp.nlargest(10, 'Importance')[['Fitur', 'Importance']]
        col1, col2 = st.columns([1, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            top10 = top_imp.sort_values('Importance')
            colors = plt.cm.plasma(np.linspace(0, 1, len(top10)))
            bars = ax.barh(range(len(top10)), top10['Importance'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels([str(f)[:25] + '...' if len(str(f)) > 25 else str(f) 
                    for f in top10['Fitur']], fontsize=9)
            ax.set_xlabel('Importance Score', fontweight='bold')
            ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.0005, i, f'{width:.4f}', va='center', fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            st.markdown("""
            <div style='font-size:1.1rem;'>
            <b>Feature Importance</b> menunjukkan fitur mana yang paling berpengaruh terhadap prediksi kesuksesan game Roblox.<br>
            Fitur dengan skor tertinggi memiliki kontribusi terbesar dalam model Random Forest.<br>
            Gunakan insight ini untuk mengoptimalkan aspek-aspek game yang penting!
            </div>
            """, unsafe_allow_html=True)
            top10 = top_imp.sort_values('Importance')


        top_imp['Importance'] = top_imp['Importance'].round(4)
        st.dataframe(top_imp.style.background_gradient(cmap='plasma'), 
                    use_container_width=True)

st.markdown("---")