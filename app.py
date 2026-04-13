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
# CSS 
# =====================================================
st.markdown("""
<style>
.main-header {
    font-size: clamp(2rem, 5vw, 3rem) !important;
    font-weight: 800 !important;
    text-align: center;
    margin: 1rem 0 2rem 0 !important;
    color: #1f77b4 !important;
}
.metric-card {
    background: linear-gradient(135deg,#ffffff,#f8f9fa);
    border-radius:12px;
    padding:18px;
    text-align:center;
    box-shadow:0px 3px 8px rgba(0,0,0,0.08);
    border:1px solid #e6e6e6;
}
.metric-card h3{ margin:0; font-size:0.9rem; color:#555; }
.metric-card h2{ font-size:1.8rem; font-weight:700; color:#1f77b4; }
.success-box {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 2rem; border-radius: 20px; text-align: center;
    box-shadow: 0 12px 48px rgba(86,171,47,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}
.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 2rem; border-radius: 20px; text-align: center;
    box-shadow: 0 12px 48px rgba(255,107,107,0.3);
    border: 3px solid rgba(255,255,255,0.3);
}
.stats-metric {
    font-size: 2.5rem !important; font-weight: 900 !important;
    color: white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #1f77b4 0%, #4a90e2 100%);
    color: white; border-radius: 12px; border: none;
    padding: 0.75rem 2rem; font-weight: 600; font-size: 1.1rem;
    transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(31,119,180,0.3);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(31,119,180,0.4); }
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

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_target_column(df):
    """Detect target column name"""
    possible_targets = ['target', 'Predicted_Success', 'success']
    for col in possible_targets:
        if col in df.columns:
            return col
    return None

def safe_column_access(df, col_name, default=None):
    """Safely access dataframe columns"""
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default] * len(df))

# =====================================================
# BENCHMARK GAME POPULER
# =====================================================
@st.cache_data
def get_contextual_benchmark(df, genre, age):
    target_col = get_target_column(df)
    if target_col is None:
        return None
    
    # Filter popular games
    popular = df[df[target_col] == 1].copy()
    
    # Filter by genre and age
    subset = popular[
        (popular['Genre'] == genre) &
        (popular['AgeRecommendation'] == age)
    ].copy()
    
    if len(subset) == 0:
        return None
    
    # Calculate derived metrics safely
    subset['_fav_rate'] = safe_column_access(subset, 'Favorites', 0) / safe_column_access(subset, 'Visits', 1).replace(0, 1)
    subset['_eng_rate'] = (safe_column_access(subset, 'Likes', 0) + safe_column_access(subset, 'Dislikes', 0)) / safe_column_access(subset, 'Visits', 1).replace(0, 1)
    subset['_like_ratio'] = safe_column_access(subset, 'Likes', 0) / (safe_column_access(subset, 'Likes', 0) + safe_column_access(subset, 'Dislikes', 0) + 1)

    return {
        'game_age'        : subset['game_age'].median(),
        'update_gap_days' : subset['update_gap_days'].median(),
        'favorite_rate'   : subset['_fav_rate'].median(),
        'engagement_rate' : subset['_eng_rate'].median(),
        'like_ratio'      : subset['_like_ratio'].median(),
    }

# =====================================================
# RECOMMENDATION ENGINE
# =====================================================
def generate_recommendations(user_vals: dict, benchmark: dict) -> list:
    recs = []

    # ── 1. like_ratio ──────────────────────────────────────────────
    user_lr = user_vals.get('like_ratio', 0)
    bench_lr = benchmark.get('like_ratio', 0.5)
    if user_lr < bench_lr * 0.85:
        pct_gap = (bench_lr - user_lr) / bench_lr * 100
        recs.append({
            'icon': '👍',
            'title': 'Like Ratio Perlu Ditingkatkan',
            'detail': (
                f"Like ratio kamu **{user_lr:.1%}**, sementara game populer rata-rata **{bench_lr:.1%}**. "
                f"Artinya ada selisih sekitar **{pct_gap:.0f}%** dari standar game populer. "
                "Coba minta feedback dari pemain dan perbaiki aspek gameplay yang sering dikritik."
            ),
            'priority': (bench_lr - user_lr)
        })

    # ── 2. update_gap_days ─────────────────────────────────────────
    user_ug = user_vals.get('update_gap_days', 30)
    bench_ug = benchmark.get('update_gap_days', 30)
    if user_ug > bench_ug * 1.5:
        recs.append({
            'icon': '🔄',
            'title': 'Frekuensi Update Terlalu Jarang',
            'detail': (
                f"Kamu terakhir update **{user_ug:.0f} hari** yang lalu, sedangkan game populer "
                f"biasanya update setiap **{bench_ug:.0f} hari**. "
                "Pemain cenderung meninggalkan game yang jarang diperbarui."
            ),
            'priority': (user_ug - bench_ug)
        })

    # ── 3. favorite_rate ───────────────────────────────────────────
    user_fr = user_vals.get('favorite_rate', 0)
    bench_fr = benchmark.get('favorite_rate', 0.05)
    if user_fr < bench_fr * 0.75:
        pct_gap = (bench_fr - user_fr) / bench_fr * 100
        recs.append({
            'icon': '❤️',
            'title': 'Tingkat Favorit Masih Rendah',
            'detail': (
                f"Favorite rate kamu **{user_fr:.3f}**, sedangkan game populer rata-rata **{bench_fr:.3f}**. "
                "Tambahkan fitur yang mendorong pemain menyimpan game ke favorit."
            ),
            'priority': (bench_fr - user_fr) * 100
        })

    # ── 4. engagement_rate ─────────────────────────────────────────
    user_er = user_vals.get('engagement_rate', 0)
    bench_er = benchmark.get('engagement_rate', 0.1)
    if user_er < bench_er * 0.75:
        pct_gap = (bench_er - user_er) / bench_er * 100
        recs.append({
            'icon': '💬',
            'title': 'Engagement Pemain Masih Rendah',
            'detail': (
                f"Engagement rate kamu **{user_er:.4f}**, sedangkan game populer rata-rata **{bench_er:.4f}**. "
                "Dorong interaksi pemain dengan menambahkan tombol rating yang menonjol."
            ),
            'priority': (bench_er - user_er) * 1000
        })
    
    recs.sort(key=lambda x: x['priority'], reverse=True)
    return recs

# =====================================================
# LOAD DATA
# =====================================================
try:
    final_model, df, X_test, y_test, y_pred, y_prob, metrics, df_imp = load_model_and_artifacts()
    
    # Get unique values safely
    unique_genres = sorted(df['Genre'].dropna().unique().tolist()) if 'Genre' in df.columns else []
    unique_ages = sorted(df['AgeRecommendation'].dropna().unique().tolist()) if 'AgeRecommendation' in df.columns else []
    
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# =====================================================
# HEADER
# =====================================================
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Classification</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#666; font-size:1.2rem; margin-bottom:2rem;'>
    Klasifikasi game populer dalam platform Roblox menggunakan <strong>Random Forest</strong>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PERFORMANCE METRICS
# =====================================================
st.markdown("### 📊 **Model Performance Overview**")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

metric_items = [
    ("Accuracy", f"{metrics.get('accuracy', 0):.1%}"),
    ("F1 Score", f"{metrics.get('f1_score', 0):.3f}"),
    ("Precision", f"{metrics.get('precision', 0):.3f}"),
    ("Recall", f"{metrics.get('recall', 0):.3f}"),
    ("F1 Score macro", f"{metrics.get('f1_score_macro', 0):.3f}"),
    ("Precision macro", f"{metrics.get('precision_macro', 0):.3f}"),
    ("Recall macro", f"{metrics.get('recall_macro', 0):.3f}"),
]

for col, (label, val) in zip([col1,col2,col3,col4,col5,col6,col7], metric_items):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3><h2>{val}</h2>
        </div>""", unsafe_allow_html=True)

# =====================================================
# TABS
# =====================================================
st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎮 Game Predictor", "📊 Model Analytics", "📈 Data Explorer"])

# ─────────────────────────────────────────────────────
# TAB 1 : PREDICTOR
# ─────────────────────────────────────────────────────
with tab1:
    st.markdown("---")
    st.markdown("#### 🔮 **Input Game Data**")

    if not unique_genres or not unique_ages:
        st.warning("⚠️ Genre or Age data not available. Using defaults.")
        unique_genres = ['All Genres']
        unique_ages = ['All Ages']

    with st.form("predict_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            genre = st.selectbox("🎨 **Genre**", options=unique_genres)
            age_rec = st.selectbox("👶 **Age Rating**", options=unique_ages)
        with c2:
            game_age = st.number_input("📅 **Game Age** (days)", min_value=0, value=300)
            update_gap = st.number_input("🔄 **Days Since Last Update**", min_value=0, value=30)

        c3, c4 = st.columns(2)
        with c3:
            visits = st.number_input("👥 **Total Visits**", min_value=0, value=50000)
            favorites = st.number_input("❤️ **Favorites**", min_value=0, value=2500)
        with c4:
            likes = st.number_input("👍 **Likes**", min_value=0, value=5000)
            dislikes = st.number_input("👎 **Dislikes**", min_value=0, value=500)

        submitted = st.form_submit_button("🚀 **ANALYZE GAME**", use_container_width=True)

    if submitted:
        with st.spinner("🔬 Analyzing game potential..."):
            fav_rate = favorites / max(visits, 1)
            eng_rate = (likes + dislikes) / max(visits, 1)
            like_ratio = likes / max(likes + dislikes, 1)

            input_df = pd.DataFrame({
                "game_age": [game_age],
                "update_gap_days": [update_gap],
                "favorite_rate": [fav_rate],
                "engagement_rate": [eng_rate],
                "like_ratio": [like_ratio],
                "Genre": [genre],
                "AgeRecommendation": [age_rec],
            })

            pred = final_model.predict(input_df)[0]
            prob = final_model.predict_proba(input_df)[:, 1][0]

        # Results
        if pred == 1:
            st.markdown(f"""
            <div class="success-box">
                <h2>🎉 HIGH SUCCESS POTENTIAL!</h2>
                <h3 style='color:#2d5a2a; font-size:2rem;'>
                    <span class="stats-metric">Probability: {prob*100:.1f}%</span>
                </h3>
                <p>🚀 This game has viral potential on Roblox!</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>⚠️ Needs Optimization</h2>
                <h3 style='color:#8b1a1a; font-size:2rem;'>
                    <span class="stats-metric">Probability: {prob*100:.1f}%</span>
                </h3>
                <p>💡 See recommendations below!</p>
            </div>""", unsafe_allow_html=True)

        # Recommendations
        user_vals = {
            'like_ratio': like_ratio,
            'update_gap_days': update_gap,
            'favorite_rate': fav_rate,
            'engagement_rate': eng_rate,
        }

        benchmark = get_contextual_benchmark(df, genre, age_rec)
        if benchmark is None:
            st.warning("⚠️ No benchmark data found. Using dataset averages.")
            benchmark = {
                'like_ratio': 0.5,
                'update_gap_days': 30,
                'favorite_rate': 0.05,
                'engagement_rate': 0.1,
            }
        
        recs = generate_recommendations(user_vals, benchmark)

        if pred == 0 and recs:
            st.markdown("### 🛠️ **Improvement Guide**")
            for r in recs:
                st.markdown(f"""
                <div style='background:#fff8f0; border-left:5px solid #ff7f0e;
                            padding:14px 18px; border-radius:10px; margin-bottom:12px;'>
                    <b style='font-size:1.05rem;'>{r['icon']} {r['title']}</b><br>
                    <span style='color:#444; font-size:0.97rem;'>{r['detail']}</span>
                </div>""", unsafe_allow_html=True)
        elif pred == 1 and recs:
            st.markdown("### ✨ **Optimization Opportunities**")
            for r in recs:
                st.markdown(f"""
                <div style='background:#f0f8ff; border-left:5px solid #1f77b4;
                            padding:14px 18px; border-radius:10px; margin-bottom:12px;'>
                    <b style='font-size:1.05rem;'>{r['icon']} {r['title']}</b><br>
                    <span style='color:#444; font-size:0.97rem;'>{r['detail']}</span>
                </div>""", unsafe_allow_html=True)

        # Benchmark info
        with st.expander("📐 Benchmark Reference"):
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Like Ratio", f"{benchmark['like_ratio']:.1%}")
            bcol2.metric("Update Gap", f"{benchmark['update_gap_days']:.0f} days")
            bcol3.metric("Favorite Rate", f"{benchmark['favorite_rate']:.4f}")
            bcol4.metric("Engagement Rate", f"{benchmark['engagement_rate']:.4f}")

# ─────────────────────────────────────────────────────
# TAB 2 : MODEL ANALYTICS
# ─────────────────────────────────────────────────────
with tab2:
    st.markdown("---")

    st.subheader("📊 **Target Distribution**")
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        fig, ax = plt.subplots(figsize=(5, 3))
        target_counts = pd.Series(y_test).value_counts().sort_index()
        bars = ax.bar(target_counts.index, target_counts.values,
                    color=['#ff9999','#66b3ff'], alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Popular', 'Popular'], fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Game Popularity Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2., h+20, f'{int(h)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    with col_desc:
        st.markdown("""
        <div style='font-size:1.1rem;'>
        <b>Distribusi Target</b> menunjukkan jumlah game Roblox yang populer dan tidak populer dalam data pengujian.<br><br>
        <ul>
            <li><b>Popular:</b> Game yang masuk persentil ke-75 ke atas berdasarkan jumlah pemain aktif</li>
            <li><b>Not Popular:</b> Game yang berada di bawah persentil ke-75</li>
        </ul>
        Dataset memiliki ketidakseimbangan kelas (~75:25), itulah mengapa F1-score macro digunakan sebagai metrik utama.
        </div>""", unsafe_allow_html=True)

    st.subheader("🔍 **Confusion Matrix**")
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Popular', 'Popular'],
                    yticklabels=['Actual: Not Popular', 'Actual: Popular'],
                    ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontweight='bold', pad=20)
        st.pyplot(fig)
    with col_desc:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div style='font-size:1.1rem;'>
        <b>Confusion Matrix</b> menunjukkan detail prediksi model:<br><br>
        <ul>
            <li>✅ <b>True Positive ({tp:,}):</b> Game populer yang berhasil diprediksi populer</li>
            <li>✅ <b>True Negative ({tn:,}):</b> Game tidak populer yang berhasil diprediksi tidak populer</li>
            <li>⚠️ <b>False Positive ({fp:,}):</b> Game tidak populer yang keliru diprediksi populer</li>
            <li>❌ <b>False Negative ({fn:,}):</b> Game populer yang gagal terdeteksi</li>
        </ul>
        Model lebih diprioritaskan untuk meminimalkan False Negative (game populer yang tidak terdeteksi),
        sehingga <i>recall</i> kelas populer dijaga tetap tinggi.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# TAB 3 : DATA EXPLORER
# ─────────────────────────────────────────────────────
with tab3:
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 Total Samples", f"{len(df):,}")
    with col2: st.metric("🔧 Features",      df.shape[1])
    with col3: st.metric("🗂️ Dataset Source", "[Kaggle Roblox Dataset](https://www.kaggle.com/datasets/jansenccruz/roblox-dataset)")
    st.markdown("---")

    # Age Recommendations
    st.subheader("👶 **Success Rate by Age Recommendation**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    Menampilkan <b>tingkat keberhasilan prediksi populer</b> berdasarkan kategori rekomendasi usia.
    Informasi ini membantu pengembang memahami segmen usia mana yang memiliki peluang lebih tinggi.
    </div><br>""", unsafe_allow_html=True)
    try:
        test_df = df.loc[X_test.index].copy()
        test_df['Predicted_Success'] = y_pred
        age_stats = test_df.groupby('AgeRecommendation')['Predicted_Success'].agg(['count','mean']).round(3)
        age_stats.columns = ['Total Games', 'Success Rate']
        age_stats['Success Rate %'] = (age_stats['Success Rate'] * 100).round(1)
        age_df = age_stats.sort_values('Success Rate', ascending=False).reset_index()
        disp = age_df[['AgeRecommendation','Total Games','Success Rate %']].copy()
        disp.columns = ['Age Recommendation','Total Games','Success Rate (%)']
        st.dataframe(disp.style.format({'Total Games':'{:.0f}','Success Rate (%)':'{:.1f}'})
                    .background_gradient(subset=['Success Rate (%)'], cmap='plasma'),
                    use_container_width=True, height=200)
    except Exception as e:
        st.error(f"Error: {e}")

    # Genres
    st.subheader("🎨 **Success Rate by Genre (Top 10)**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    Menampilkan <b>tingkat keberhasilan prediksi populer</b> berdasarkan genre game.
    Genre dengan success rate tinggi cenderung lebih kompetitif sekaligus lebih menjanjikan.
    </div><br>""", unsafe_allow_html=True)
    try:
        test_df = df.loc[X_test.index].copy()
        test_df['Predicted_Success'] = y_pred
        genre_stats = test_df.groupby('Genre')['Predicted_Success'].agg(['count','mean']).round(3)
        genre_stats.columns = ['Total Games', 'Success Rate']
        genre_stats['Success Rate %'] = (genre_stats['Success Rate'] * 100).round(1)
        genre_df = genre_stats.sort_values('Success Rate', ascending=False).head(10).reset_index()
        disp = genre_df[['Genre','Total Games','Success Rate %']].copy()
        disp.columns = ['Genre','Total Games','Success Rate (%)']
        st.dataframe(disp.style.format({'Total Games':'{:.0f}','Success Rate (%)':'{:.1f}'})
                    .background_gradient(subset=['Success Rate (%)'], cmap='viridis'),
                    use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Error: {e}")

    # Dataset Stats
    st.subheader("📈 **Dataset Statistics**")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    Statistik deskriptif seluruh fitur dalam dataset — rata-rata, standar deviasi, min, dan maks.
    Berguna untuk memahami skala dan distribusi data yang digunakan melatih model.
    </div><br>""", unsafe_allow_html=True)
    try:
        desc = df.describe(include='all').T.round(2)
        st.dataframe(desc, use_container_width=True, height=400)
    except:
        st.warning("📊 Dataset summary temporarily unavailable")

    # Feature Importance
    if df_imp is not None and len(df_imp) > 0:
        st.subheader("🏆 **Top 10 Feature Importance**")
        top_imp = df_imp.nlargest(10, 'Importance')[['Fitur', 'Importance']]
        c1, c2 = st.columns([1, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            top10 = top_imp.sort_values('Importance')
            colors = plt.cm.plasma(np.linspace(0, 1, len(top10)))
            bars = ax.barh(range(len(top10)), top10['Importance'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels(
                [str(f)[:25]+'...' if len(str(f)) > 25 else str(f) for f in top10['Fitur']],
                fontsize=9)
            ax.set_xlabel('Importance Score', fontweight='bold')
            ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10)
            ax.grid(axis='x', alpha=0.3)
            for i, bar in enumerate(bars):
                w = bar.get_width()
                ax.text(w+0.0005, i, f'{w:.4f}', va='center', fontweight='bold', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        with c2:
            st.markdown("""
            <div style='font-size:1.1rem;'>
            <b>Feature Importance</b> menunjukkan seberapa besar kontribusi setiap fitur
            dalam keputusan model Random Forest.<br><br>
            <ul>
                <li><b>like_ratio:</b> Proporsi like vs total interaksi — mencerminkan kualitas game</li>
                <li><b>update_gap_days:</b> Jarak hari sejak update terakhir — konsistensi pengembang</li>
                <li><b>engagement_rate:</b> Rasio interaksi (like+dislike) per kunjungan</li>
                <li><b>favorite_rate:</b> Proporsi pemain yang memfavoritkan game</li>
                <li><b>game_age:</b> Usia game dalam hari</li>
            </ul>
            Fitur dengan skor tinggi adalah aspek yang paling menentukan popularitas.
            </div>""", unsafe_allow_html=True)
        top_imp_disp = top_imp.copy()
        top_imp_disp['Importance'] = top_imp_disp['Importance'].round(4)
        st.dataframe(top_imp_disp.style.background_gradient(cmap='plasma'), use_container_width=True)
