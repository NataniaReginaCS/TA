import streamlit as st

st.set_page_config(page_title="🎮 Roblox Game Success Predictor", page_icon="🎮", layout="wide")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.markdown("""<style>
.main-header{font-size:clamp(2rem,5vw,3rem)!important;font-weight:800!important;text-align:center;margin:1rem 0 2rem 0!important;color:#1f77b4!important;}
.metric-card{background:linear-gradient(135deg,#ffffff,#f8f9fa);border-radius:12px;padding:18px;text-align:center;box-shadow:0px 3px 8px rgba(0,0,0,0.08);border:1px solid #e6e6e6;}
.metric-card h3{margin:0;font-size:0.9rem;color:#555;}.metric-card h2{font-size:1.8rem;font-weight:700;color:#1f77b4;}
.success-box{background:linear-gradient(135deg,#56ab2f 0%,#a8e6cf 100%);padding:2rem;border-radius:20px;text-align:center;box-shadow:0 12px 48px rgba(86,171,47,0.3);border:3px solid rgba(255,255,255,0.3);}
.error-box{background:linear-gradient(135deg,#ff6b6b 0%,#ee5a24 100%);padding:2rem;border-radius:20px;text-align:center;box-shadow:0 12px 48px rgba(255,107,107,0.3);border:3px solid rgba(255,255,255,0.3);}
.stats-metric{font-size:2.5rem!important;font-weight:900!important;color:white!important;text-shadow:0 2px 4px rgba(0,0,0,0.1);}
.stButton>button{background:linear-gradient(135deg,#1f77b4 0%,#4a90e2 100%);color:white;border-radius:12px;border:none;padding:0.75rem 2rem;font-weight:600;font-size:1.1rem;transition:all 0.3s ease;box-shadow:0 4px 20px rgba(31,119,180,0.3);}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(31,119,180,0.4);}
@media(max-width:768px){.metric-card{padding:1rem 0.75rem!important;margin:0.5rem 0;}.success-box,.error-box{padding:1.5rem 1rem!important;}}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        final_model = joblib.load("final_model.pkl")
        df          = joblib.load("processed_df.pkl")
        X_test      = joblib.load("X_test.pkl")
        y_test      = joblib.load("y_test.pkl")
        y_pred      = joblib.load("y_pred.pkl")
        y_prob      = joblib.load("y_prob.pkl")
        metrics     = joblib.load("metrics.pkl")
        df_imp = None
        try:
            df_imp = joblib.load("feature_importance_df.pkl")
        except Exception:
            pass
        return final_model, df, X_test, y_test, y_pred, y_prob, metrics, df_imp
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# ─────────────────────────────────────────────────────────────────
# BENCHMARK ENGINE
# Justifikasi pemilihan MEDIAN (bukan mean):
#   Distribusi fitur rasio (like_ratio, favorite_rate, engagement_rate,
#   update_gap_days) bersifat highly right-skewed, dibuktikan pada EDA
#   (histogram fitur di Bab 4). Mean pada distribusi skewed tertarik
#   ke outlier, sehingga tidak mewakili "game populer tipikal".
#   Median robust terhadap outlier [Han et al., 2011] dan konsisten
#   dengan metodologi penelitian yang menggunakan kuartil (Q3) sebagai
#   threshold definisi popularitas [Chen & Lu, 2023].
#
# Justifikasi threshold berbasis KUARTIL (bukan % arbitrer):
#   Threshold ditetapkan dari distribusi kuartil game populer itu sendiri
#   — pendekatan data-driven yang dapat direproduksi [Tukey, 1977]:
#   - like_ratio, favorite_rate, engagement_rate -> trigger jika < Q1
#     Makna: 75% game populer sejenis SUDAH lebih baik dari nilai user.
#   - update_gap_days -> trigger jika > Q3 (nilai besar = jarang update)
#     Makna: 75% game populer sejenis LEBIH SERING update dari user.
#   Pendekatan ini lebih defensible secara ilmiah vs angka persentase
#   arbitrer [Pargent et al., 2023].
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def build_benchmark(_df):
    pop = _df[_df['target'] == 1].copy()
    if 'like_ratio' not in pop.columns:
        pop['like_ratio']      = pop['likes'] / (pop['likes'] + pop['dislikes'] + 1)
    if 'favorite_rate' not in pop.columns:
        pop['favorite_rate']   = pop['favorites'] / pop['visits'].replace(0, 1)
    if 'engagement_rate' not in pop.columns:
        pop['engagement_rate'] = (pop['likes'] + pop['dislikes']) / pop['visits'].replace(0, 1)

    def iqr(s):
        return {'median': s.median(), 'q1': s.quantile(0.25), 'q3': s.quantile(0.75)}

    def bm(grp):
        return {'like_ratio': iqr(grp['like_ratio']), 'update_gap_days': iqr(grp['update_gap_days']),
                'favorite_rate': iqr(grp['favorite_rate']), 'engagement_rate': iqr(grp['engagement_rate'])}

    global_bm = bm(pop)
    genre_bm  = {g: {**bm(grp), 'count': len(grp)}
                 for g, grp in pop.groupby('Genre') if len(grp) >= 5}
    age_bm    = {a: {**bm(grp), 'count': len(grp)}
                 for a, grp in pop.groupby('AgeRecommendation') if len(grp) >= 5}
    return global_bm, genre_bm, age_bm

# ─────────────────────────────────────────────────────────────────
# SMART RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────
def generate_recommendations(user_vals, bm_global, bm_genre, bm_age, user_genre, user_age):
    used_genre_bm = user_genre in bm_genre
    bm     = bm_genre.get(user_genre, bm_global)
    bm_src = f"genre **{user_genre}**" if used_genre_bm else "**seluruh game populer (global)**"
    ab     = bm_age.get(user_age)
    recs   = []

    # 1. like_ratio — trigger jika < Q1 game populer sejenis
    lr_q1, lr_med = bm['like_ratio']['q1'], bm['like_ratio']['median']
    if user_vals['like_ratio'] < lr_q1:
        age_note = (f" Pada kategori **{user_age}**, median like ratio game populer "
                    f"adalah **{ab['like_ratio']['median']:.1%}**.") if ab else ""
        recs.append({'icon': '👍', 'title': 'Like Ratio di Bawah Standar Game Populer Sejenis',
            'detail': (
                f"Like ratio kamu **{user_vals['like_ratio']:.1%}** berada di bawah Q1 game populer "
                f"bergenre {bm_src} (**Q1: {lr_q1:.1%}**, median: **{lr_med:.1%}**). "
                f"**75% game populer sejenis sudah memiliki like ratio lebih tinggi dari kamu.**"
                f"{age_note} "
                "**Saran:** Minta feedback aktif dari pemain, perbaiki aspek yang paling dikritik "
                "(kontrol, tingkat kesulitan, keseimbangan), perbarui thumbnail dan deskripsi game."),
            'priority': lr_med - user_vals['like_ratio']})

    # 2. update_gap_days — trigger jika > Q3 (nilai besar = jarang update = buruk)
    ug_q3, ug_med = bm['update_gap_days']['q3'], bm['update_gap_days']['median']
    if user_vals['update_gap_days'] > ug_q3:
        age_note = (f" Pada kategori **{user_age}**, median update gap game populer "
                    f"adalah **{ab['update_gap_days']['median']:.0f} hari**.") if ab else ""
        recs.append({'icon': '🔄', 'title': 'Frekuensi Update di Bawah Standar Game Populer Sejenis',
            'detail': (
                f"Game kamu belum diperbarui selama **{user_vals['update_gap_days']:.0f} hari**, "
                f"melebihi Q3 update gap game populer bergenre {bm_src} "
                f"(**Q3: {ug_q3:.0f} hari**, median: **{ug_med:.0f} hari**). "
                f"**75% game populer sejenis melakukan update lebih sering dari kamu.**"
                f"{age_note} "
                "**Saran:** Rilis update berkala minimal 1–2 kali per bulan — bug fix, konten baru, "
                "atau event musiman sudah cukup menjaga retensi pemain."),
            'priority': user_vals['update_gap_days'] - ug_med})

    # 3. favorite_rate — trigger jika < Q1 game populer sejenis
    fr_q1, fr_med = bm['favorite_rate']['q1'], bm['favorite_rate']['median']
    if user_vals['favorite_rate'] < fr_q1:
        age_note = (f" Pada kategori **{user_age}**, median favorite rate game populer "
                    f"adalah **{ab['favorite_rate']['median']:.4f}**.") if ab else ""
        recs.append({'icon': '❤️', 'title': 'Favorite Rate di Bawah Standar Game Populer Sejenis',
            'detail': (
                f"Favorite rate kamu **{user_vals['favorite_rate']:.4f}** (favorit / kunjungan) "
                f"berada di bawah Q1 game populer bergenre {bm_src} "
                f"(**Q1: {fr_q1:.4f}**, median: **{fr_med:.4f}**). "
                f"**75% game populer sejenis memiliki favorite rate lebih tinggi.**"
                f"{age_note} "
                "**Saran:** Tambahkan insentif favorit — item eksklusif, reminder soft di akhir sesi, "
                "atau konten yang membuat pemain ingin kembali lagi."),
            'priority': (fr_med - user_vals['favorite_rate']) * 100})

    # 4. engagement_rate — trigger jika < Q1 game populer sejenis
    er_q1, er_med = bm['engagement_rate']['q1'], bm['engagement_rate']['median']
    if user_vals['engagement_rate'] < er_q1:
        age_note = (f" Pada kategori **{user_age}**, median engagement rate game populer "
                    f"adalah **{ab['engagement_rate']['median']:.4f}**.") if ab else ""
        recs.append({'icon': '💬', 'title': 'Engagement Rate di Bawah Standar Game Populer Sejenis',
            'detail': (
                f"Engagement rate kamu **{user_vals['engagement_rate']:.4f}** "
                f"((like+dislike) / kunjungan) berada di bawah Q1 game populer bergenre {bm_src} "
                f"(**Q1: {er_q1:.4f}**, median: **{er_med:.4f}**). "
                f"**75% game populer sejenis memiliki engagement rate lebih tinggi.**"
                f"{age_note} "
                "**Saran:** Tampilkan tombol rating secara menonjol, adakan tantangan komunitas, "
                "dan aktif merespons ulasan pemain di halaman game."),
            'priority': (er_med - user_vals['engagement_rate']) * 1000})

    recs.sort(key=lambda x: x['priority'], reverse=True)
    return recs, bm_src, used_genre_bm

# ─────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────
final_model, df, X_test, y_test, y_pred, y_prob, metrics, df_imp = load_artifacts()
bm_global, bm_genre, bm_age = build_benchmark(df)
unique_genres = sorted(df.get("Genre", pd.Series()).dropna().unique().tolist())
unique_ages   = sorted(df.get("AgeRecommendation", pd.Series()).dropna().unique().tolist())

# ─────────────────────────────────────────────────────────────────
# HEADER & METRICS
# ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🚀 Roblox Game Success Classification</h1>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#666;font-size:1.2rem;margin-bottom:2rem;'>Klasifikasi game populer dalam platform Roblox menggunakan <strong>Random Forest</strong></div>", unsafe_allow_html=True)
st.markdown("### 📊 **Model Performance Overview**")
cols = st.columns(7)
for col, (label, val) in zip(cols, [
    ("Accuracy", f"{metrics['accuracy']:.1%}"),
    ("F1 Score", f"{metrics['f1_score']:.3f}"),
    ("Precision", f"{metrics['precision']:.3f}"),
    ("Recall", f"{metrics['recall']:.3f}"),
    ("F1 macro", f"{metrics.get('f1_score_macro', metrics['f1_score']):.3f}"),
    ("Prec macro", f"{metrics.get('precision_macro', metrics['precision']):.3f}"),
    ("Rec macro", f"{metrics.get('recall_macro', metrics['recall']):.3f}"),
]):
    with col:
        st.markdown(f'<div class="metric-card"><h3>{label}</h3><h2>{val}</h2></div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎮 Game Predictor", "📊 Model Analytics", "📈 Data Explorer"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("---")
    with st.expander("ℹ️ Catatan Penting tentang Model Ini", expanded=False):
        st.markdown("""
**Bagaimana model ini bekerja?**
Model Random Forest dilatih dari 9.734 data game Roblox (Kaggle). **"Populer"** = game di **persentil ke-75 ke atas** berdasarkan jumlah pemain aktif — hanya ~25% game teratas.

**Mengapa game saya bisa tidak populer meskipun semua fitur sudah baik?**
- Model mempertimbangkan **kombinasi** semua fitur sekaligus, bukan satu per satu.
- **Ambang batas sangat ketat** — hanya top 25% yang dianggap populer.
- **Game terlalu baru** — pola interaksi belum terbentuk secara representatif.
- **Dataset statis** — tidak menangkap dinamika jangka panjang atau tren musiman.
- Model ini adalah **alat bantu analitis**, bukan jaminan kesuksesan.

**Bagaimana panduan peningkatan ditentukan?**
Panduan muncul hanya jika nilai kamu berada **di bawah Q1 (kuartil pertama) distribusi game populer sejenis (genre yang sama)**. Artinya 75% game populer bergenre sama sudah lebih baik dari kamu pada aspek tersebut.
        """)

    st.markdown("#### 🔮 **Input Game Data**")
    with st.form("predict_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            genre   = st.selectbox("🎨 **Genre**", options=unique_genres)
            age_rec = st.selectbox("👶 **Age Rating**", options=unique_ages)
        with c2:
            game_age   = st.number_input("📅 **Game Age (days)**", min_value=0, value=300, help="Hari sejak game pertama dibuat.")
            update_gap = st.number_input("🔄 **Days Since Last Update**", min_value=0, value=30, help="0 = baru saja diperbarui. Semakin kecil = semakin baik.")
        c3, c4 = st.columns(2)
        with c3:
            visits    = st.number_input("👥 **Total Visits**", min_value=0, value=50000)
            favorites = st.number_input("❤️ **Favorites**",    min_value=0, value=2500)
        with c4:
            likes    = st.number_input("👍 **Likes**",    min_value=0, value=5000)
            dislikes = st.number_input("👎 **Dislikes**", min_value=0, value=500)
        submitted = st.form_submit_button("🚀 **ANALYZE GAME**", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing game potential..."):
            fav_rate   = favorites / max(visits, 1)
            eng_rate   = (likes + dislikes) / max(visits, 1)
            like_ratio = likes / max(likes + dislikes, 1)
            input_df   = pd.DataFrame({
                "game_age": [game_age], "update_gap_days": [float(update_gap)],
                "favorite_rate": [fav_rate], "engagement_rate": [eng_rate],
                "like_ratio": [like_ratio], "Genre": [genre], "AgeRecommendation": [age_rec]})
            pred = final_model.predict(input_df)[0]
            prob = final_model.predict_proba(input_df)[:, 1][0]

        if pred == 1:
            st.markdown(f'<div class="success-box"><h2>🎉 HIGH SUCCESS POTENTIAL!</h2><h3 style="color:#2d5a2a;font-size:2rem;"><span class="stats-metric">Probability of success: {prob*100:.1f}%</span></h3><p>🚀 Game kamu diprediksi populer!</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box"><h2>⚠️ Needs Optimization</h2><h3 style="color:#8b1a1a;font-size:2rem;"><span class="stats-metric">Probability of success: {prob*100:.1f}%</span></h3><p>💡 Lihat panduan di bawah untuk memahami aspek yang bisa ditingkatkan.</p></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        user_vals = {'like_ratio': like_ratio, 'update_gap_days': float(update_gap),
                     'favorite_rate': fav_rate, 'engagement_rate': eng_rate}
        recs, bm_src, used_genre_bm = generate_recommendations(
            user_vals, bm_global, bm_genre, bm_age, genre, age_rec)

        genre_count = bm_genre.get(genre, {}).get('count', 0)
        cb = "#ff7f0e" if pred == 0 else "#1f77b4"
        bg = "#fff8f0" if pred == 0 else "#f0f8ff"
        st.markdown("### 🛠️ **Panduan Peningkatan**" if pred == 0 else "### ✨ **Aspek yang Masih Bisa Dioptimalkan**")

        caption = (f"Panduan dibandingkan dengan **{genre_count} game populer bergenre {genre}** dalam dataset. "
                   f"Informasi usia (**{age_rec}**) ditampilkan sebagai konteks tambahan. "
                   f"Rekomendasi muncul hanya jika nilai di bawah Q1 game populer sejenis."
                   if used_genre_bm else
                   f"Genre **{genre}** memiliki terlalu sedikit sampel game populer. "
                   f"Panduan dibandingkan dengan seluruh game populer dalam dataset (global benchmark).")
        st.caption(caption)

        if recs:
            for r in recs:
                st.markdown(f'<div style="background:{bg};border-left:5px solid {cb};padding:14px 18px;border-radius:10px;margin-bottom:12px;"><b style="font-size:1.05rem;">{r["icon"]} {r["title"]}</b><br><br><span style="color:#444;font-size:0.97rem;">{r["detail"]}</span></div>', unsafe_allow_html=True)
        else:
            if pred == 0:
                st.info(
                    f"**Semua fitur utama sudah berada di atas Q1 game populer bergenre {genre}.**\n\n"
                    "Secara metrik individual performa kamu sudah kompetitif, namun model tetap "
                    "memprediksi tidak populer. Kemungkinan penyebabnya:\n\n"
                    "- **Kombinasi fitur:** Model RF mempertimbangkan interaksi antar semua fitur sekaligus.\n"
                    "- **Game terlalu baru:** Pola interaksi belum terbentuk secara representatif.\n"
                    "- **Kompetisi genre tinggi:** Simulation dan Roleplay sangat kompetitif di Roblox.\n"
                    "- **Keterbatasan model:** Dataset statis tidak menangkap dinamika jangka panjang.")
            else:
                st.success("Semua fitur utama sudah optimal! Pertahankan kualitas update dan keterlibatan komunitas.")

        with st.expander(f"📐 Detail Benchmark — Genre: {genre} | Age: {age_rec}"):
            bm_show = bm_genre.get(genre, bm_global)
            src_lbl = f"genre {genre} ({genre_count} game populer)" if used_genre_bm else "global (semua game populer)"
            st.markdown(f"**Sumber benchmark:** {src_lbl}")
            st.info("**Dasar threshold (data-driven, bukan arbitrer):**\n"
                    "- `like_ratio`, `favorite_rate`, `engagement_rate` → rekomendasi jika di bawah **Q1** (75% game populer sejenis sudah lebih baik)\n"
                    "- `update_gap_days` → rekomendasi jika di atas **Q3** (75% game populer sejenis lebih sering update)")
            rows = []
            for feat in ['like_ratio', 'update_gap_days', 'favorite_rate', 'engagement_rate']:
                user_v = user_vals[feat]
                bv     = bm_show[feat]
                is_ug  = feat == 'update_gap_days'
                fmt    = (lambda x: f"{x:.0f} hari") if is_ug else (lambda x: f"{x:.4f}")
                threshold = f"Q3 = {bv['q3']:.0f} hari" if is_ug else f"Q1 = {bv['q1']:.4f}"
                ok = user_v <= bv['q3'] if is_ug else user_v >= bv['q1']
                rows.append({'Fitur': feat, 'Nilai Kamu': fmt(user_v), 'Q1': fmt(bv['q1']),
                             'Median': fmt(bv['median']), 'Q3': fmt(bv['q3']),
                             'Threshold': threshold, 'Status': "✅ Sudah baik" if ok else "⚠️ Perlu ditingkatkan"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if age_rec in bm_age:
                ab = bm_age[age_rec]
                st.markdown(f"**Konteks usia — {age_rec}** ({ab['count']} game populer)")
                age_rows = []
                for feat in ['like_ratio', 'update_gap_days', 'favorite_rate', 'engagement_rate']:
                    fmt = (lambda x: f"{x:.0f} hari") if feat == 'update_gap_days' else (lambda x: f"{x:.4f}")
                    age_rows.append({'Fitur': feat, 'Q1': fmt(ab[feat]['q1']),
                                     'Median': fmt(ab[feat]['median']), 'Q3': fmt(ab[feat]['q3'])})
                st.dataframe(pd.DataFrame(age_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("---")
    st.subheader("📊 Target Distribution")
    ci1, ci2 = st.columns([1,2])
    with ci1:
        fig, ax = plt.subplots(figsize=(5,3))
        tc = pd.Series(y_test).value_counts().sort_index()
        bars = ax.bar(tc.index, tc.values, color=['#ff9999','#66b3ff'], alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Not Popular','Popular'], fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold'); ax.set_title('Game Popularity Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2., h+20, f'{int(h)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.tight_layout(); st.pyplot(fig)
    with ci2:
        st.markdown("""<div style='font-size:1.1rem;'><b>Distribusi Target</b> pada 2.018 sampel data pengujian.<br><br><ul>
        <li><b>Popular (1):</b> Game di atas persentil ke-75 jumlah pemain aktif</li>
        <li><b>Not Popular (0):</b> Game di bawah persentil ke-75</li></ul>
        Ketidakseimbangan kelas (~75:25) menjadi alasan utama penggunaan <b>F1-score macro</b> sebagai metrik evaluasi utama.
        </div>""", unsafe_allow_html=True)

    st.subheader("🔍 Confusion Matrix")
    ci1, ci2 = st.columns([1,2])
    with ci1:
        fig, ax = plt.subplots(figsize=(7,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Popular','Popular'], yticklabels=['Actual: Not Popular','Actual: Popular'],
                    ax=ax, cbar_kws={'label':'Count'})
        ax.set_title('Confusion Matrix', fontweight='bold', pad=20); st.pyplot(fig)
    with ci2:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""<div style='font-size:1.1rem;'><b>Confusion Matrix</b> — 2.018 sampel data uji:<br><br><ul>
        <li>✅ <b>True Positive ({tp:,}):</b> Game populer yang berhasil dideteksi</li>
        <li>✅ <b>True Negative ({tn:,}):</b> Game tidak populer yang benar diklasifikasikan</li>
        <li>⚠️ <b>False Positive ({fp:,}):</b> Game tidak populer salah diprediksi populer</li>
        <li>❌ <b>False Negative ({fn:,}):</b> Game populer yang tidak terdeteksi</li></ul>
        Parameter <code>class_weight='balanced'</code> menekan False Negative agar game populer tidak terlewatkan.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Total Samples", f"{len(df):,}"); c2.metric("🔧 Features", df.shape[1]); c3.metric("🗂️ Dataset", "Kaggle Roblox Dataset")
    st.markdown("---")

    st.subheader("👶 Success Rate by Age Recommendation")
    try:
        tdf = df.loc[X_test.index].copy(); tdf['Predicted_Success'] = y_pred
        ag = tdf.groupby('AgeRecommendation')['Predicted_Success'].agg(['count','mean'])
        ag.columns = ['Total Games','Success Rate']; ag['Success Rate %'] = (ag['Success Rate']*100).round(1)
        ag = ag.sort_values('Success Rate', ascending=False).reset_index()
        disp = ag[['AgeRecommendation','Total Games','Success Rate %']].copy()
        disp.columns = ['Age Recommendation','Total Games','Success Rate (%)']
        st.dataframe(disp.style.format({'Total Games':'{:.0f}','Success Rate (%)':'{:.1f}'}).background_gradient(subset=['Success Rate (%)'], cmap='plasma'), use_container_width=True, height=200)
    except Exception as e: st.error(f"Error: {e}")

    st.subheader("🎨 Success Rate by Genre (Top 10)")
    try:
        tdf = df.loc[X_test.index].copy(); tdf['Predicted_Success'] = y_pred
        gn = tdf.groupby('Genre')['Predicted_Success'].agg(['count','mean'])
        gn.columns = ['Total Games','Success Rate']; gn['Success Rate %'] = (gn['Success Rate']*100).round(1)
        gn = gn.sort_values('Success Rate', ascending=False).head(10).reset_index()
        disp = gn[['Genre','Total Games','Success Rate %']].copy(); disp.columns = ['Genre','Total Games','Success Rate (%)']
        st.dataframe(disp.style.format({'Total Games':'{:.0f}','Success Rate (%)':'{:.1f}'}).background_gradient(subset=['Success Rate (%)'], cmap='viridis'), use_container_width=True, height=300)
    except Exception as e: st.error(f"Error: {e}")

    st.subheader("📈 Dataset Statistics")
    try:
        st.dataframe(df.describe(include='all').T.round(2), use_container_width=True, height=400)
    except Exception: st.warning("Dataset summary temporarily unavailable")

    if df_imp is not None and len(df_imp) > 0:
        st.subheader("🏆 Top 10 Feature Importance")
        top_imp = df_imp.nlargest(10, 'Importance')[['Fitur','Importance']]
        ci1, ci2 = st.columns([1,2])
        with ci1:
            fig, ax = plt.subplots(figsize=(7,5))
            t10 = top_imp.sort_values('Importance')
            bars = ax.barh(range(len(t10)), t10['Importance'], color=plt.cm.plasma(np.linspace(0,1,len(t10))), alpha=0.8)
            ax.set_yticks(range(len(t10))); ax.set_yticklabels([str(f)[:25]+'...' if len(str(f))>25 else str(f) for f in t10['Fitur']], fontsize=9)
            ax.set_xlabel('Importance Score', fontweight='bold'); ax.set_title('Feature Importance', fontsize=12, fontweight='bold', pad=10); ax.grid(axis='x', alpha=0.3)
            for i,bar in enumerate(bars):
                w = bar.get_width(); ax.text(w+0.0005, i, f'{w:.4f}', va='center', fontweight='bold', fontsize=9)
            plt.tight_layout(); st.pyplot(fig)
        with ci2:
            st.markdown("""<div style='font-size:1.1rem;'><b>Feature Importance</b> — kontribusi tiap fitur dalam keputusan model RF:<br><br><ul>
            <li><b>like_ratio</b> (0.230): Proporsi like dari total interaksi</li>
            <li><b>update_gap_days</b> (0.165): Hari sejak update terakhir</li>
            <li><b>engagement_rate</b> (0.141): Rasio interaksi per kunjungan</li>
            <li><b>favorite_rate</b> (0.137): Proporsi pemain yang memfavoritkan game</li>
            <li><b>game_age</b> (0.110): Usia game dalam hari</li></ul>
            Dominasi fitur numerik hasil rekayasa memvalidasi tahap <i>feature engineering</i>.</div>""", unsafe_allow_html=True)
        ti = top_imp.copy(); ti['Importance'] = ti['Importance'].round(4)
        st.dataframe(ti.style.background_gradient(cmap='plasma'), use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#aaa;font-size:0.85rem;'>Roblox Game Success Classifier · Random Forest · Dataset: Kaggle Roblox (9.734 games)</div>", unsafe_allow_html=True)