import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 16px 20px;
    backdrop-filter: blur(8px);
}
div[data-testid="metric-container"] label {
    color: #a89fd8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0edff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, #7c3aed 0%, #4f46e5 50%, #0ea5e9 100%);
    border-radius: 20px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 40px rgba(124,58,237,0.35);
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
}
.hero-banner p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.82);
    margin: 0;
    font-weight: 400;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 700;
    color: #c4b5fd;
    border-left: 4px solid #7c3aed;
    padding-left: 12px;
    margin: 1.6rem 0 1rem 0;
    letter-spacing: -0.01em;
}

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(79,70,229,0.2));
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    margin: 1rem 0;
}
.pred-box .label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a89fd8;
    font-weight: 600;
    margin-bottom: 8px;
}
.pred-box .price {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f0edff;
    font-family: 'IBM Plex Mono', monospace;
}
.pred-box .sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    margin-top: 6px;
}

/* Info box */
.info-box {
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #bae6fd;
    margin: 0.5rem 0;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #a89fd8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0edff !important;
    border-bottom-color: #7c3aed !important;
}

/* Plotly/Matplotlib figures */
.stPlotlyChart, .stImage {
    border-radius: 14px;
    overflow: hidden;
}

/* Slider labels */
.stSlider label {
    color: #c4b5fd !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.55) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Select boxes */
.stSelectbox label { color: #c4b5fd !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#4a4568",
    "axes.labelcolor": "#c4b5fd",
    "xtick.color": "#9c8fc8",
    "ytick.color": "#9c8fc8",
    "text.color": "#e8e6f0",
    "grid.color": "#2e2a4a",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.facecolor": "#1a1730",
    "legend.edgecolor": "#3d3760",
    "figure.autolayout": True,
})

ACCENT = "#7c3aed"
COLORS = ["#7c3aed", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]

# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("RealEstate_HousePrice_Dataset_4200.csv")
    return df

# ─── Model training ────────────────────────────────────────────────────────────
FEATURES = ['area_sqft', 'bedrooms', 'bathrooms', 'location_score',
            'age_years', 'distance_city_km', 'lot_size_sqft',
            'has_garage', 'has_pool', 'renovation_years_ago']
TARGET = 'house_price_inr'

@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]

    # ── Simple LR ──
    X_s = df[['area_sqft']]
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_s, y, test_size=0.2, random_state=42)
    slr = LinearRegression().fit(Xs_tr, ys_tr)
    ys_pred = slr.predict(Xs_te)

    # ── Multiple LR ──
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X, y, test_size=0.2, random_state=18)
    mlr = LinearRegression().fit(Xm_tr, ym_tr)
    ym_pred = mlr.predict(Xm_te)

    # ── Polynomial ──
    poly2 = PolynomialFeatures(degree=2)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_s, y, test_size=0.2, random_state=42)
    Xp_tr2 = poly2.fit_transform(Xp_tr)
    Xp_te2 = poly2.transform(Xp_te)
    plr = LinearRegression().fit(Xp_tr2, yp_tr)
    yp_pred = plr.predict(Xp_te2)

    def metrics(yt, yp, n, p):
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        r2 = r2_score(yt, yp)
        adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, AdjR2=adj)

    n = len(ys_te)
    results = {
        "Simple LR": {
            "model": slr, "metrics": metrics(ys_te, ys_pred, n, 1),
            "y_test": ys_te, "y_pred": ys_pred, "X_test": Xs_te
        },
        "Multiple LR": {
            "model": mlr, "metrics": metrics(ym_te, ym_pred, len(ym_te), len(FEATURES)),
            "y_test": ym_te, "y_pred": ym_pred
        },
        "Polynomial (d=2)": {
            "model": plr, "poly": poly2, "metrics": metrics(yp_te, yp_pred, len(yp_te), 2),
            "y_test": yp_te, "y_pred": yp_pred
        },
    }
    return results, mlr, Xm_te, ym_te

@st.cache_resource
def train_gradient_descent(df):
    X = df.drop(['house_id', TARGET], axis=1)
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    ytr_v, yte_v = ytr.values, yte.values

    def bgd(X, y, lr=0.01, epochs=300):
        m, n = X.shape; w, b, hist = np.zeros(n), 0, []
        for _ in range(epochs):
            yp = X @ w + b
            w -= lr/m * X.T @ (yp - y)
            b -= lr/m * np.sum(yp - y)
            hist.append(mean_squared_error(y, X @ w + b))
        return w, b, hist

    def sgd(X, y, lr=0.001, epochs=50):
        m, n = X.shape; w, b, hist = np.zeros(n), 0, []
        for _ in range(epochs):
            for i in np.random.permutation(m):
                xi, yi = X[i], y[i]
                yp = xi @ w + b
                w -= lr * xi * (yp - yi); b -= lr * (yp - yi)
            hist.append(mean_squared_error(y, X @ w + b))
        return w, b, hist

    def mbgd(X, y, lr=0.01, epochs=200, bs=64):
        m, n = X.shape; w, b, hist = np.zeros(n), 0, []
        for _ in range(epochs):
            idx = np.random.permutation(m)
            for i in range(0, m, bs):
                Xb, yb = X[idx[i:i+bs]], y[idx[i:i+bs]]
                yp = Xb @ w + b
                w -= lr/len(Xb) * Xb.T @ (yp - yb)
                b -= lr/len(Xb) * np.sum(yp - yb)
            hist.append(mean_squared_error(y, X @ w + b))
        return w, b, hist

    t0 = time.time(); bw, bb, bh = bgd(Xtr_s, ytr_v); bt = time.time() - t0
    t0 = time.time(); sw, sb, sh = sgd(Xtr_s, ytr_v); st2 = time.time() - t0
    t0 = time.time(); mw, mb, mh = mbgd(Xtr_s, ytr_v); mt = time.time() - t0

    gd = {
        "BGD":  {"pred": Xte_s @ bw + bb, "hist": bh, "time": bt},
        "SGD":  {"pred": Xte_s @ sw + sb, "hist": sh, "time": st2},
        "MBGD": {"pred": Xte_s @ mw + mb, "hist": mh, "time": mt},
    }
    for k in gd:
        gd[k]["r2"]   = r2_score(yte_v, gd[k]["pred"])
        gd[k]["rmse"]  = np.sqrt(mean_squared_error(yte_v, gd[k]["pred"]))
    return gd

# ─── Helpers ───────────────────────────────────────────────────────────────────
def fmt_inr(val):
    val = int(val)
    if val >= 1_00_00_000: return f"₹{val/1_00_00_000:.2f} Cr"
    if val >= 1_00_000:    return f"₹{val/1_00_000:.2f} L"
    return f"₹{val:,}"

def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    return fig, ax

def show_metric_chart(label, values, names):
    fig, ax = dark_fig((7, 3.5))
    bars = ax.bar(names, values, color=COLORS[:len(names)], width=0.5, zorder=3)
    ax.bar_label(bars, fmt=lambda x: f"{x:.4f}" if x < 10 else f"{x:,.0f}",
                 padding=5, color="#c4b5fd", fontsize=9, fontweight="bold")
    ax.set_title(label, fontsize=11, fontweight="bold", color="#e8e6f0", pad=10)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Could not find `RealEstate_HousePrice_Dataset_4200.csv`. Place it in the same folder as this script.")
    st.stop()

model_results, mlr_model, Xm_te, ym_te = train_models(df)
gd_results = train_gradient_descent(df)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🏠 House Price Predictor</h1>
  <p>Regression analysis &amp; ML model comparison on 4,200 real estate records — INR pricing.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio("", [
        "📊 Dataset Overview",
        "🔮 Predict Price",
        "📈 Model Comparison",
        "⬇️ Gradient Descent",
        "⚖️ Bias–Variance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset stats**")
    st.markdown(f"- Rows: `{len(df):,}`")
    st.markdown(f"- Features: `{len(FEATURES)}`")
    st.markdown(f"- Avg price: `{fmt_inr(df[TARGET].mean())}`")
    st.markdown(f"- Max price: `{fmt_inr(df[TARGET].max())}`")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.markdown('<div class="section-header">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(10).style.format({TARGET: lambda x: fmt_inr(x), "area_sqft": "{:,.0f}"}),
        use_container_width=True, height=280
    )

    st.markdown('<div class="section-header">📐 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-header">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = FEATURES + [TARGET]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu",
                linewidths=0.5, linecolor="#1a1730",
                annot_kws={"size": 8}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold", color="#f0edff", pad=12)
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🏷️ Price Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 3.5))
        ax.hist(df[TARGET]/1e7, bins=40, color=ACCENT, alpha=0.8, edgecolor="#f0edff", linewidth=0.3)
        ax.set_xlabel("Price (₹ Crore)"); ax.set_ylabel("Count")
        ax.set_title("House Price Distribution", fontsize=11, fontweight="bold", color="#e8e6f0")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">📐 Area vs Price</div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 3.5))
        sc = ax.scatter(df['area_sqft'], df[TARGET]/1e7, c=df['location_score'],
                        cmap='plasma', alpha=0.45, s=18, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='Location Score', shrink=0.8)
        ax.set_xlabel("Area (sqft)"); ax.set_ylabel("Price (₹ Crore)")
        ax.set_title("Area vs Price (colored by Location Score)", fontsize=10, fontweight="bold", color="#e8e6f0")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-header">🛏️ Bedrooms & Bathrooms Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, title, c in zip(['bedrooms', 'bathrooms'],
                              ['Bedrooms', 'Bathrooms'], [c1, c2]):
        with c:
            fig, ax = dark_fig((5, 3))
            vc = df[col].value_counts().sort_index()
            ax.bar(vc.index, vc.values, color=COLORS[1], width=0.6)
            ax.set_xlabel(title); ax.set_ylabel("Count")
            ax.set_title(f"{title} Count", fontsize=10, fontweight="bold")
            ax.yaxis.grid(True); ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – Predict Price
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown('<div class="section-header">🔮 Enter House Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ Adjust the sliders to match the property you want to estimate. The <b>Multiple Linear Regression</b> model will predict the price instantly.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        area      = st.slider("Area (sqft)",             500, 5000, 2000, 50)
        bedrooms  = st.slider("Bedrooms",                1, 6, 3)
        bathrooms = st.slider("Bathrooms",               1, 5, 2)
        loc_score = st.slider("Location Score (1–10)",   1.0, 10.0, 7.0, 0.1)
    with c2:
        age_years  = st.slider("Age of House (years)",   0, 50, 10)
        dist_city  = st.slider("Distance to City (km)",  1.0, 50.0, 15.0, 0.5)
        lot_size   = st.slider("Lot Size (sqft)",        800, 8000, 3000, 100)
    with c3:
        has_garage = st.selectbox("Has Garage?",  [1, 0], format_func=lambda x: "Yes" if x else "No")
        has_pool   = st.selectbox("Has Pool?",    [0, 1], format_func=lambda x: "Yes" if x else "No")
        reno_years = st.slider("Renovation (years ago)", 0, 30, 5)

    inp = np.array([[area, bedrooms, bathrooms, loc_score,
                     age_years, dist_city, lot_size,
                     has_garage, has_pool, reno_years]])
    pred = mlr_model.predict(inp)[0]

    st.markdown(f"""
    <div class="pred-box">
        <div class="label">Estimated House Price</div>
        <div class="price">{fmt_inr(pred)}</div>
        <div class="sub">Predicted by Multiple Linear Regression · INR</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance bar chart
    st.markdown('<div class="section-header">📊 Feature Coefficients (Model Weights)</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": mlr_model.coef_}).sort_values("Coefficient")
    fig, ax = dark_fig((8, 4))
    colors_bar = [COLORS[4] if v < 0 else COLORS[0] for v in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_bar, height=0.6)
    ax.axvline(0, color="#c4b5fd", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Coefficient Value"); ax.set_title("Feature Coefficients", fontsize=11, fontweight="bold")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="section-header">📊 Metrics Summary</div>', unsafe_allow_html=True)

    rows = []
    for name, res in model_results.items():
        m = res["metrics"]
        rows.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R2"], "Adj R²": m["AdjR2"]})
    cmp_df = pd.DataFrame(rows).set_index("Model")

    def highlight_best(s):
        if s.name in ("R²", "Adj R²"):
            best = s.max()
            return ["background-color: rgba(124,58,237,0.3); font-weight:bold" if v == best else "" for v in s]
        else:
            best = s.min()
            return ["background-color: rgba(16,185,129,0.2); font-weight:bold" if v == best else "" for v in s]

    st.dataframe(
        cmp_df.style
            .format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.4f}", "Adj R²": "{:.4f}"})
            .apply(highlight_best),
        use_container_width=True
    )

    # R² comparison
    c1, c2 = st.columns(2)
    with c1:
        show_metric_chart("R² Score (higher = better)",
                          [r["metrics"]["R2"] for r in model_results.values()],
                          list(model_results.keys()))
    with c2:
        show_metric_chart("RMSE (lower = better)",
                          [r["metrics"]["RMSE"] for r in model_results.values()],
                          list(model_results.keys()))

    # Actual vs Predicted plots
    st.markdown('<div class="section-header">📉 Actual vs Predicted</div>', unsafe_allow_html=True)
    cols = st.columns(len(model_results))
    for col, (name, res) in zip(cols, model_results.items()):
        with col:
            fig, ax = dark_fig((4.5, 4.5))
            yt = res["y_test"].values / 1e7
            yp = res["y_pred"] / 1e7
            ax.scatter(yt, yp, alpha=0.4, s=12, color=COLORS[list(model_results).index(name)])
            mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([mn, mx], [mn, mx], color="#f59e0b", linewidth=1.4, linestyle="--")
            ax.set_xlabel("Actual (₹ Cr)"); ax.set_ylabel("Predicted (₹ Cr)")
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig, use_container_width=True)

    # Residual plots
    st.markdown('<div class="section-header">🔵 Residual Analysis</div>', unsafe_allow_html=True)
    cols2 = st.columns(len(model_results))
    for col, (name, res) in zip(cols2, model_results.items()):
        with col:
            residuals = res["y_test"].values - res["y_pred"]
            fig, ax = dark_fig((4.5, 3.5))
            ax.scatter(res["y_pred"] / 1e7, residuals / 1e5, alpha=0.4, s=12, color=COLORS[list(model_results).index(name)])
            ax.axhline(0, color="#f59e0b", linewidth=1.4, linestyle="--")
            ax.set_xlabel("Predicted (₹ Cr)"); ax.set_ylabel("Residual (₹ L)")
            ax.set_title(f"Residuals – {name}", fontsize=9, fontweight="bold")
            ax.spines[['top','right']].set_visible(False)
            st.pyplot(fig, use_container_width=True)

    # Polynomial degree comparison
    st.markdown('<div class="section-header">📐 Polynomial Regression – Area vs Price Fit</div>', unsafe_allow_html=True)
    X_area = df[['area_sqft']]
    y_price = df[TARGET]
    X_range = np.linspace(X_area.min().values[0], X_area.max().values[0], 200).reshape(-1, 1)
    fig, ax = dark_fig((9, 4.5))
    ax.scatter(X_area, y_price / 1e7, alpha=0.2, s=10, color="#9c8fc8", label="Data", zorder=2)
    lin = LinearRegression().fit(X_area, y_price)
    ax.plot(X_range, lin.predict(X_range) / 1e7, color=COLORS[0], lw=2, label="Linear", zorder=4)
    for d, c in zip([2, 3], [COLORS[1], COLORS[2]]):
        pf = PolynomialFeatures(degree=d)
        Xp = pf.fit_transform(X_area)
        m = LinearRegression().fit(Xp, y_price)
        ax.plot(X_range, m.predict(pf.transform(X_range)) / 1e7, color=c, lw=2, label=f"Poly d={d}", zorder=5)
    ax.set_xlabel("Area (sqft)"); ax.set_ylabel("Price (₹ Crore)")
    ax.set_title("Regression Curve Comparison", fontsize=11, fontweight="bold")
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – Gradient Descent
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⬇️ Gradient Descent":
    st.markdown('<div class="section-header">📊 Gradient Descent Comparison</div>', unsafe_allow_html=True)

    rows = []
    labels = {"BGD": "Batch GD", "SGD": "Stochastic GD", "MBGD": "Mini-Batch GD"}
    for k, res in gd_results.items():
        rows.append({"Method": labels[k], "R² Score": res["r2"],
                     "RMSE": res["rmse"], "Training Time (s)": res["time"]})
    gd_df = pd.DataFrame(rows).set_index("Method")
    st.dataframe(
        gd_df.style.format({"R² Score": "{:.4f}", "RMSE": "{:,.0f}", "Training Time (s)": "{:.3f}"}),
        use_container_width=True
    )

    # Convergence plot
    st.markdown('<div class="section-header">📉 Convergence Curves</div>', unsafe_allow_html=True)
    fig, ax = dark_fig((10, 4.5))
    for (k, res), color in zip(gd_results.items(), COLORS):
        ax.plot(res["hist"], color=color, lw=2, label=labels[k])
    ax.set_xlabel("Epochs"); ax.set_ylabel("MSE Cost")
    ax.set_title("Gradient Descent Convergence", fontsize=12, fontweight="bold")
    ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    # Bar charts
    c1, c2, c3 = st.columns(3)
    names = [labels[k] for k in gd_results]
    with c1:
        show_metric_chart("R² Score", [gd_results[k]["r2"] for k in gd_results], names)
    with c2:
        show_metric_chart("RMSE", [gd_results[k]["rmse"] for k in gd_results], names)
    with c3:
        show_metric_chart("Training Time (s)", [gd_results[k]["time"] for k in gd_results], names)

    st.markdown('<div class="info-box">💡 <b>Mini-Batch GD</b> offers the best balance — faster than Batch GD and more stable than SGD. Ideal for real-world datasets.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – Bias-Variance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Bias–Variance":
    st.markdown('<div class="section-header">⚖️ Train vs Test Performance</div>', unsafe_allow_html=True)

    X_s  = df[['area_sqft']]; X_m = df[FEATURES]; y = df[TARGET]
    pf2  = PolynomialFeatures(degree=2)
    Xp2  = pf2.fit_transform(X_s)

    splits = [(X_s, y, 1, "Simple LR"), (X_m, y, len(FEATURES), "Multiple LR"), (Xp2, y, 2, "Polynomial d=2")]
    bv_rows = []
    for Xd, yd, p, label in splits:
        Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=42)
        m = LinearRegression().fit(Xtr, ytr)
        train_r2 = r2_score(ytr, m.predict(Xtr))
        test_r2  = r2_score(yte, m.predict(Xte))
        train_rmse = np.sqrt(mean_squared_error(ytr, m.predict(Xtr)))
        test_rmse  = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
        bv_rows.append({"Model": label, "Train R²": train_r2, "Test R²": test_r2,
                        "Train RMSE": train_rmse, "Test RMSE": test_rmse})

    bv_df = pd.DataFrame(bv_rows).set_index("Model")
    st.dataframe(
        bv_df.style.format({"Train R²": "{:.4f}", "Test R²": "{:.4f}",
                             "Train RMSE": "{:,.0f}", "Test RMSE": "{:,.0f}"}),
        use_container_width=True
    )

    # R² gap chart
    st.markdown('<div class="section-header">📈 Bias–Variance Tradeoff Curve</div>', unsafe_allow_html=True)
    fig, ax = dark_fig((9, 4.5))
    models = bv_df.index.tolist()
    ax.plot(models, bv_df["Train R²"], marker='o', color=COLORS[0], lw=2.5, ms=9, label="Train R²")
    ax.plot(models, bv_df["Test R²"],  marker='s', color=COLORS[1], lw=2.5, ms=9, label="Test R²")
    for i, (tr, te) in enumerate(zip(bv_df["Train R²"], bv_df["Test R²"])):
        ax.fill_between([i-0.15, i+0.15], [tr, tr], [te, te], alpha=0.15, color=COLORS[4])
    ax.set_xlabel("Model Complexity ➜"); ax.set_ylabel("R² Score")
    ax.set_title("Train vs Test R² (Bias–Variance Tradeoff)", fontsize=12, fontweight="bold")
    ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    # RMSE gap
    st.markdown('<div class="section-header">📉 RMSE Comparison</div>', unsafe_allow_html=True)
    fig, ax = dark_fig((9, 4))
    x_pos = np.arange(len(models))
    w = 0.35
    ax.bar(x_pos - w/2, bv_df["Train RMSE"] / 1e5, width=w, color=COLORS[0], label="Train RMSE", alpha=0.85)
    ax.bar(x_pos + w/2, bv_df["Test RMSE"]  / 1e5, width=w, color=COLORS[1], label="Test RMSE",  alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(models)
    ax.set_ylabel("RMSE (₹ Lakh)"); ax.set_title("Train vs Test RMSE", fontsize=11, fontweight="bold")
    ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    best = bv_df["Test R²"].idxmax()
    st.markdown(f'<div class="info-box">🏆 Best model by Test R²: <b>{best}</b> — R² = <b>{bv_df.loc[best, "Test R²"]:.4f}</b></div>', unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; padding:12px 0;'>"
    "🏠 House Price Predictor · Built with Streamlit · Dataset: 4,200 real estate records"
    "</div>",
    unsafe_allow_html=True
)
