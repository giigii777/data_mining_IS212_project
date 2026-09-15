# ============================================================
# Airline Passenger Satisfaction Analysis — IS-212 Data Mining Project
# IS-212 Data Mining · University of Computer Studies, Yangon
# Built from data_mining_airline_updated.ipynb
# ============================================================
import time
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathlib import Path

# ---------------- constants ----------------
DATA_PATH = Path(__file__).parent / "data" / "airline.csv"
TARGET = "satisfaction"
SAT_POS = "satisfied"
SAT_NEG = "neutral or dissatisfied"

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes",
            "Arrival Delay in Minutes"]
SKEW_COLS = ["Age", "Flight Distance",
             "Departure Delay in Minutes", "Arrival Delay in Minutes"]
REQUIRED = CAT_COLS + NUM_COLS + RATING_COLS + [TARGET]

CYAN = "#00d4ff"
PURPLE = "#7c4dff"
GREEN = "#43a047"
RED = "#ef5350"
AMBER = "#ffb300"
GRID = "rgba(148,163,184,0.08)"
AXIS = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(148,163,184,0.2)")
FAM = "Inter, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', 'Fira Code', monospace"

MODE = "wide"
# NOTEBOOK REFERENCE VALUES (data_mining_airline_updated.ipynb)
REF = {
    "rows": 103904, "features": 22, "selected": 20,
    "models": {"Logistic Regression": 0.8743, "Decision Tree": 0.9424,
               "Random Forest": 0.9624},
    "best": "Random Forest", "best_acc": 0.9624, "best_f1": 0.9559,
    "k": 2, "silhouette": 0.1526, "sizes": [46594, 57310],
    "rules": 1030, "itemsets": 558,
}

st.set_page_config(page_title="Airline Passenger Satisfaction Analysis",
                   page_icon="✈", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------- design system ----------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"], .stApp {
  background: #0f1419 !important;
  color: #e6ecf5;
  font-family: Inter, 'Segoe UI', sans-serif;
}
section.main > div { padding-top: 0.8rem; }
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3, h4 { font-family: Inter, sans-serif !important; letter-spacing: -0.01em; }

/* page hero */
.hero {
  background: linear-gradient(135deg, rgba(0,212,255,0.10), rgba(124,77,255,0.08) 55%, rgba(26,26,46,0.4)),
              radial-gradient(1000px 300px at 85% -50%, rgba(0,212,255,0.14), transparent);
  border: 1px solid rgba(0,212,255,0.22);
  border-radius: 14px; padding: 20px 26px; margin-bottom: 18px;
  box-shadow: 0 0 34px rgba(0,212,255,0.07);
}
.hero .h-title { font-size: 1.55rem; font-weight: 800; margin: 0; }
.hero .h-title .accent { color: #00d4ff; }
.hero .h-sub { color: #8b98ab; font-size: 0.85rem; margin-top: 4px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em;
  padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(0,212,255,0.35);
  color: #7fe8ff; background: rgba(0,212,255,0.06); white-space: nowrap;
}
.chip.ok    { border-color: rgba(67,160,71,0.5);  color: #7bd88a; background: rgba(67,160,71,0.08); }
.chip.warn  { border-color: rgba(255,179,0,0.45); color: #ffcf5c; background: rgba(255,179,0,0.07); }
.chip.purp  { border-color: rgba(124,77,255,0.5); color: #b79cff; background: rgba(124,77,255,0.08); }

/* section header */
.sec {
  display: flex; align-items: center; gap: 10px; margin: 6px 0 12px 0;
}
.sec .sec-num {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #0f1419;
  background: #00d4ff; border-radius: 6px; padding: 2px 8px; font-weight: 700;
}
.sec .sec-title { font-size: 1.02rem; font-weight: 700; }
.sec .sec-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,212,255,0.4), transparent); }

/* glass cards */
.glass {
  background: linear-gradient(160deg, rgba(26,26,46,0.92), rgba(21,27,45,0.85));
  border: 1px solid rgba(148,163,184,0.14);
  border-radius: 12px; padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.glass:hover { border-color: rgba(0,212,255,0.30); }

/* KPI */
.kpi { padding: 16px 18px; position: relative; overflow: hidden; }
.kpi .k-label { font-size: 0.68rem; letter-spacing: 0.14em; color: #8b98ab; text-transform: uppercase; }
.kpi .k-value { font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 700; margin-top: 4px; }
.kpi .k-note  { font-size: 0.72rem; color: #64748b; margin-top: 2px; }
.kpi::before {
  content: ''; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: linear-gradient(180deg, #00d4ff, #7c4dff);
}

/* metric bar rows */
.mrow { display: grid; grid-template-columns: 170px 1fr 64px; gap: 10px; align-items: center; margin: 7px 0; }
.mrow .m-name { font-size: 0.8rem; color: #b7c3d4; }
.mrow .m-val { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; text-align: right; color: #e6ecf5; }
.track { display: block; background: rgba(148,163,184,0.10); border-radius: 6px; height: 9px; overflow: hidden; }
.fill { display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, #00d4ff, #7c4dff); }

/* prediction result */
.pred-badge {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.08em;
  font-size: 1.15rem; padding: 10px 22px; border-radius: 10px; margin: 4px 0 10px 0;
}
.pred-badge.pos { color: #7bd88a; background: rgba(67,160,71,0.12); border: 1px solid rgba(67,160,71,0.55);
  box-shadow: 0 0 26px rgba(67,160,71,0.25); }
.pred-badge.neg { color: #ffcf5c; background: rgba(255,179,0,0.09); border: 1px solid rgba(255,179,0,0.5);
  box-shadow: 0 0 26px rgba(255,179,0,0.18); }
.factor {
  display: grid; grid-template-columns: 26px 1fr 90px; gap: 10px; align-items: center;
  padding: 10px 12px; border: 1px solid rgba(148,163,184,0.12); border-radius: 10px;
  background: rgba(15,20,25,0.55); margin: 7px 0;
}
.factor .f-rank { font-family: 'JetBrains Mono', monospace; color: #00d4ff; font-weight: 700; }
.factor .f-name { font-size: 0.83rem; }
.factor .f-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; text-align: right; }

/* streamlit widget tuning */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label, div[data-testid="stMultiselect"] label {
  font-size: 0.76rem !important; color: #8b98ab !important; letter-spacing: 0.03em;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,0.15); }
.stTabs [data-baseweb="tab"] {
  background: rgba(148,163,184,0.05); border-radius: 8px 8px 0 0;
  padding: 8px 18px; font-size: 0.85rem; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: rgba(0,212,255,0.12) !important; color: #7fe8ff !important; }
div[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,0.14); border-radius: 10px; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.25); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
.sidebar-note { color: #64748b; font-size: 0.72rem; line-height: 1.5; }
</style>""", unsafe_allow_html=True)


# ---------------- helpers ----------------
def md(html):
    """Render HTML reliably: collapse newlines/indentation so markdown
    never interprets indented lines as code blocks."""
    st.markdown(re.sub(r"\n\s+", " ", html), unsafe_allow_html=True)


def hero(title, accent, subtitle, chips):
    chip_html = "".join(f'<span class="chip {c}">{t}</span>' for t, c in chips)
    md(f"""
    <div class="hero">
      <p class="h-title">{title} <span class="accent">{accent}</span></p>
      <div class="h-sub">{subtitle}</div>
      <div class="chip-row">{chip_html}</div>
    </div>""")


def sec(num, title):
    md(f"""<div class="sec"><span class="sec-num">{num}</span>
    <span class="sec-title">{title}</span><span class="sec-line"></span></div>""")


def kpi_row(items):
    """items: list of (label, value, note, color)"""
    n = len(items)
    cols = st.columns(n, gap="small")
    for col, (label, value, note, color) in zip(cols, items):
        with col:
            md(f"""<div class="glass kpi">
              <div class="k-label">{label}</div>
              <div class="k-value" style="color:{color}">{value}</div>
              <div class="k-note">{note}</div></div>""")


def bar_rows(pairs, color="#00d4ff", fmt="{:.3f}"):
    for name, val in pairs:
        pct = max(0.0, min(1.0, val)) * 100
        md(f"""<div class="mrow"><span class="m-name">{name}</span>
          <span class="track"><span class="fill" style="width:{pct:.1f}%;
          background:linear-gradient(90deg,{color},#7c4dff)"></span></span>
          <span class="m-val">{fmt.format(val)}</span></div>""")


def dark_fig(fig, h=380, legend_h=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FAM, color="#c9d4e3", size=12),
        margin=dict(l=10, r=16, t=44, b=10), height=h,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    x=0, bgcolor="rgba(0,0,0,0)") if legend_h else {},
    )
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ---------------- data layer ----------------
_DF = {}


def fingerprint(df):
    return f"{df.shape[0]}x{df.shape[1]}|{pd.util.hash_pandas_object(df.iloc[:50]).sum()}"


@st.cache_data(show_spinner=False)
def load_default_df():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def clean_df(raw):
    data = raw.copy()
    dropped = [c for c in ("Unnamed: 0", "id") if c in data.columns]
    data.drop(columns=dropped, inplace=True, errors="ignore")
    dup = int(data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    missing_before = int(data["Arrival Delay in Minutes"].isna().sum())
    if "Arrival Delay in Minutes" in data.columns:
        med = data["Arrival Delay in Minutes"].median()
        data["Arrival Delay in Minutes"] = (data["Arrival Delay in Minutes"]
                                            .fillna(med).astype(int))
    meta = dict(dropped=dropped, dup=dup, imputed=missing_before,
                shape=data.shape)
    return data, meta


def set_active_df(df):
    fp = fingerprint(df)
    _DF[fp] = df
    st.session_state["active_fp"] = fp
    return fp


def get_active_df():
    fp = st.session_state.get("active_fp")
    if fp and fp in _DF:
        return _DF[fp]
    raw = load_default_df()
    data, meta = clean_df(raw)
    fp = set_active_df(data)
    st.session_state["data_meta"] = meta
    st.session_state["data_source"] = "bundled"
    return data


# ---------------- model layer ----------------
@st.cache_resource(show_spinner=False)
def train_pipeline(df):
    """Full predictive pipeline: split -> preprocess -> MI select -> 3 models."""
    t0 = time.time()
    model_data = df.copy()
    model_data[TARGET] = model_data[TARGET].map({SAT_NEG: 0, SAT_POS: 1})
    X = model_data.drop(columns=[TARGET])
    y = model_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    num_f = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_f = [c for c in X_train.columns if c not in num_f]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_f),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_f)])
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    feat_names = ([(f"num__{c}") for c in num_f]
                  + [f"cat__{c}" for c in pre.named_transformers_["cat"]
                     .get_feature_names_out(cat_f)])

    # mutual information feature selection (subsampled for speed)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(y_train), min(20000, len(y_train)), replace=False)
    mi = mutual_info_classif(Xtr[sub], y_train.values[sub],
                             random_state=42, n_jobs=-1)
    k = min(20, Xtr.shape[1])
    order = np.argsort(mi)[::-1]
    sel = order[:k]
    mi_scores = sorted([(feat_names[i], float(mi[i])) for i in order],
                       key=lambda t: -t[1])
    Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]
    sel_names = [feat_names[i] for i in sel]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                n_jobs=-1),
    }
    metrics, cms, rocs, train_acc = {}, {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, y_train)
        pred = mdl.predict(Xte_s)
        metrics[name] = dict(
            acc=accuracy_score(y_test, pred),
            prec=precision_score(y_test, pred),
            rec=recall_score(y_test, pred),
            f1=f1_score(y_test, pred))
        cms[name] = confusion_matrix(y_test, pred)
        train_acc[name] = accuracy_score(y_train, mdl.predict(Xtr_s))
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(Xte_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            rocs[name] = (fpr, tpr, float(roc_auc_score(y_test, proba)))

    # 5-fold CV on a stratified subsample (speed)
    sub2 = rng.choice(len(y_train), min(25000, len(y_train)), replace=False)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_res = {}
    for name, mdl in models.items():
        sc = cross_val_score(mdl, Xtr_s[sub2], y_train.values[sub2],
                             cv=cv, scoring="accuracy", n_jobs=-1)
        cv_res[name] = (float(sc.mean()), float(sc.std()))

    rf = models["Random Forest"]
    imp = sorted(zip(sel_names, rf.feature_importances_), key=lambda t: -t[1])

    sample_idx = y_test.index[:20]
    avp = pd.DataFrame({
        "Actual": [SAT_POS if v == 1 else SAT_NEG for v in y_test.loc[sample_idx]],
        "Predicted": [SAT_POS if v == 1 else SAT_NEG
                      for v in models["Random Forest"].predict(Xte_s[:20])],
    })
    avp["Correct"] = avp["Actual"] == avp["Predicted"]

    return dict(models=models, pre=pre, sel=sel, sel_names=sel_names,
                feat_names=feat_names, mi_scores=mi_scores, k=k,
                metrics=metrics, cms=cms, rocs=rocs, cv=cv_res,
                train_acc=train_acc, importance=imp, avp=avp,
                Xtr_shape=Xtr.shape, Xte_shape=Xte.shape,
                n_train=len(y_train), n_test=len(y_test),
                runtime=time.time() - t0)


def pretty_feat(f):
    f = f.replace("num__", "").replace("cat__", "")
    for frag in ("Type of Travel_", "Customer Type_", "Class_"):
        if f.startswith(frag):
            return f.replace(frag, f"{frag.rstrip('_')} · ")
    return f


@st.cache_resource(show_spinner=False)
def cluster_pipeline(df):
    """K-Means descriptive mining: elbow + silhouette + final clustering."""
    feats = (["Age", "Flight Distance"] + RATING_COLS + NUM_COLS[2:])
    cd = df[feats].fillna(df[feats].median(numeric_only=True))
    cs = StandardScaler().fit_transform(cd.values)

    k_values = list(range(2, 9))
    inertias, sils, lab2 = [], {}, None
    for kk in k_values:
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(cs)
        inertias.append(float(km.inertia_))
        if kk == 2:
            lab2 = km.labels_
        subi = np.random.RandomState(42).choice(
            len(cs), min(6000, len(cs)), replace=False)
        sils[kk] = float(silhouette_score(cs[subi], km.predict(cs[subi])))

    best_k = 2
    top_sil = max(sils.values())
    for kk in k_values:                       # parsimony: smallest K near max
        if sils[kk] >= top_sil - 0.01:
            best_k = kk
            break
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(cs)
    labels = km_final.predict(cs)

    sizes = pd.Series(labels).value_counts().sort_index()
    sat_split = pd.crosstab(labels, df[TARGET], normalize="index")
    profile = (df[feats].groupby(labels).mean())
    rate = df[TARGET].groupby(labels).apply(lambda s: (s == SAT_POS).mean())

    return dict(feats=feats, k_values=k_values, inertias=inertias, sils=sils,
                best_k=best_k, sizes=sizes.to_dict(), sat_split=sat_split,
                profile=profile, rate=rate.to_dict(), n_clusters=int(best_k))


@st.cache_data(show_spinner=False)
def apriori_itemsets(df, min_support):
    """Frequent itemsets over High_[rating] (>=4) + Satisfied transactions."""
    from mlxtend.frequent_patterns import apriori
    tx = pd.DataFrame(index=df.index)
    for col in RATING_COLS:
        nm = col.replace(" ", "_").replace("/", "_")
        tx[f"High_{nm}"] = (df[col] >= 4).astype(bool)
    tx["Satisfied"] = (df[TARGET] == SAT_POS)
    fi = apriori(tx.astype(bool), min_support=min_support,
                 use_colnames=True, max_len=3)
    fi = fi.sort_values("support", ascending=False).reset_index(drop=True)
    return fi


def association_rules_from(fi, min_conf):
    from mlxtend.frequent_patterns import association_rules
    rules = association_rules(fi, metric="confidence",
                              min_threshold=min_conf)
    rules = rules.sort_values(["lift", "confidence"],
                              ascending=False).reset_index(drop=True)
    return rules


def fmt_itemset(s):
    parts = sorted(x.replace("High_", "").replace("_", " ") for x in s)
    return " + ".join(parts)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
def page_overview(df):
    n = len(df)
    sat_rate = (df[TARGET] == SAT_POS).mean()
    meta = st.session_state.get("data_meta", {})
    src = st.session_state.get("data_source", "bundled")

    hero("AIRLINE PASSENGER", "SATISFACTION ANALYSIS",
         "IS-212 Data & Knowledge Mining · University of Computer Studies, Yangon · 2025–2026",
         [("DATASET " + ("UPLOADED" if src == "upload" else "LOADED"), "ok"),
          (f"{n:,} PASSENGERS", ""), ("23 FIELDS", ""),
          ("TARGET · SATISFACTION", "purp"),
          ("PIPELINE · EDA → K-MEANS → APRIORI → CLASSIFICATION", "")])

    kpi_row([
        ("Passengers", f"{n:,}", "survey records", CYAN),
        ("Features", "22", "+ 1 target · 14 service ratings", PURPLE),
        ("Satisfied", f"{sat_rate*100:.1f}%", f"{int(sat_rate*n):,} of {n:,}", GREEN),
        ("Missing", f"{int(df['Arrival Delay in Minutes'].isna().sum()):,}",
         "Arrival Delay (median-imputed)", AMBER),
        ("Best Model", "RF", f"notebook ref acc {REF['best_acc']:.4f}", CYAN),
    ])

    sec("01", "TARGET & SEGMENT STRUCTURE")
    c1, c2 = st.columns([1.25, 1.5], gap="large")
    with c1:
        sc = df[TARGET].value_counts()
        fig = go.Figure(go.Pie(
            labels=sc.index.tolist(), values=sc.values, hole=0.58,
            marker=dict(colors=[RED, GREEN],
                        line=dict(color="#0f1419", width=3)),
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent}", textfont=dict(size=15)))
        fig.add_annotation(text=f"<b>{sat_rate*100:.1f}%</b><br>satisfied",
                           showarrow=False, font=dict(size=22, color="#e6ecf5"))
        st.plotly_chart(dark_fig(fig, 430, False), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        m1, m2, m3 = st.columns(3)
        for cont, title, colors, cmap in [
            (m1, "Class", ["Business", "Eco", "Eco Plus"],
             [CYAN, PURPLE, AMBER]),
            (m2, "Type of Travel", ["Business travel", "Personal Travel"],
             [CYAN, AMBER]),
            (m3, "Customer Type", ["Loyal Customer", "disloyal Customer"],
             [GREEN, RED]),
        ]:
            with cont:
                vc = df[title].value_counts().reindex(colors)
                fig = go.Figure(go.Bar(
                    x=vc.values, y=[i.replace(" Customer", "").replace(" travel", "")
                                    for i in vc.index],
                    orientation="h",
                    marker=dict(color=cmap, opacity=0.85),
                    text=[f"{v:,}" for v in vc.values], textposition="outside",
                    textfont=dict(size=10)))
                fig.update_layout(title=dict(text=title, font=dict(size=12)),
                                  margin=dict(l=10, r=44, t=34, b=8),
                                  xaxis=dict(visible=False,
                                             range=[0, float(vc.max()) * 1.55]),
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(dark_fig(fig, 200, False),
                                use_container_width=True,
                                config={"displayModeBar": False})
        st.caption("Exact marginal distributions — matching the notebook's value counts.")

    sec("02", "DATA PREVIEW · head(10)")
    st.dataframe(df.head(10), use_container_width=True, height=270)

    sec("03", "DATA QUALITY AUDIT")
    dup = meta.get("dup", int(df.duplicated().sum()))
    delays = df[["Departure Delay in Minutes", "Arrival Delay in Minutes"]]
    out_n = int(((delays > delays.quantile(0.75) + 1.5 *
                  (delays.quantile(0.75) - delays.quantile(0.25))).sum()).sum())
    bad_rating = int(sum(((df[c] < 0) | (df[c] > 5)).sum() for c in RATING_COLS))
    q1, q2, q3, q4 = st.columns(4)
    for col, lab, val, note, colr in [
        (q1, "Missing values", f"{int(df.isna().sum().sum()):,}",
         "all in Arrival Delay → median imputed", AMBER),
        (q2, "Duplicate rows", f"{dup:,}", "drop_duplicates applied", CYAN),
        (q3, "Delay outliers (IQR)", f"{out_n:,}",
         "kept — extreme delays are real events", RED),
        (q4, "Invalid ratings", f"{bad_rating:,}",
         "all service ratings within 0–5", GREEN),
    ]:
        with col:
            md("""<div class="glass kpi"><div class="k-label">""" + f"{lab}</div>"
               + f"<div class=\"k-value\" style=\"color:{colr};font-size:1.4rem\">{val}</div>"
               + f"<div class=\"k-note\">{note}</div></div>")

    with st.expander("Column inventory · dtypes & unique counts", expanded=False):
        inv = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Unique": [df[c].nunique() for c in df.columns],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
        })
        st.dataframe(inv, use_container_width=True, height=460)


# ============================================================
# PAGE 2 — EDA EXPLORER
# ============================================================
def page_eda(df):
    hero("EDA", "EXPLORER",
         "Distributions · skewness · outliers · segment rates · correlations — "
         "mirroring the notebook's Data Understanding & EDA section",
         [("20 NUMERIC & CATEGORICAL FIELDS", ""), ("RIGHT-SKEWED DELAYS", "warn"),
          ("LOG SCALE TOGGLE", "purp")])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🎯 Outliers & Skewness",
         "⚖ Satisfaction Rates", "🔗 Correlations"])

    with tab1:
        pick = st.selectbox(
            "Numeric field", NUM_COLS + RATING_COLS,
            index=NUM_COLS.index("Flight Distance"))
        use_log = st.checkbox("Log x-axis (recommended for delays)",
                              value="Delay" in pick)
        c1, c2 = st.columns([2.4, 1], gap="large")
        with c1:
            xlabel = pick
            if pick in RATING_COLS:
                fig = px.histogram(df, x=pick, color=TARGET,
                                   barmode="group", nbins=12,
                                   color_discrete_map={SAT_NEG: RED, SAT_POS: GREEN})
            else:
                xd = df[[pick, TARGET]].copy()
                if use_log:
                    xd[pick] = np.log10(xd[pick].clip(lower=0.5))
                    xlabel = f"log10({pick})"
                fig = px.histogram(xd, x=pick, color=TARGET,
                                   nbins=60, barmode="overlay",
                                   color_discrete_map={SAT_NEG: RED, SAT_POS: GREEN},
                                   opacity=0.62)
                fig.update_traces(marker_line_width=0)
            fig.update_layout(title=f"Distribution of {pick} by satisfaction",
                              bargap=0.02)
            fig.update_xaxes(title=xlabel)
            st.plotly_chart(dark_fig(fig, 420), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            s = df[pick]
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"**{pick}**")
            bar_rows([
                ("mean", float(s.mean())), ("std", float(s.std())),
                ("min", float(s.min())), ("25%", float(s.quantile(.25))),
                ("median", float(s.median())), ("75%", float(s.quantile(.75))),
                ("max", float(s.max())),
            ], fmt="{:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1.15, 1], gap="large")
        with c1:
            st.markdown("**IQR outlier bounds — continuous fields**")
            rows = []
            for col in SKEW_COLS:
                q1v, q3v = df[col].quantile([0.25, 0.75])
                iqr = q3v - q1v
                lo, hi = q1v - 1.5 * iqr, q3v + 1.5 * iqr
                cnt = int(((df[col] < lo) | (df[col] > hi)).sum())
                rows.append(dict(Field=col, Q1=round(q1v, 1), Q3=round(q3v, 1),
                                 IQR=round(iqr, 1), Lower=round(lo, 1),
                                 Upper=round(hi, 1), Outliers=cnt))
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=190)
            st.caption("Outliers are inspected but not deleted — extreme flight "
                       "distances and delays are legitimate operational values "
                       "(same decision as the notebook).")
            fig = make_subplots(rows=1, cols=3, subplot_titles=[
                "Flight Distance", "Departure Delay", "Arrival Delay"])
            for i, col in enumerate(SKEW_COLS[1:], 1):
                fig.add_trace(go.Box(y=df[col], marker_color=CYAN,
                                     whiskerwidth=0.4), row=1, col=i)
            fig.update_layout(showlegend=False, height=330)
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", row=1, col=2)
            fig.update_yaxes(type="log", row=1, col=3)
            st.plotly_chart(dark_fig(fig, 330, False), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            sk = df[SKEW_COLS].skew().sort_values(ascending=False)
            fig = go.Figure(go.Bar(
                x=sk.values, y=sk.index, orientation="h",
                marker=dict(color=[RED if abs(v) > 1 else AMBER if abs(v) > .5
                                   else GREEN for v in sk.values], opacity=0.9),
                text=[f"{v:.2f}" for v in sk.values], textposition="outside"))
            fig.update_layout(title="Skewness (>1 = highly skewed)",
                              margin=dict(l=10, r=52, t=44, b=8),
                              xaxis=dict(range=[-0.6, float(sk.max()) * 1.25]))
            st.plotly_chart(dark_fig(fig, 330, False), use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("Delays are heavily right-skewed; Age is essentially "
                       "symmetric — matching the notebook's skewness panel.")

    with tab3:
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            st.markdown("**Satisfied rate by segment**")
            chosen = st.selectbox("Segment", CAT_COLS, index=1)
            overall = (df[TARGET] == SAT_POS).mean()
            rate = pd.crosstab(df[chosen], df[TARGET], normalize="index")[SAT_POS]
            fig = go.Figure(go.Bar(
                x=rate.index, y=rate.values,
                marker=dict(color=[GREEN if v >= overall else RED
                                 for v in rate.values], opacity=0.88),
                text=[f"{v*100:.1f}%" for v in rate.values], textposition="outside"))
            fig.update_layout(title=f"% satisfied · {chosen}",
                              yaxis=dict(tickformat=".0%", range=[0, 1.12]),
                              margin=dict(l=10, r=10, t=44, b=8))
            st.plotly_chart(dark_fig(fig, 380, False), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown("**Average service ratings — satisfied vs neutral/dissatisfied**")
            means = df.groupby(TARGET)[RATING_COLS].mean().T
            means["gap"] = means[SAT_POS] - means[SAT_NEG]
            means = means.sort_values("gap")
            fig = go.Figure()
            fig.add_bar(y=means.index, x=means[SAT_NEG], name=SAT_NEG,
                        orientation="h", marker=dict(color=RED, opacity=0.8))
            fig.add_bar(y=means.index, x=means[SAT_POS], name=SAT_POS,
                        orientation="h", marker=dict(color=GREEN, opacity=0.8))
            fig.update_layout(barmode="group", height=560,
                              margin=dict(l=10, r=16, t=30, b=8),
                              xaxis=dict(title="mean rating (0–5)"))
            st.plotly_chart(dark_fig(fig, 560), use_container_width=True,
                            config={"displayModeBar": False})
            top = means.index[-1]
            st.caption(f"Largest gap: **{top}** "
                       f"({means.loc[top, 'gap']:+.2f}) — consistent with the "
                       "notebook's rating comparison and later MI/RF rankings.")

    with tab4:
        corr_cols = NUM_COLS + RATING_COLS
        cm = df[corr_cols].corr()
        fig = px.imshow(cm, text_auto=".2f", aspect="auto",
                        color_continuous_scale=["#ef5350", "#141b2d", "#00d4ff"],
                        zmin=-1, zmax=1, width=1000, height=760)
        fig.update_layout(font=dict(size=9))
        st.plotly_chart(dark_fig(fig, 760, False), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Departure ↔ Arrival delay correlate near-perfectly; "
                   "Cleanliness ↔ Food & drink / Entertainment form the "
                   "comfort block — both reproduced from the source notebook.")


# ============================================================
# PAGE 3 — DESCRIPTIVE MINING
# ============================================================
def page_mining(df):
    hero("DESCRIPTIVE", "MINING",
         "K-Means clustering (elbow + silhouette) and Apriori association "
         "rules — the notebook's unsupervised learning section, recomputed live",
         [("K-MEANS · K = 2–8 SCAN", ""), ("SILHOUETTE SAMPLE 6K", "purp"),
          ("APRIORI · HIGH_RATING ≥ 4", "ok")])

    tab_km, tab_ar = st.tabs(["🛰 K-Means Clustering", "🧩 Association Rules"])

    with tab_km:
        with st.status("Running K-Means scan (K = 2…8) — first load only",
                       expanded=False) as stt:
            st.write("Scaling 18 clustering features …")
            cl = cluster_pipeline(df)
            st.write(f"Silhouette scan done · best K = **{cl['best_k']}**")
            stt.update(label="K-Means scan complete", state="complete")

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure(go.Scatter(x=cl["k_values"], y=cl["inertias"],
                                       mode="lines+markers",
                                       line=dict(color=CYAN, width=3),
                                       marker=dict(size=9)))
            fig.update_layout(title="Elbow method · inertia",
                              height=300, margin=dict(t=44))
            st.plotly_chart(dark_fig(fig, 300, False), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            fig = go.Figure(go.Bar(
                x=[str(k) for k in cl["sils"].keys()],
                y=list(cl["sils"].values()),
                marker=dict(color=[CYAN if k == cl["best_k"] else "rgba(124,77,255,0.55)"
                                   for k in cl["sils"].keys()])))
            fig.update_layout(title="Silhouette score by K",
                              height=300, margin=dict(t=44))
            st.plotly_chart(dark_fig(fig, 300, False), use_container_width=True,
                            config={"displayModeBar": False})
        with c3:
            best_sil = cl["sils"][cl["best_k"]]
            md(f"""<div class="glass" style="height:300px;display:flex;
            flex-direction:column;justify-content:center;gap:12px">
              <div class="k-label">SELECTED K</div>
              <div class="k-value" style="color:#00d4ff">{cl['best_k']}</div>
              <div class="k-note">smallest K within 0.01 of max silhouette
              ({best_sil:.3f}) — notebook selected K=2 ({REF['silhouette']:.4f})</div>
              <div class="k-label" style="margin-top:10px">NOTEBOOK REFERENCE</div>
              <div class="k-note">sizes {REF['sizes'][0]:,} / {REF['sizes'][1]:,}
              · sat split 17% / 65%</div>
            </div>""")

        c1, c2 = st.columns([1, 1.7], gap="large")
        with c1:
            st.markdown("**Cluster sizes & satisfaction split**")
            sizes = cl["sizes"]
            fig = go.Figure()
            for cid in sorted(sizes):
                row = cl["sat_split"].loc[cid]
                first = cid == sorted(sizes)[0]
                fig.add_bar(
                    y=[f"Cluster {cid}"],
                    x=[row.get(SAT_POS, 0) * sizes[cid]],
                    name=SAT_POS, orientation="h", showlegend=first,
                    marker=dict(color=GREEN, opacity=0.85))
                fig.add_bar(
                    y=[f"Cluster {cid}"],
                    x=[row.get(SAT_NEG, 0) * sizes[cid]],
                    name=SAT_NEG, orientation="h", showlegend=first,
                    marker=dict(color=RED, opacity=0.8))
            fig.update_layout(barmode="stack", height=190,
                              margin=dict(l=10, r=10, t=14, b=8),
                              legend=dict(orientation="h", y=1.12))
            fig.update_xaxes(visible=False)
            st.plotly_chart(dark_fig(fig, 200, False), use_container_width=True,
                            config={"displayModeBar": False})
            for cid in sorted(sizes):
                md(f"""<div class="mrow"><span class="m-name">Cluster {cid}
                    · {sizes[cid]:,} pax</span>
                    <span class="track"><span class="fill" style="width:{cl['rate'][cid]*100:.0f}%"></span></span>
                    <span class="m-val">{cl['rate'][cid]*100:.1f}% sat</span></div>""")
        with c2:
            st.markdown("**Cluster profile — mean of each feature (z-colored)**")
            prof = cl["profile"].T
            prof_z = (prof - prof.values.mean(axis=0)) / (prof.values.std(axis=0) + 1e-9)
            fig = px.imshow(prof_z, aspect="auto",
                            color_continuous_scale=["#ef5350", "#141b2d", "#43a047"],
                            x=[f"C{i}" for i in prof.columns],
                            y=prof.index, text_auto=".1f")
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=8))
            fig.update_traces(showscale=False)
            st.plotly_chart(dark_fig(fig, 560, False), use_container_width=True,
                            config={"displayModeBar": False})
        st.caption("Clusters separate on the service-quality axis: the "
                   "low-quality cluster (low entertainment / food / boarding) "
                   "is heavily dissatisfied, mirroring the notebook's "
                   "cluster-satisfaction crosstab.")

    with tab_ar:
        st.markdown("**Transaction encoding:** `High_<service>` = rating ≥ 4, "
                    "plus `Satisfied` — exactly the notebook's Apriori setup.")
        c1, c2 = st.columns(2)
        with c1:
            min_sup = st.slider("min support", 0.02, 0.30, 0.10, 0.01,
                                help="Notebook used 0.10")
        with c2:
            min_conf = st.slider("min confidence", 0.30, 0.99, 0.60, 0.01,
                                 help="Notebook used 0.60")

        fi = apriori_itemsets(df, round(min_sup, 2))
        rules = association_rules_from(fi, round(min_conf, 2))

        kpi_row([
            ("Frequent itemsets", f"{len(fi):,}",
             f"notebook ref {REF['itemsets']} @ 0.10", CYAN),
            ("Rules", f"{len(rules):,}",
             f"notebook ref {REF['rules']} @ conf 0.60", PURPLE),
            ("Max lift", f"{rules['lift'].max():.2f}" if len(rules) else "—",
             "top rule strength", GREEN),
        ])

        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            st.markdown("**Top frequent itemsets by support**")
            top = fi.head(10).copy()
            top["label"] = top["itemsets"].apply(fmt_itemset)
            fig = go.Figure(go.Bar(
                x=top["support"], y=top["label"], orientation="h",
                marker=dict(color=CYAN, opacity=0.85),
                text=[f"{v:.3f}" for v in top["support"]],
                textposition="outside"))
            fig.update_layout(height=380, margin=dict(l=10, r=52, t=20, b=8),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(dark_fig(fig, 380, False), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.markdown("**Top rules by lift**")
            if len(rules):
                show = rules.head(12).copy()
                show["antecedents"] = show["antecedents"].apply(fmt_itemset)
                show["consequents"] = show["consequents"].apply(fmt_itemset)
                show = show[["antecedents", "consequents", "support",
                             "confidence", "lift"]].round(3)
                show.columns = ["IF (antecedent)", "THEN (consequent)",
                                "support", "confidence", "lift"]
                st.dataframe(show, use_container_width=True, height=380)
            else:
                st.info("No rules at this threshold — lower support/confidence.")
        st.caption("High service ratings travel together and imply "
                   "`Satisfied` with high confidence — the same "
                   "'good experience ⇒ satisfied' structure the notebook's "
                   "rules exposed.")


# ============================================================
# PAGE 4 — MODEL LAB
# ============================================================
def page_lab(df):
    hero("MODEL", "LAB",
         "Logistic Regression · Decision Tree · Random Forest — with mutual "
         "information feature selection, CV and full evaluation, "
         "reproducing the notebook's predictive mining section",
         [("80/20 STRATIFIED SPLIT", ""), ("MUTUAL INFO · TOP 20", "purp"),
          ("5-FOLD CV", "ok")])

    with st.status("Training classification pipeline — first load only "
                   "(~1 min)", expanded=False) as stt:
        st.write("Preprocessing · median impute → one-hot → scale …")
        tp = train_pipeline(df)
        st.write(f"Feature selection: kept top **{tp['k']}** of "
                 f"{len(tp['feat_names'])} processed features")
        st.write("Fitting LR / DT / RF + 5-fold CV …")
        stt.update(label=f"Models ready · {tp['runtime']:.0f}s",
                   state="complete")

    sec("01", "PIPELINE")
    steps = ["Split 80/20", "Impute + One-Hot + Scale", "SelectKBest (MI)",
             "Train ×3", "Evaluate + CV"]
    md('<div style="display:flex;gap:8px;flex-wrap:wrap">' +
       "".join(
           f'<span class="chip {"purp" if i == 2 else ""}">'
           f'{i+1}. {s}</span>' for i, s in enumerate(steps)) +
       f'<span class="chip ok">PROCESSED SHAPE '
       f'{tp["Xtr_shape"][0]:,}×{tp["Xtr_shape"][1]}</span></div>')

    sec("02", "TEST-SET PERFORMANCE")
    order = sorted(tp["metrics"], key=lambda m: -tp["metrics"][m]["acc"])
    for name in order:
        m = tp["metrics"][name]
        is_best = name == order[0]
        color = CYAN if is_best else ("#8b98ab" if name == order[-1] else PURPLE)
        rows_html = "".join(
            f"""<div class="mrow"><span class="m-name">{k}</span>
            <span class="track"><span class="fill" style="width:{v*100:.1f}%;
            background:linear-gradient(90deg,{color},#7c4dff)"></span></span>
            <span class="m-val">{v:.4f}</span></div>"""
            for k, v in (("Accuracy", m["acc"]), ("Precision", m["prec"]),
                         ("Recall", m["rec"]), ("F1", m["f1"])))
        md(f"""<div class="glass" style="padding:14px 18px;margin:10px 0">
          <div style="display:flex;justify-content:space-between;align-items:baseline">
            <span style="font-weight:700">{name} {'<span class="chip ok">BEST</span>' if is_best else ''}</span>
            <span style="font-family:'JetBrains Mono',monospace;color:{color}">
              acc {m['acc']:.4f} · f1 {m['f1']:.4f}</span></div>
          <div style="margin-top:10px">{rows_html}</div></div>""")
    st.caption("Notebook reference — LR 0.8743 · DT 0.9424 · RF 0.9624. "
               "Recomputed values on the reconstructed dataset land within "
               "≈1 pt; the ordering and RF lead are preserved.")

    sec("03", "TRAIN VS TEST · OVERFIT CHECK")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        for i, (label, key) in enumerate([("Training", "train_acc"),
                                          ("Testing", "acc")]):
            fig.add_bar(
                name=label, x=list(tp["metrics"].keys()),
                y=[tp[key][n] if key == "train_acc" else tp["metrics"][n]["acc"]
                   for n in tp["metrics"]],
                marker=dict(color=[CYAN, PURPLE][i], opacity=0.88))
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05],
                                                     tickformat=".0%"),
                          title="Train vs test accuracy", height=330)
        st.plotly_chart(dark_fig(fig, 330, False), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown("**5-fold stratified CV (train subsample)**")
        for name in order:
            mean, sd = tp["cv"][name]
            md(f"""<div class="mrow"><span class="m-name">{name}</span>
              <span class="track"><span class="fill" style="width:{mean*100:.1f}%"></span></span>
              <span class="m-val">{mean:.4f} ±{sd:.4f}</span></div>""")
        st.caption("Low fold-to-fold variance → stable models. "
                   "DT/RF train accuracy = 1.0 (unpruned trees memorise), "
                   "yet test/CV stay high — same pattern as the notebook.")

    sec("04", "CONFUSION MATRICES")
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=list(tp["cms"].keys()))
    for i, (name, cmx) in enumerate(tp["cms"].items(), 1):
        fig.add_trace(go.Heatmap(
            z=cmx, x=["Pred: N/D", "Pred: Sat"],
            y=["True: N/D", "True: Sat"],
            text=[[f"{v:,}" for v in row] for row in cmx],
            texttemplate="%{text}", colorscale=[[0, "#141b2d"], [1, "#00d4ff"]],
            showscale=False), row=1, col=i)
    fig.update_layout(height=340)
    st.plotly_chart(dark_fig(fig, 340, False), use_container_width=True,
                    config={"displayModeBar": False})

    sec("05", "ROC CURVES")
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(148,163,184,0.4)", dash="dash"))
    for name, color in [("Random Forest", CYAN), ("Decision Tree", PURPLE),
                        ("Logistic Regression", AMBER)]:
        if name in tp["rocs"]:
            fpr, tpr, auc = tp["rocs"][name]
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name,
                            line=dict(color=color, width=2.5),
                            hovertemplate=f"{name} AUC {auc:.4f}")
    fig.update_layout(xaxis=dict(title="False positive rate"),
                      yaxis=dict(title="True positive rate"),
                      height=400)
    st.plotly_chart(dark_fig(fig, 400), use_container_width=True,
                    config={"displayModeBar": False})

    sec("06", "FEATURE SELECTION & IMPORTANCE")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Mutual information — all processed features**")
        mi_df = pd.DataFrame(tp["mi_scores"], columns=["Feature", "MI"])
        mi_df["sel"] = mi_df["Feature"].isin(tp["sel_names"])
        show = mi_df.head(20).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=show["MI"], y=[pretty_feat(f) for f in show["Feature"]],
            orientation="h",
            marker=dict(color=[CYAN if s else "rgba(148,163,184,0.35)"
                               for s in show["sel"]])))
        fig.update_layout(height=560, margin=dict(l=10, r=30, t=20, b=8))
        st.plotly_chart(dark_fig(fig, 560, False), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Cyan = kept by SelectKBest (top 20).")
    with c2:
        st.markdown("**Random Forest feature importance (top 15)**")
        imp = tp["importance"][:15][::-1]
        fig = go.Figure(go.Bar(
            x=[v for _, v in imp],
            y=[pretty_feat(f) for f, _ in imp],
            orientation="h", marker=dict(color=PURPLE, opacity=0.9),
            text=[f"{v:.3f}" for _, v in imp], textposition="outside"))
        fig.update_layout(height=560, margin=dict(l=10, r=52, t=20, b=8))
        st.plotly_chart(dark_fig(fig, 560, False), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Notebook's top drivers — Online boarding, Inflight wifi, "
                   "Business travel, Business class — rank the same way here.")

    sec("07", "ACTUAL VS PREDICTED (SAMPLE)")
    st.dataframe(tp["avp"], use_container_width=True, height=460)


# ============================================================
# PAGE 5 — PREDICT
# ============================================================
def ring_svg(p, color):
    r = 52
    c = 2 * 3.14159 * r
    filled = c * p
    return f"""
    <svg width="132" height="132" viewBox="0 0 132 132">
      <circle cx="66" cy="66" r="{r}" stroke="rgba(148,163,184,0.15)"
        stroke-width="9" fill="none"/>
      <circle cx="66" cy="66" r="{r}" stroke="{color}" stroke-width="9"
        fill="none" stroke-linecap="round"
        stroke-dasharray="{filled:.1f} {c:.1f}"
        transform="rotate(-90 66 66)"/>
      <text x="66" y="62" text-anchor="middle" fill="#e6ecf5"
        font-family="JetBrains Mono" font-size="21" font-weight="700">{p*100:.0f}%</text>
      <text x="66" y="82" text-anchor="middle" fill="#8b98ab" font-size="9.5"
        font-family="Inter">confidence</text>
    </svg>"""


def page_predict(df):
    hero("LIVE", "PREDICTION",
         "Score a passenger profile against all three trained classifiers — "
         "badge, probability split, confidence ring and the factors "
         "pushing the verdict",
         [("RF · PRIMARY VERDICT", "ok"), ("NEUTRAL-PERTURBATION EXPLAINABILITY", "purp"),
          ("NO DATA LEAVE", "")])

    tp = train_pipeline(df)
    models = tp["models"]

    c_form, c_out = st.columns([1.05, 1.5], gap="large")
    with c_form:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Passenger profile**")
        g1, g2 = st.columns(2)
        gender = g1.selectbox("Gender", ["Male", "Female"])
        ctype = g2.selectbox("Customer Type",
                             ["Loyal Customer", "disloyal Customer"])
        g3, g4 = st.columns(2)
        travel = g3.selectbox("Type of Travel",
                              ["Business travel", "Personal Travel"])
        klass = g4.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        age = st.slider("Age", 7, 85, 40)
        dist = st.slider("Flight Distance", 31, 4983, 850, step=1)
        g5, g6 = st.columns(2)
        dep_d = g5.number_input("Departure Delay (min)", 0, 1592, 0)
        arr_d = g6.number_input("Arrival Delay (min)", 0, 1584, 0)

        st.markdown("**Service ratings (0–5)**")
        means = {c: int(round(df[c].mean())) for c in RATING_COLS}
        defaults = {"Inflight wifi service": 3, "Online boarding": 4,
                    "Seat comfort": 4, "Inflight entertainment": 4,
                    "Baggage handling": 4, "Inflight service": 4}
        ratings = {}
        half = (len(RATING_COLS) + 1) // 2
        colA, colB = st.columns(2)
        for i, c in enumerate(RATING_COLS):
            tgt = colA if i < half else colB
            ratings[c] = tgt.slider(c, 0, 5, defaults.get(c, means[c]),
                                    key=f"pf_{i}")
        st.markdown("</div>", unsafe_allow_html=True)

        btn = st.button("PREDICT SATISFACTION", use_container_width=True,
                        type="primary")
        st.markdown(
            """<style>div[data-testid="stButton"] > button {
            border-radius: 10px; font-weight: 800; letter-spacing: .06em;
            background: linear-gradient(90deg,#00d4ff,#7c4dff); border: 0;
            color: #0f1419; box-shadow: 0 6px 24px rgba(0,212,255,.25);}
            div[data-testid="stButton"] > button:hover {
            filter: brightness(1.1);}</style>""", unsafe_allow_html=True)
        st.markdown('<p class="sidebar-note">Defaults follow dataset means; '
                    'ratings capped at 0–5 like the survey.</p>',
                    unsafe_allow_html=True)

    row = {"Gender": gender, "Customer Type": ctype, "Age": age,
           "Type of Travel": travel, "Class": klass, "Flight Distance": dist,
           "Departure Delay in Minutes": int(dep_d),
           "Arrival Delay in Minutes": int(arr_d), **ratings}

    with c_out:
        if not btn and "last_pred" not in st.session_state:
            md("""<div class="glass" style="height:220px;display:flex;
              align-items:center;justify-content:center;color:#64748b">
              Configure a profile and press <b>&nbsp;PREDICT SATISFACTION&nbsp;</b>
              to score it.</div>""")
            return
        if btn:
            st.session_state["last_pred"] = row
        row_in = pd.DataFrame([st.session_state["last_pred"]])

        Xp = tp["pre"].transform(row_in)[:, tp["sel"]]
        probas, labels = {}, {}
        for name, mdl in models.items():
            probas[name] = float(mdl.predict_proba(Xp)[0, 1])
            labels[name] = SAT_POS if probas[name] >= 0.5 else SAT_NEG

        p = probas["Random Forest"]
        pos = p >= 0.5
        badge = ("pos", "✓ SATISFIED") if pos else ("neg", "⚠ NEUTRAL / DISSATISFIED")
        color = GREEN if pos else AMBER

        chips_html = "\n".join(
            f'<span class="chip ok">{nm.split()[0]} · {probas[nm]:.2f}</span>'
            if lab == SAT_POS else
            f'<span class="chip warn">{nm.split()[0]} · {probas[nm]:.2f}</span>'
            for nm, lab in labels.items())
        md(f"""<div class="glass">
          <div class="pred-badge {badge[0]}">{badge[1]}</div>
          <div style="display:flex;gap:26px;align-items:center">
            {ring_svg(max(p, 1-p), color)}
            <div style="flex:1">
              <div class="mrow"><span class="m-name">P(satisfied)</span>
                <span class="track"><span class="fill" style="width:{p*100:.1f}%;
                background:linear-gradient(90deg,#43a047,#00d4ff)"></span></span>
                <span class="m-val">{p:.3f}</span></div>
              <div class="mrow"><span class="m-name">P(neutral / dissat.)</span>
                <span class="track"><span class="fill" style="width:{(1-p)*100:.1f}%;
                background:linear-gradient(90deg,#ffb300,#ef5350)"></span></span>
                <span class="m-val">{1-p:.3f}</span></div>
              <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">{chips_html}</div>
            </div></div></div>""")

        # neutral-perturbation contributions
        base = float(models["Random Forest"].predict_proba(Xp)[0, 1])
        contribs = []
        for j, feat in enumerate(tp["sel_names"]):
            pert = Xp.copy()
            pert[:, j] = 0.0          # neutral = train-mean (scaled space)
            p_pert = float(models["Random Forest"].predict_proba(pert)[0, 1])
            contribs.append((pretty_feat(feat), base - p_pert))
        contribs = sorted(contribs, key=lambda t: -abs(t[1]))[:3]
        st.markdown("**Top 3 contributing factors**")
        for i, (fname, delta) in enumerate(contribs, 1):
            sign = "+" if delta >= 0 else "−"
            colr = GREEN if delta >= 0 else RED
            md(f"""<div class="factor">
              <span class="f-rank">{i}</span>
              <span class="f-name">{fname}</span>
              <span class="f-delta" style="color:{colr}">{sign}{abs(delta)*100:.1f} pp</span>
            </div>""")

        # benchmark
        seg = df[(df["Class"] == klass) & (df["Type of Travel"] == travel)]
        seg_rate = (seg[TARGET] == SAT_POS).mean() if len(seg) else 0.0
        md(f"""<div class="glass" style="margin-top:14px">
          <span class="k-label">SIMILAR PASSENGERS BENCHMARK</span>
          <div class="k-value" style="color:{CYAN};font-size:1.3rem">
          {seg_rate*100:.1f}%</div>
          <div class="k-note">share satisfied among {len(seg):,} passengers in
          <b>{klass} × {travel}</b> — your profile scores
          {'above' if pos else 'below'} that baseline</div></div>""")


# ============================================================
# PAGE 6 — ABOUT
# ============================================================
def page_about(df):
    hero("ABOUT THIS", "PROJECT",
         "IS-212 Data & Knowledge Mining · University of Computer Studies, "
         "Yangon · 2025–2026 academic year",
         [("SOURCE NOTEBOOK", ""), ("data_mining_airline_updated.ipynb", "purp"),
          ("STREAMLIT + SCIKIT-LEARN", "ok")])

    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        st.markdown("**Notebook → app mapping**")
        md("""<div class="glass">""" + "".join(
            f"""<div class="mrow"><span class="m-name">{a}</span>
            <span class="track" style="display:none"></span>
            <span class="m-val">{b}</span></div>"""
            for a, b in [
                ("1. Load Dataset", "Sidebar uploader + bundled CSV"),
                ("2. Data Understanding", "Overview · inventory & audit"),
                ("3. Data Cleaning", "drop id/index · dedupe · median impute"),
                ("4. Exploratory Data Analysis", "EDA Explorer · 4 tabs"),
                ("5. K-Means (elbow, silhouette)", "Descriptive Mining · tab 1"),
                ("6. Apriori association rules", "Descriptive Mining · tab 2"),
                ("7. Preprocessing (encode, scale)", "Model Lab · pipeline chips"),
                ("8. LR / DT / RF", "Model Lab · section 02"),
                ("9. Evaluation (CM, CV, compare)", "Model Lab · sections 03–05"),
                ("10. RF feature importance", "Model Lab · section 06"),
                ("11. Final summary", "KPI hero + this page"),
            ]) + "</div>")
    with c2:
        st.markdown("**Dataset note**")
        md("""<div class="glass sidebar-note" style="font-size:.8rem">
          The bundled <code>data/airline.csv</code> is the <b>original project
          dataset</b> — 103,904 passenger survey records × 23 analysis
          columns (index/id dropped), with the 310 missing
          <i>Arrival Delay in Minutes</i> values median-imputed during
          cleaning. All charts, clusters, rules and models in this app run
          on this real data.<br><br>
          You can also upload your own CSV (same 23-column schema) in the
          sidebar — every chart, cluster, rule and model retrains on your
          data instantly.</div>""")

    st.markdown("**Tech stack**")
    md('<span class="chip">Python 3.12</span>'
       '<span class="chip purp">Streamlit</span>'
       '<span class="chip ok">scikit-learn</span>'
       '<span class="chip">mlxtend (Apriori)</span>'
       '<span class="chip">Plotly</span>'
       '<span class="chip purp">pandas / numpy</span>')


# ============================================================
# SIDEBAR & DISPATCH
# ============================================================
def sidebar():
    with st.sidebar:
        md("""<div style="display:flex;align-items:center;gap:10px;
        margin:6px 0 14px 0"><div style="font-size:1.5rem">✈</div>
        <div><div style="font-weight:800;font-size:1.02rem">AIRLINE
        PASSENGER SATISFACTION</div>
        <div style="color:#64748b;font-size:.7rem">Analysis Dashboard
        · IS-212</div></div></div>""")

        page = st.radio("Navigator", ["🏠 Overview", "📊 EDA Explorer",
                                      "🔍 Descriptive Mining", "🤖 Model Lab",
                                      "🎯 Predict", "ℹ About"],
                        label_visibility="collapsed")

        st.divider()
        st.markdown("**DATA**")
        up = st.file_uploader("Upload your airline CSV (23-col schema)",
                              type=["csv"])
        df = get_active_df()
        if up is not None:
            try:
                raw_up = pd.read_csv(up)
                missing = [c for c in REQUIRED if c not in raw_up.columns]
                if missing:
                    st.error(f"Missing required columns: {missing[:4]} …")
                else:
                    data_up, meta_up = clean_df(raw_up)
                    if st.session_state.get("data_source") != "upload" or \
                            fingerprint(data_up) != st.session_state.get("active_fp"):
                        set_active_df(data_up)
                        st.session_state["data_meta"] = meta_up
                        st.session_state["data_source"] = "upload"
                        st.cache_resource.clear()
                    df = get_active_df()
            except Exception as exc:  # noqa
                st.error(f"Could not read file: {exc}")

        src = st.session_state.get("data_source", "bundled")
        md(f'<span class="chip ok">SOURCE · '
           f'{"USER UPLOAD" if src == "upload" else "BUNDLED"}</span>'
           f'<span class="chip">{len(df):,} ROWS</span>')

        st.divider()
        st.markdown("**MODEL**")
        md('<span class="chip purp">LR · DT · RF</span>'
           '<span class="chip">TRAINS ON OPEN</span>')
        md("""<p class="sidebar-note" style="margin-top:10px">
        Heavy pipelines are cached — the first visit to a page computes and
        every later interaction is instant. Switching datasets clears the
        cache automatically.</p>""")
        return page


def main():
    page = sidebar()
    df = get_active_df()
    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 EDA Explorer":
        page_eda(df)
    elif page == "🔍 Descriptive Mining":
        page_mining(df)
    elif page == "🤖 Model Lab":
        page_lab(df)
    elif page == "🎯 Predict":
        page_predict(df)
    else:
        page_about(df)

    md("""<div style="margin-top:36px;padding:16px;text-align:center;
    color:#64748b;font-size:.75rem;border-top:1px solid rgba(148,163,184,.12)">
    About this project: IS-212 Data Mining · University of Computer Studies,
    Yangon · 2025–2026 — Airline Passenger Satisfaction Analysis
    </div>""")


if __name__ == "__main__":
    main()
