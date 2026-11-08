"""PhishLens: AI-based phishing URL detection. Run with:  streamlit run app.py"""
from __future__ import annotations

import warnings
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from phishlens import components as ui
from phishlens import data, history
from phishlens.engine import Engine, InvalidURL, describe_feature
from phishlens.theme import CSS

warnings.filterwarnings("ignore")

PAGES = ["Scan", "Overview", "Model performance", "History"]
EXAMPLES = [
    "https://www.openautoalliance.net",
    "http://www.oosp.pplink.club",
    "http://secure-paypal.verify-account.login-update.xyz/signin?id=8812",
]
MAX_BATCH = 200

st.set_option("client.toolbarMode", "minimal")

st.set_page_config(page_title="PhishLens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stToolbar"] {
    visibility: visible;
}

[data-testid="stToolbar"] button[aria-label*="Fork"],
[data-testid="stToolbar"] a[aria-label*="Fork"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model…")
def get_engine() -> Engine:
    return Engine()


@st.cache_data
def get_meta() -> tuple[dict, dict]:
    return data.metadata(), data.run_summary()


engine = get_engine()
meta, summary = get_meta()


# --------------------------------------------------------------------------- state helpers
def go_to(page: str) -> None:
    st.session_state["page"] = page


def load_example(url: str) -> None:
    st.session_state["url_input"] = url
    st.session_state["pending_scan"] = True


def run_scan(raw: str) -> None:
    try:
        result = engine.scan(raw)
    except InvalidURL as exc:
        st.session_state["scan_error"], st.session_state["last"] = str(exc), None
        return
    history.add(result, "single")
    st.session_state["scan_error"], st.session_state["last"] = None, result


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(ui.brand(), unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    url_only = data.feature_settings()
    acc = float(url_only[(url_only["setting"].str.startswith("URL features")) & (url_only["model"] == "Logistic Regression")]["accuracy"].iloc[0])
    st.markdown(ui.side_note("Logistic regression", len(engine.columns), acc), unsafe_allow_html=True)


# --------------------------------------------------------------------------- pages
def page_scan() -> None:
    st.markdown(ui.page_head("Check a link before you open it", "Paste a web address. PhishLens scores how closely its structure matches known phishing URLs, and shows what drove the score."), unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["One URL", "Several URLs"])

    with tab_single:
        with st.form("scan_form", clear_on_submit=False):
            c1, c2 = st.columns([6, 1.4])
            raw = c1.text_input("URL to scan", key="url_input", placeholder="https://example.com/account/verify", label_visibility="collapsed")
            submitted = c2.form_submit_button("Scan URL", type="primary")
        if submitted:
            run_scan(raw)
        elif st.session_state.pop("pending_scan", False):
            run_scan(st.session_state.get("url_input", ""))

        st.markdown('<div class="pl-try">Or load an example</div>', unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for col, url in zip(cols, EXAMPLES):
            label = (urlsplit(url).hostname or url).removeprefix("www.")
            label = label if len(label) <= 30 else label[:29] + "…"
            col.button(label, key=f"ex_{url}", on_click=load_example, args=(url,))

        if st.session_state.get("scan_error"):
            st.markdown(ui.error(st.session_state["scan_error"]), unsafe_allow_html=True)
        elif st.session_state.get("last") is not None:
            st.markdown(ui.result_panel(st.session_state["last"]), unsafe_allow_html=True)
        else:
            st.markdown(ui.empty_scan(), unsafe_allow_html=True)

    with tab_batch:
        st.markdown('<p class="pl-sub" style="margin-top:1rem">One URL per line, up to 200. Every scan is added to History.</p>', unsafe_allow_html=True)
        text = st.text_area("URLs", key="batch_input", height=170, label_visibility="collapsed", placeholder="https://www.example.com\nhttp://login-secure.example.xyz/verify")
        if st.button("Scan all", type="primary", key="batch_go"):
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:MAX_BATCH]
            rows, skipped = [], []
            for ln in lines:
                try:
                    r = engine.scan(ln)
                except InvalidURL as exc:
                    skipped.append((ln, str(exc)))
                    continue
                history.add(r, "batch")
                rows.append({"url": r.url, "verdict": r.verdict, "p_phishing": r.p_phishing, "risk": r.risk, "reliability": r.reliability, "flags": len(r.flags)})
            st.session_state["batch"] = (pd.DataFrame(rows), skipped)
        batch = st.session_state.get("batch")
        if batch is not None:
            df, skipped = batch
            if df.empty and not skipped:
                st.markdown(ui.error("Add at least one URL, one per line."), unsafe_allow_html=True)
            if not df.empty:
                n_p = int((df["verdict"] == "Phishing").sum())
                st.markdown(ui.stats([(str(len(df)), "URLs scanned", ""), (str(n_p), "likely phishing", "danger"), (str(len(df) - n_p), "likely legitimate", "safe")]), unsafe_allow_html=True)
                st.markdown(ui.scan_table(df.assign(time=pd.Timestamp.now(tz="UTC")), show_time=False), unsafe_allow_html=True)
                st.download_button("Download results (CSV)", df.to_csv(index=False).encode(), "phishlens_batch.csv", "text/csv")
            if skipped:
                with st.expander(f"{len(skipped)} line(s) skipped"):
                    for ln, why in skipped:
                        st.markdown(f"`{ln[:80]}`: {why}")


def page_overview() -> None:
    st.markdown(ui.page_head("Overview", "Activity from your scans and a snapshot of the model and data behind them."), unsafe_allow_html=True)
    hist = history.load()
    total = len(hist)
    n_p = int((hist["verdict"] == "Phishing").sum()) if total else 0
    low = int((hist["reliability"] == "Low").sum()) if total else 0
    st.markdown(ui.stats([
        (f"{total:,}", "URLs scanned", ""),
        (f"{n_p:,}", "flagged as phishing", "danger"),
        (f"{total - n_p:,}", "judged legitimate", "safe"),
        (f"{low:,}", "scans outside the training pattern", ""),
    ]), unsafe_allow_html=True)

    if total == 0:
        st.markdown(ui.empty_scan().replace("Nothing scanned yet", "No scan activity yet"), unsafe_allow_html=True)
        st.button("Scan a URL", on_click=go_to, args=("Scan",), type="primary", key="ov_scan")
    else:
        a, b = st.columns([3, 2], gap="large")
        with a:
            st.markdown(ui.section("Scans per day", "Last 14 days. Red is phishing, green is legitimate."), unsafe_allow_html=True)
            st.markdown(ui.daily_bars(hist), unsafe_allow_html=True)
        with b:
            st.markdown(ui.section("Verdict split"), unsafe_allow_html=True)
            st.markdown(ui.split_bar(n_p, total - n_p, "phishing", "legitimate"), unsafe_allow_html=True)
        st.markdown(ui.section("Recent scans"), unsafe_allow_html=True)
        st.markdown(ui.scan_table(hist.head(6)), unsafe_allow_html=True)
        st.button("See full history", on_click=go_to, args=("History",), key="ov_hist")

    ds = summary["data"]
    st.markdown(ui.section("The model behind the scanner", "Trained on the UCI PhiUSIIL phishing URL dataset after cleaning and removing duplicate URLs."), unsafe_allow_html=True)
    st.markdown(ui.stats([
        (f"{ds['rows_after_url_dedup']:,}", "unique URLs after cleaning", ""),
        (f"{ds['train_rows']:,} / {ds['test_rows']:,}", "train / test split", ""),
        (str(len(engine.columns)), "features read from the URL string", ""),
        (f"{acc * 100:.1f}%", "test accuracy, URL features only", "safe"),
    ]), unsafe_allow_html=True)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    cc = ds["class_counts"]
    st.markdown(ui.split_bar(cc["Phishing(0)"], cc["Legitimate(1)"], "phishing URLs", "legitimate URLs"), unsafe_allow_html=True)

    st.markdown(ui.section("Exploratory analysis", "Figures from the data cleaning and EDA stage."), unsafe_allow_html=True)
    options = [f for f in data.EDA_TITLES if (data.EDA_IMAGES / f).exists()]
    pick = st.selectbox("Figure", options, format_func=lambda f: data.EDA_TITLES[f], label_visibility="collapsed", key="eda_pick")
    if pick:
        st.image(str(data.EDA_IMAGES / pick))


def page_models() -> None:
    st.markdown(ui.page_head("Model performance", "How Logistic Regression, Decision Tree and Random Forest compare, and what each result does and does not tell us."), unsafe_allow_html=True)
    st.markdown(ui.callout(
        "<b>Read the 100% scores with care.</b> With all 50 features the models see page content such as lines of code and "
        "external references, plus <b>URLSimilarityIndex</b>, which separates the classes almost on its own. A pasted link has none of that, "
        "so the scanner uses the 21 URL-only features instead. Scores there are lower and more honest."
    ), unsafe_allow_html=True)

    st.markdown(ui.section("Scores by feature setting", "Tuned models on the held-out test set. Bars start at 97% so the gaps are visible."), unsafe_allow_html=True)
    st.markdown(ui.zoom_table(data.feature_settings()), unsafe_allow_html=True)

    st.markdown(ui.section("Where the errors fall", "Confusion matrices on the test set, using tuned models. Phishing is the positive class."), unsafe_allow_html=True)
    t_url, t_all = st.tabs(["URL features only (used by the scanner)", "All 50 features"])
    for tab, flag in ((t_url, True), (t_all, False)):
        with tab:
            cms = data.confusion_matrices(flag)
            cols = st.columns(len(cms))
            for col, (name, (tp, fn, fp, tn)) in zip(cols, cms.items()):
                col.markdown(ui.confusion(name, tp, fn, fp, tn), unsafe_allow_html=True)

    st.markdown(ui.section("What the models rely on"), unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Random Forest, all features** (importance)")
        rows = [(describe_feature(n, 0)[0] if n in engine.columns else n, v) for n, v in data.rf_importance(10)]
        st.markdown(ui.bar_rows(rows, "{:.3f}"), unsafe_allow_html=True)
    with b:
        st.markdown("**Logistic Regression, URL only** (size of weight)")
        coefs = sorted(zip(engine.columns, abs(engine.clf.coef_[0])), key=lambda x: -x[1])[:8]
        st.markdown(ui.bar_rows([(describe_feature(n, 0)[0], float(v)) for n, v in coefs], "{:.1f}"), unsafe_allow_html=True)
    st.markdown(ui.callout(
        "The URL-only model leans on <b>letters in the address</b>, <b>HTTPS</b> and <b>domain length</b>. "
        "In the training data every legitimate URL is a bare <b>https://www.</b> homepage, so the model also treats deep links and "
        "hosts without www as suspicious. That is a limit of the dataset, and the scanner flags such inputs as low reliability."
    ), unsafe_allow_html=True)

    st.markdown(ui.section("Training plots", "The charts produced by the training notebook."), unsafe_allow_html=True)
    graphs = [g for g in data.GRAPH_TITLES if (data.GRAPHS / g).exists()]
    pick = st.selectbox("Plot", graphs, format_func=lambda g: data.GRAPH_TITLES[g], label_visibility="collapsed", key="graph_pick")
    if pick:
        st.image(str(data.GRAPHS / pick))


def page_history() -> None:
    st.markdown(ui.page_head("Scan history", "Every URL you have scanned in this session, newest first. It is private to you and clears when you refresh the page, so download the CSV to keep a copy."), unsafe_allow_html=True)
    hist = history.load()
    if hist.empty:
        st.markdown(ui.empty_scan().replace("Nothing scanned yet", "No scans in history"), unsafe_allow_html=True)
        st.button("Scan a URL", on_click=go_to, args=("Scan",), type="primary", key="hist_scan")
        return
    f1, f2 = st.columns([2, 3])
    verdict = f1.radio("Verdict", ["All", "Phishing", "Legitimate"], horizontal=True, label_visibility="collapsed", key="hist_verdict")
    query = f2.text_input("Search", placeholder="Filter by text in the URL", label_visibility="collapsed", key="hist_query")
    view = hist
    if verdict != "All":
        view = view[view["verdict"] == verdict]
    if query:
        view = view[view["url"].str.contains(query, case=False, regex=False)]
    st.markdown(f'<p class="pl-sub" style="margin-top:.8rem">{len(view):,} of {len(hist):,} scans</p>', unsafe_allow_html=True)
    if view.empty:
        st.markdown(ui.error("No scans match these filters."), unsafe_allow_html=True)
    else:
        st.markdown(ui.scan_table(view.head(100)), unsafe_allow_html=True)
        if len(view) > 100:
            st.caption("Showing the newest 100. Download the CSV for everything.")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    d1, d2, _ = st.columns([1.3, 1.3, 3])
    d1.download_button("Download CSV", hist.to_csv(index=False).encode(), "phishlens_history.csv", "text/csv")
    if not st.session_state.get("confirm_clear"):
        if d2.button("Clear history", key="clear_ask"):
            st.session_state["confirm_clear"] = True
            st.rerun()
    else:
        st.markdown(ui.error("This deletes every scan from this session and cannot be undone."), unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.3, 1.3, 3])
        if c1.button("Delete all scans", type="primary", key="clear_yes"):
            history.clear()
            st.session_state["confirm_clear"] = False
            st.rerun()
        if c2.button("Cancel", key="clear_no"):
            st.session_state["confirm_clear"] = False
            st.rerun()


{"Scan": page_scan, "Overview": page_overview, "Model performance": page_models, "History": page_history}[st.session_state["page"]]()
