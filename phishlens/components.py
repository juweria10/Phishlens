"""HTML builders for PhishLens. Every user-controlled string is escaped; URLs are never links."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from .engine import RISK_BANDS, ScanResult

esc = html.escape


def h(s: str) -> str:
    """Collapse indentation and blank lines so Streamlit's markdown never reads HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


# --------------------------------------------------------------------------- shell
def brand() -> str:
    return h('<div class="pl-brand"><span class="mark"></span><span class="name">PhishLens</span></div>')


def side_note(model_name: str, n_features: int, accuracy: float) -> str:
    return h(f"""
    <div class="pl-side-note"><b>Scanner model</b><br>{esc(model_name)} on {n_features} URL features.
    Test accuracy {accuracy * 100:.1f}%.<br><br>It reads the address only. It never opens the page.</div>""")


def page_head(title: str, lede: str) -> str:
    return h(f'<h1 class="pl-h1">{esc(title)}</h1><p class="pl-lede">{esc(lede)}</p>')


def section(title: str, sub: str = "") -> str:
    s = f'<h2 class="pl-h2">{esc(title)}</h2>'
    if sub:
        s += f'<p class="pl-sub">{esc(sub)}</p>'
    return h(s)


def error(msg: str) -> str:
    return h(f'<div class="pl-error">{esc(msg)}</div>')


# --------------------------------------------------------------------------- result
def risk_scale(p: float) -> str:
    pos = min(max(p, 0.0), 1.0) * 100
    names = [name for _, name in RISK_BANDS]
    labels = "".join(f"<span>{n}</span>" for n in names)
    return h(f"""
    <div class="pl-scale"><div class="track"><i class="seg"></i><i class="seg"></i><i class="seg"></i><i class="seg"></i>
    <span class="pin" style="left:{pos:.1f}%"></span></div><div class="labels">{labels}</div></div>""")


def anatomy(r: ScanResult) -> str:
    a = r.anatomy
    https = a.scheme.startswith("https")
    parts = [
        ("scheme ok" if https else "scheme warn", a.scheme, "secure" if https else "not encrypted"),
    ]
    if a.subdomain:
        parts.append(("sub", a.subdomain, "subdomain"))
    parts.append(("domain", a.domain, "registered name"))
    if a.tld:
        parts.append(("tld", a.tld, "ending"))
    if a.rest:
        parts.append(("rest", a.rest, "path and query"))
    cells = "".join(
        f'<span class="part {cls}"><span class="txt">{esc(txt)}</span><span class="tag">{esc(tag)}</span></span>'
        for cls, txt, tag in parts
    )
    return h(f'<div class="pl-anat"><div class="cap">How the address breaks down</div><div class="row">{cells}</div></div>')


def drivers(r: ScanResult) -> str:
    if not r.drivers:
        return '<div class="pl-none">No single signal stood out.</div>'
    top = max(abs(d.weight) for d in r.drivers) or 1
    rows = []
    for d in r.drivers:
        w = abs(d.weight) / top * 100
        rows.append(f"""
        <div class="pl-driver"><div><div class="t">{esc(d.label)} <span class="v">{esc(d.value)}</span></div>
        <div class="dir">{'Raises' if d.toward == 'phishing' else 'Lowers'} phishing risk</div></div>
        <div class="bar"><i class="{d.toward}" style="left:0;width:{w:.0f}%"></i></div></div>""")
    return h("".join(rows))


def flags(r: ScanResult) -> str:
    if not r.flags:
        return '<div class="pl-none">None of the structural checks fired: HTTPS, no IP host, no look-alike patterns, no credential words.</div>'
    return h("".join(
        f'<div class="pl-flag {f.level}"><span class="dot"></span><div><div class="t">{esc(f.title)}</div>'
        f'<div class="d">{esc(f.detail)}</div></div></div>'
        for f in r.flags
    ))


def result_panel(r: ScanResult) -> str:
    phish = r.is_phishing
    cls = "is-phish" if phish else "is-safe"
    title = "Likely phishing" if phish else "Likely legitimate"
    sub = (
        "Do not enter passwords or payment details on this site."
        if phish else "Nothing in the address looks like phishing. This does not prove the site is safe."
    )
    chips = f'<span class="pl-chip">{esc(r.risk)} risk</span>'
    chips += f'<span class="pl-chip">{len(r.flags)} URL check{"s" if len(r.flags) != 1 else ""} flagged</span>'
    if r.reliability == "Low":
        chips += '<span class="pl-chip low">Low reliability</span>'
    notes = "".join(f'<div class="pl-note">{esc(n)}</div>' for n in r.notes)
    return h(f"""
    <section class="pl-result {cls}" aria-live="polite">
      <div class="pl-res-top">
        <div><p class="pl-verdict">{title}</p><div class="pl-verdict-sub">{sub}</div><div class="pl-chips">{chips}</div></div>
        <div class="pl-score"><div class="n">{r.p_phishing * 100:.0f}%</div><div class="l">phishing probability</div></div>
      </div>
      {risk_scale(r.p_phishing)}
      {anatomy(r)}
      {notes}
      <div class="pl-cols">
        <div><div class="cap">What moved the score</div>{drivers(r)}</div>
        <div><div class="cap">URL checks, separate from the model</div>{flags(r)}</div>
      </div>
      <div class="pl-foot">Scored by a logistic regression trained on URL structure only. The page itself was not loaded.</div>
    </section>""")


def empty_scan() -> str:
    return h("""
    <div class="pl-empty"><div class="t">Nothing scanned yet</div>
    <div class="d">Paste a link above, or load one of the examples. Links are shown as plain text here, so you cannot click through to a suspicious site by accident.</div></div>""")


# --------------------------------------------------------------------------- tables
def _ago(ts: pd.Timestamp) -> str:
    secs = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return ts.strftime("%d %b %Y")


def scan_table(df: pd.DataFrame, show_time: bool = True) -> str:
    if df.empty:
        return ""
    rows = []
    for r in df.itertuples():
        phish = r.verdict == "Phishing"
        tag = f'<span class="pl-tag {"phish" if phish else "safe"}">{"Phishing" if phish else "Legitimate"}</span>'
        if getattr(r, "reliability", "Normal") == "Low":
            tag += '<span class="pl-tag low">low reliability</span>'
        when = f'<td class="when">{esc(_ago(r.time))}</td>' if show_time else ""
        rows.append(
            f'<tr>{when}<td class="url">{esc(r.url)}</td><td>{tag}</td>'
            f'<td class="num"><span class="pl-mini"><i style="width:{r.p_phishing * 100:.0f}%"></i></span>{r.p_phishing * 100:.0f}%</td>'
            f'<td>{esc(r.risk)}</td></tr>'
        )
    head = "<th>When</th>" if show_time else ""
    return h(
        f'<div class="pl-table-wrap"><table class="pl-table"><thead><tr>{head}<th>URL</th><th>Verdict</th>'
        f'<th>Phishing probability</th><th>Risk</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


# --------------------------------------------------------------------------- overview
def stats(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="pl-stat"><div class="n {cls}">{esc(n)}</div><div class="l">{esc(label)}</div></div>' for n, label, cls in items)
    return h(f'<div class="pl-stats">{cells}</div>')


def split_bar(a: int, b: int, a_label: str, b_label: str) -> str:
    total = max(a + b, 1)
    return h(f"""
    <div class="pl-split" role="img" aria-label="{a} {esc(a_label)}, {b} {esc(b_label)}"><i class="p" style="width:{a / total * 100:.1f}%"></i><i class="l" style="width:{b / total * 100:.1f}%"></i></div>
    <div class="pl-legend"><span><span class="sw" style="background:var(--danger)"></span><b>{a:,}</b> {esc(a_label)}</span>
    <span><span class="sw" style="background:var(--safe)"></span><b>{b:,}</b> {esc(b_label)}</span></div>""")


def daily_bars(df: pd.DataFrame, days: int = 14) -> str:
    end = pd.Timestamp.now(tz="UTC").normalize()
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    d = df.copy()
    if not d.empty:
        d["day"] = d["time"].dt.normalize()
    cols, labels = [], []
    counts = []
    for day in idx:
        sub = d[d["day"] == day] if not d.empty else d
        p = int((sub["verdict"] == "Phishing").sum()) if not d.empty else 0
        counts.append((p, (len(sub) - p) if not d.empty else 0))
    peak = max([p + l for p, l in counts] + [1])
    for day, (p, l) in zip(idx, counts):
        cols.append(
            f'<div class="pl-day" title="{day:%d %b}: {p} phishing, {l} legitimate">'
            f'<i class="p" style="height:{p / peak * 100:.0f}%"></i><i class="l" style="height:{l / peak * 100:.0f}%"></i></div>'
        )
        labels.append(f"<span>{day:%d}</span>")
    return h(f'<div class="pl-days" role="img" aria-label="Scans per day, last {days} days">{"".join(cols)}</div><div class="pl-day-labels">{"".join(labels)}</div>')


# --------------------------------------------------------------------------- model page
def bar_rows(rows: list[tuple[str, float]], fmt: str = "{:.3f}", scale_max: float | None = None) -> str:
    peak = scale_max or max(v for _, v in rows) or 1
    out = "".join(
        f'<div class="pl-bar-row"><div class="k">{esc(k)}</div><div class="pl-track"><i style="width:{max(v, 0) / peak * 100:.1f}%"></i></div>'
        f'<div class="v">{fmt.format(v)}</div></div>'
        for k, v in rows
    )
    return h(out)


def zoom_table(df: pd.DataFrame, lo: float = 0.97) -> str:
    """Rows of model scores drawn on an axis that starts at `lo`, so small gaps are visible."""
    head = "<th>Setting</th><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th>"
    body = []
    for r in df.itertuples():
        cells = ""
        for v in (r.accuracy, r.f1, r.precision, r.recall):
            w = max(0.0, (v - lo) / (1 - lo)) * 100
            cells += f'<td><div class="pl-track dim"><i style="width:{w:.0f}%"></i></div><div class="num" style="font-family:var(--mono);font-size:.78rem;margin-top:.25rem">{v * 100:.2f}%</div></td>'
        body.append(f"<tr><td>{esc(r.setting)}</td><td>{esc(r.model)}</td>{cells}</tr>")
    return h(f'<div class="pl-table-wrap"><table class="pl-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def confusion(title: str, tp: int, fn: int, fp: int, tn: int) -> str:
    """Rows are the true class, columns the prediction. Phishing is the positive class."""
    def cell(v, good):
        cls = "zero" if v == 0 else ("ok" if good else "bad")
        return f'<div class="{cls}">{v:,}</div>'
    return h(f"""
    <div><div class="pl-cm-title">{esc(title)}</div><div class="pl-cm">
    <div></div><div class="h">Predicted phishing</div><div class="h">Predicted legitimate</div>
    <div class="side">Actually phishing</div>{cell(tp, True)}{cell(fn, False)}
    <div class="side">Actually legitimate</div>{cell(fp, False)}{cell(tn, True)}</div></div>""")


def callout(text: str) -> str:
    return h(f'<div class="pl-callout">{text}</div>')  # callers pass trusted, pre-escaped markup
