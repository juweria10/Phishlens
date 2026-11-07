"""PhishLens visual system. Tokens are documented in design/design-system.md."""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root{
  --canvas:#F1F3EF; --paper:#FAFBF8; --ink:#0E1A19; --body:#2B3836; --muted:#5E6B68;
  --rule:#D3D9D2; --rule-soft:#E3E7E1;
  --night:#0C1716; --night-2:#142322; --night-3:#1F3331; --on-night:#EAF0ED; --on-night-muted:#97AAA5;
  --brand:#0F6B63; --brand-soft:#DCEBE8;
  --danger:#D2402A; --danger-soft:#F6DED8; --danger-night:#FF7A63;
  --warn:#A9690F; --warn-soft:#F5E8CF;
  --safe:#2A8559; --safe-soft:#DAEDE2; --safe-night:#59D49A;
  --serif:'Newsreader','Iowan Old Style',Georgia,serif;
  --sans:'IBM Plex Sans',-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;
}

/* ---------- Streamlit chrome ---------- */
html,body,[data-testid="stAppViewContainer"],.stApp{background:var(--canvas);color:var(--body);font-family:var(--sans);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;height:0;}
.block-container{max-width:1120px;padding:2.4rem 2.4rem 5rem;}
h1,h2,h3{font-family:var(--serif);color:var(--ink);font-weight:400;letter-spacing:-.01em;}
p,li,label,span,div{font-family:var(--sans);}
a{color:var(--brand);}
:focus-visible{outline:2px solid var(--brand);outline-offset:2px;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:var(--night);border-right:0;min-width:248px;}
[data-testid="stSidebar"] *{color:var(--on-night);}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1.4rem;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;}
[data-testid="stSidebar"] label[data-baseweb="radio"]{
  padding:.55rem .9rem;margin:0;border-left:2px solid transparent;border-radius:0 6px 6px 0;width:100%;cursor:pointer;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child{display:none;}
[data-testid="stSidebar"] label[data-baseweb="radio"] p{font-size:.95rem;font-weight:500;color:var(--on-night-muted);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:var(--night-2);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:var(--night-2);border-left-color:var(--safe-night);}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p{color:var(--on-night);}

.pl-brand{display:flex;align-items:center;gap:.6rem;padding:0 .9rem 1.3rem;}
.pl-brand .mark{width:26px;height:26px;border:2px solid var(--safe-night);border-radius:50%;position:relative;flex:none;}
.pl-brand .mark:after{content:"";position:absolute;width:2px;height:10px;background:var(--safe-night);right:-4px;bottom:-6px;transform:rotate(-45deg);}
.pl-brand .name{font-family:var(--serif);font-size:1.45rem;color:var(--on-night);letter-spacing:-.01em;}
.pl-side-note{margin:1.6rem .9rem 0;padding-top:1rem;border-top:1px solid var(--night-3);font-size:.8rem;line-height:1.5;color:var(--on-night-muted);}
.pl-side-note b{color:var(--on-night);font-weight:500;}

/* ---------- page heads ---------- */
.pl-h1{font-family:var(--serif);font-size:2.6rem;line-height:1.08;letter-spacing:-.02em;color:var(--ink);margin:0 0 .5rem;}
.pl-lede{font-size:1.05rem;line-height:1.55;color:var(--muted);max-width:60ch;margin:0 0 1.6rem;}
.pl-h2{font-family:var(--serif);font-size:1.6rem;line-height:1.2;color:var(--ink);margin:2.6rem 0 .3rem;letter-spacing:-.01em;}
.pl-sub{font-size:.95rem;color:var(--muted);max-width:68ch;margin:0 0 1.1rem;line-height:1.55;}
.pl-rule{border:0;border-top:1px solid var(--rule);margin:2rem 0;}

/* ---------- inputs ---------- */
[data-testid="stForm"]{border:0;padding:0;background:transparent;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{
  font-family:var(--mono);font-size:.98rem;background:var(--paper);color:var(--ink);
  border:1px solid var(--rule);border-radius:8px;padding:.8rem 1rem;
}
[data-testid="stTextInput"] input{height:3.1rem;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px var(--brand-soft);}
[data-testid="stTextInput"] > div > div,[data-testid="stTextArea"] > div > div{background:transparent;border:0;box-shadow:none;}
.stButton > button,[data-testid="stFormSubmitButton"] > button,[data-testid="stDownloadButton"] > button{
  font-family:var(--sans);font-weight:500;font-size:.95rem;border-radius:8px;min-height:3.1rem;
  border:1px solid var(--rule);background:var(--paper);color:var(--ink);
}
.stButton > button:hover,[data-testid="stDownloadButton"] > button:hover{border-color:var(--ink);color:var(--ink);}
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--brand)!important;border-color:var(--brand)!important;color:#fff!important;
}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  background:#0B534D!important;border-color:#0B534D!important;color:#fff!important;
}
.pl-try{font-size:.85rem;color:var(--muted);margin:.9rem 0 .35rem;}
.pl-try-row .stButton > button{min-height:2.3rem;font-family:var(--mono);font-size:.8rem;font-weight:400;padding:.2rem .7rem;}

/* tabs */
[data-baseweb="tab-list"]{gap:1.4rem;border-bottom:1px solid var(--rule);}
[data-baseweb="tab"]{padding:.5rem 0;background:transparent;font-weight:500;color:var(--muted);}
[data-baseweb="tab"][aria-selected="true"]{color:var(--ink);}
[data-baseweb="tab-highlight"]{background:var(--brand);height:2px;}
[data-baseweb="tab-border"]{display:none;}

/* ---------- result panel ---------- */
.pl-result{background:var(--night);color:var(--on-night);border-radius:12px;padding:2rem 2.2rem 1.6rem;margin-top:1.6rem;}
.pl-result.is-phish{box-shadow:inset 4px 0 0 var(--danger-night);}
.pl-result.is-safe{box-shadow:inset 4px 0 0 var(--safe-night);}
.pl-res-top{display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem;flex-wrap:wrap;}
.pl-verdict{font-family:var(--serif);font-size:3rem;line-height:1;letter-spacing:-.02em;margin:0;}
.is-phish .pl-verdict{color:var(--danger-night);} .is-safe .pl-verdict{color:var(--safe-night);}
.pl-verdict-sub{color:var(--on-night-muted);font-size:.95rem;margin-top:.55rem;}
.pl-score{text-align:right;}
.pl-score .n{font-family:var(--mono);font-size:2.2rem;line-height:1;color:var(--on-night);}
.pl-score .l{font-size:.82rem;color:var(--on-night-muted);margin-top:.3rem;}
.pl-chips{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem;}
.pl-chip{font-size:.78rem;font-weight:500;padding:.2rem .65rem;border-radius:99px;background:var(--night-3);color:var(--on-night);}
.pl-chip.low{background:#4A3A12;color:#F2CE7A;}

.pl-scale{margin:1.8rem 0 .4rem;}
.pl-scale .track{position:relative;display:grid;grid-template-columns:15fr 35fr 35fr 15fr;gap:3px;height:10px;}
.pl-scale .seg{background:var(--night-3);}
.pl-scale .seg:nth-child(1){background:#2C6B50;border-radius:5px 0 0 5px;}
.pl-scale .seg:nth-child(2){background:#8C7A2B;}
.pl-scale .seg:nth-child(3){background:#B25A2B;}
.pl-scale .seg:nth-child(4){background:#B5382A;border-radius:0 5px 5px 0;}
.pl-scale .pin{position:absolute;top:-7px;width:3px;height:24px;background:var(--on-night);border-radius:2px;transform:translateX(-1px);}
.pl-scale .labels{display:grid;grid-template-columns:15fr 35fr 35fr 15fr;gap:3px;margin-top:.5rem;font-size:.76rem;color:var(--on-night-muted);}

.pl-anat{margin-top:1.6rem;}
.pl-anat .cap,.pl-cols .cap{font-size:.82rem;color:var(--on-night-muted);margin-bottom:.5rem;}
.pl-anat .row{font-family:var(--mono);font-size:1.02rem;display:flex;flex-wrap:wrap;gap:0;align-items:flex-start;word-break:break-all;}
.pl-anat .part{display:flex;flex-direction:column;padding-right:0;}
.pl-anat .txt{padding:.35rem .1rem .25rem;border-bottom:2px solid var(--night-3);}
.pl-anat .tag{font-family:var(--sans);font-size:.7rem;color:var(--on-night-muted);padding-top:.3rem;}
.pl-anat .part.scheme .txt{color:var(--on-night-muted);}
.pl-anat .part.domain .txt{color:var(--on-night);border-bottom-color:var(--on-night);font-weight:500;}
.pl-anat .part.warn .txt{border-bottom-color:var(--danger-night);}
.pl-anat .part.ok .txt{border-bottom-color:var(--safe-night);}
.pl-anat .part.sub .txt,.pl-anat .part.tld .txt,.pl-anat .part.rest .txt{color:#C4D2CE;}

.pl-cols{display:grid;grid-template-columns:1.15fr 1fr;gap:2.4rem;margin-top:2rem;padding-top:1.6rem;border-top:1px solid var(--night-3);}
@media (max-width:820px){.pl-cols{grid-template-columns:1fr;gap:1.6rem;}.pl-verdict{font-size:2.3rem;}.pl-score{text-align:left;}.block-container{padding:1.4rem 1.1rem 4rem;}.pl-h1{font-size:2.1rem;}}
.pl-driver{display:grid;grid-template-columns:minmax(0,1fr) 92px;gap:.9rem;align-items:center;padding:.5rem 0;border-bottom:1px solid var(--night-3);}
.pl-driver:last-child{border-bottom:0;}
.pl-driver .t{font-size:.92rem;color:var(--on-night);line-height:1.35;}
.pl-driver .v{font-size:.8rem;color:var(--on-night-muted);font-family:var(--mono);}
.pl-driver .bar{height:6px;background:var(--night-3);border-radius:3px;position:relative;overflow:hidden;}
.pl-driver .bar i{position:absolute;top:0;bottom:0;border-radius:3px;}
.pl-driver .bar i.phishing{background:var(--danger-night);} .pl-driver .bar i.legitimate{background:var(--safe-night);}
.pl-driver .dir{font-size:.72rem;color:var(--on-night-muted);margin-top:.25rem;}
.pl-flag{display:grid;grid-template-columns:10px 1fr;gap:.7rem;padding:.55rem 0;border-bottom:1px solid var(--night-3);}
.pl-flag:last-child{border-bottom:0;}
.pl-flag .dot{width:8px;height:8px;border-radius:50%;margin-top:.45rem;background:var(--on-night-muted);}
.pl-flag.alert .dot{background:var(--danger-night);} .pl-flag.warn .dot{background:#E9B04A;}
.pl-flag .t{font-size:.92rem;color:var(--on-night);} .pl-flag .d{font-size:.82rem;color:var(--on-night-muted);line-height:1.45;margin-top:.1rem;}
.pl-none{font-size:.9rem;color:var(--on-night-muted);line-height:1.5;}
.pl-foot{margin-top:1.6rem;font-size:.8rem;color:var(--on-night-muted);line-height:1.5;}
.pl-note{margin-top:1.4rem;padding:.9rem 1.1rem;border-left:3px solid #E9B04A;background:#2A2410;color:#F2DCA8;font-size:.9rem;line-height:1.5;border-radius:0 8px 8px 0;}

.pl-error{margin-top:1.2rem;padding:.9rem 1.1rem;border-left:3px solid var(--danger);background:var(--danger-soft);color:#6B2216;border-radius:0 8px 8px 0;font-size:.95rem;}
.pl-empty{margin-top:2.4rem;padding:2rem 0;border-top:1px solid var(--rule);}
.pl-empty .t{font-family:var(--serif);font-size:1.4rem;color:var(--ink);margin-bottom:.4rem;}
.pl-empty .d{color:var(--muted);max-width:56ch;line-height:1.55;}

/* ---------- overview ---------- */
.pl-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);margin:1rem 0 0;}
@media (max-width:820px){.pl-stats{grid-template-columns:repeat(2,minmax(0,1fr));}}
.pl-stat{padding:1.1rem 1.2rem 1.2rem 0;} .pl-stat + .pl-stat{padding-left:1.2rem;border-left:1px solid var(--rule-soft);}
.pl-stat .n{font-family:var(--serif);font-size:2.6rem;line-height:1;color:var(--ink);letter-spacing:-.02em;}
.pl-stat .n.danger{color:var(--danger);} .pl-stat .n.safe{color:var(--safe);}
.pl-stat .l{font-size:.85rem;color:var(--muted);margin-top:.45rem;line-height:1.35;}

.pl-split{display:flex;height:14px;border-radius:7px;overflow:hidden;gap:2px;margin:.9rem 0 .5rem;}
.pl-split i{display:block;height:100%;} .pl-split .p{background:var(--danger);} .pl-split .l{background:var(--safe);}
.pl-legend{display:flex;gap:1.4rem;font-size:.85rem;color:var(--muted);}
.pl-legend b{color:var(--ink);font-weight:500;}
.pl-legend .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.4rem;}

.pl-days{display:flex;align-items:flex-end;gap:6px;height:120px;margin-top:.8rem;border-bottom:1px solid var(--rule);}
.pl-day{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%;min-width:0;}
.pl-day i{display:block;width:100%;} .pl-day .p{background:var(--danger);} .pl-day .l{background:var(--safe);}
.pl-day-labels{display:flex;gap:6px;margin-top:.35rem;}
.pl-day-labels span{flex:1;text-align:center;font-size:.72rem;color:var(--muted);min-width:0;}

.pl-two{display:grid;grid-template-columns:1fr 1fr;gap:3rem;}
@media (max-width:820px){.pl-two{grid-template-columns:1fr;gap:1rem;}}

/* ---------- tables ---------- */
.pl-table{width:100%;border-collapse:collapse;font-size:.9rem;}
.pl-table th{text-align:left;font-weight:500;font-size:.8rem;color:var(--muted);padding:.55rem .8rem .55rem 0;border-bottom:1px solid var(--ink);}
.pl-table td{padding:.7rem .8rem .7rem 0;border-bottom:1px solid var(--rule-soft);vertical-align:middle;color:var(--body);}
.pl-table td.url{font-family:var(--mono);font-size:.84rem;word-break:break-all;max-width:420px;color:var(--ink);}
.pl-table td.num{font-family:var(--mono);font-size:.85rem;}
.pl-table .when{color:var(--muted);font-size:.82rem;white-space:nowrap;}
.pl-tag{display:inline-block;font-size:.78rem;font-weight:500;padding:.15rem .6rem;border-radius:99px;white-space:nowrap;}
.pl-tag.phish{background:var(--danger-soft);color:#9B2A18;} .pl-tag.safe{background:var(--safe-soft);color:#1D6342;}
.pl-tag.low{background:var(--warn-soft);color:#7A4A06;margin-left:.35rem;}
.pl-mini{display:inline-block;width:90px;height:6px;background:var(--rule-soft);border-radius:3px;vertical-align:middle;margin-right:.55rem;position:relative;overflow:hidden;}
.pl-mini i{position:absolute;left:0;top:0;bottom:0;background:var(--danger);border-radius:3px;}
.pl-table-wrap{overflow-x:auto;}

/* ---------- model page ---------- */
.pl-track{position:relative;height:8px;background:var(--rule-soft);border-radius:4px;min-width:140px;}
.pl-track i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--brand);}
.pl-track.dim i{background:#7FA6A1;}
.pl-cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:3px;font-size:.85rem;max-width:340px;}
.pl-cm div{padding:.7rem .6rem;text-align:center;}
.pl-cm .h{color:var(--muted);font-size:.76rem;padding:.2rem .4rem;} .pl-cm .side{text-align:right;color:var(--muted);font-size:.76rem;align-self:center;padding:0 .5rem 0 0;}
.pl-cm .ok{background:var(--brand-soft);color:var(--brand);font-family:var(--mono);font-size:1.1rem;}
.pl-cm .bad{background:var(--danger-soft);color:#9B2A18;font-family:var(--mono);font-size:1.1rem;}
.pl-cm .zero{background:var(--rule-soft);color:var(--muted);font-family:var(--mono);font-size:1.1rem;}
.pl-cm-title{font-weight:500;color:var(--ink);margin-bottom:.5rem;}
.pl-callout{border-left:3px solid var(--brand);background:var(--paper);padding:1rem 1.3rem;margin:1.2rem 0;border-radius:0 8px 8px 0;font-size:.95rem;line-height:1.6;color:var(--body);max-width:78ch;}
.pl-callout b{color:var(--ink);font-weight:500;}
.pl-bar-row{display:grid;grid-template-columns:210px 1fr 64px;gap:1rem;align-items:center;padding:.4rem 0;font-size:.88rem;}
@media (max-width:820px){.pl-bar-row{grid-template-columns:120px 1fr 52px;}}
.pl-bar-row .v{font-family:var(--mono);font-size:.82rem;text-align:right;color:var(--ink);}
.pl-bar-row .k{color:var(--body);}

@media (prefers-reduced-motion:no-preference){.pl-result{animation:plin .35s ease-out;}}
@keyframes plin{from{opacity:.0;transform:translateY(6px);}to{opacity:1;transform:none;}}
"""

CSS += """
[data-testid="stFormSubmitButton"] > button,[data-testid="stFormSubmitButton"]{width:100%;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div{background:var(--paper);border-color:var(--rule);border-radius:8px;}
[data-testid="stRadio"] [role="radiogroup"] p{font-size:.92rem;}
"""
CSS += """
[data-testid="stSidebar"] {
    min-width: 248px;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
"""