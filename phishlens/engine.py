"""PhishLens scanning engine.

The team's full model uses 50 features, but 29 of them come from the page itself (lines of code,
forms, images...). A typed URL does not give us those, so the scanner uses the saved URL-only
pipeline, which needs 21 features that can be computed from the URL string alone.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_model"
ASSET_DIR = ROOT / "assets"

# Class 0 is phishing, class 1 is legitimate (see model_metadata.json).
PHISHING, LEGITIMATE = 0, 1

RISK_BANDS = [  # (upper bound of phishing probability, level)
    (0.15, "Low"),
    (0.50, "Moderate"),
    (0.85, "High"),
    (1.01, "Critical"),
]

SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "password", "wallet", "banking", "paypal", "suspend", "unlock", "billing",
)
ABUSED_TLDS = {
    "zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link",
    "icu", "buzz", "cfd", "sbs", "rest", "lol",
}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"}


class InvalidURL(ValueError):
    """Raised when the input cannot be treated as a web address."""


# --------------------------------------------------------------------------- data classes
@dataclass
class Driver:
    label: str
    value: str
    weight: float          # signed logit contribution; > 0 pushes toward phishing
    toward: str            # "phishing" or "legitimate"


@dataclass
class Flag:
    level: str             # "alert", "warn" or "info"
    title: str
    detail: str


@dataclass
class Anatomy:
    scheme: str
    subdomain: str
    domain: str
    tld: str
    rest: str


@dataclass
class ScanResult:
    url: str
    host: str
    p_phishing: float
    verdict: str                       # "Phishing" or "Legitimate"
    risk: str
    notes: list[str] = field(default_factory=list)
    drivers: list[Driver] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    anatomy: Anatomy | None = None
    features: dict = field(default_factory=dict)
    reliability: str = "Normal"       # "Low" when the URL is outside the training pattern

    @property
    def is_phishing(self) -> bool:
        return self.verdict == "Phishing"

    @property
    def confidence(self) -> float:
        """Probability of the predicted class."""
        return self.p_phishing if self.is_phishing else 1 - self.p_phishing


# --------------------------------------------------------------------------- input handling
def normalize(raw: str) -> tuple[str, list[str]]:
    """Clean user input into an absolute http(s) URL. Returns (url, notes)."""
    notes: list[str] = []
    s = (raw or "").strip().strip("\"'<>")
    if not s:
        raise InvalidURL("Enter a URL to scan.")
    if re.search(r"\s", s):
        raise InvalidURL("URLs cannot contain spaces. Check for a stray space or line break.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", s):
        s = "http://" + s
        notes.append("No scheme was given, so the URL was scanned as http://. Add https:// if the link uses it.")
    scheme = s.split("://", 1)[0].lower()
    if scheme not in ("http", "https"):
        raise InvalidURL(f"Only http and https links can be scanned (got {scheme}://).")
    try:
        host = urlsplit(s).hostname
    except ValueError:
        raise InvalidURL("That address is malformed. Check the brackets and characters.") from None
    if not host:
        raise InvalidURL("No domain found. Enter a full address such as https://example.com/page.")
    if "." not in host and not re.fullmatch(r"[0-9a-fA-F:]+", host):
        raise InvalidURL(f"“{host}” does not look like a public domain. Include the full name, e.g. example.com.")
    return s, notes


def _split_parts(url: str) -> tuple[str, str, str, str]:
    """scheme-less string, netloc, host (no userinfo/port), tld."""
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", url)
    netloc = re.split(r"[/?#]", s, maxsplit=1)[0]
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    host = netloc.split(":")[0] if not netloc.startswith("[") else netloc
    tld = host.rsplit(".", 1)[-1].lower() if "." in host else ""
    return s, netloc, host, tld


# --------------------------------------------------------------------------- engine
class Engine:
    def __init__(self) -> None:
        self.pipeline = joblib.load(MODEL_DIR / "phishing_model_url_only_pipeline.joblib")
        self.columns: list[str] = json.load(open(MODEL_DIR / "feature_columns_url_only.json"))
        self.tld_prob: dict[str, float] = json.load(open(ASSET_DIR / "tld_prob.json"))
        self.char_prob: dict[str, float] = json.load(open(ASSET_DIR / "char_prob.json"))
        self.imputer = self.pipeline.named_steps["imputer"]
        self.scaler = self.pipeline.named_steps.get("scaler")
        self.clf = self.pipeline.steps[-1][1]
        self.phish_idx = list(self.clf.classes_).index(PHISHING)

    # ---- features
    def features(self, url: str) -> dict:
        s, netloc, host, tld = _split_parts(url)
        # The training data counts letters, digits and symbols after dropping the scheme and a
        # leading "www." — reproducing that keeps live scans comparable with the dataset.
        body = s[4:] if s.lower().startswith("www.") else s
        n = max(len(url), 1)
        letters = sum(c.isalpha() for c in body)
        digits = sum(c.isdigit() for c in body)
        equals, qmarks, amps = body.count("="), body.count("?"), body.count("&")
        other = sum(1 for c in body if not c.isalnum() and c not in "=?&")
        encoded = len(re.findall(r"%[0-9a-fA-F]{2}", url))
        dom = re.sub(r"^www\.", "", host, flags=re.I)
        runs = re.findall(r"[A-Za-z]+", dom)
        continuation = (max(map(len, runs)) if runs else 0) / max(1, len(dom.replace(".", "")))
        char_prob = float(np.mean([self.char_prob.get(c, 0.0) for c in url]))
        is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
        return {
            "URLLength": len(url),
            "DomainLength": len(netloc),
            "IsDomainIP": int(is_ip),
            "CharContinuationRate": continuation,
            "TLDLegitimateProb": self.tld_prob.get(tld, 0.0),
            "URLCharProb": char_prob,
            "TLDLength": len(tld),
            "NoOfSubDomain": max(host.count(".") - 1, 0),
            "HasObfuscation": int(encoded > 0),
            "NoOfObfuscatedChar": encoded,
            "ObfuscationRatio": encoded / n,
            "NoOfLettersInURL": letters,
            "LetterRatioInURL": letters / n,
            "NoOfDegitsInURL": digits,
            "DegitRatioInURL": digits / n,
            "NoOfEqualsInURL": equals,
            "NoOfQMarkInURL": qmarks,
            "NoOfAmpersandInURL": amps,
            "NoOfOtherSpecialCharsInURL": other,
            "SpacialCharRatioInURL": other / n,
            "IsHTTPS": int(url.lower().startswith("https://")),
        }

    # ---- scanning
    def scan(self, raw: str) -> ScanResult:
        url, notes = normalize(raw)
        feats = self.features(url)
        X = pd.DataFrame([feats])[self.columns].astype(float)
        p_phish = float(self.pipeline.predict_proba(X)[0, self.phish_idx])
        verdict = "Phishing" if p_phish >= 0.5 else "Legitimate"
        risk = next(level for bound, level in RISK_BANDS if p_phish < bound)
        s, _, host, _ = _split_parts(url)
        reliability = "Normal"
        deep = bool(re.sub(r"^[^/?#]*", "", s).strip("/"))
        if (deep or not host.lower().startswith("www.")) and not feats["IsDomainIP"]:
            reliability = "Low"
            notes.append(
                "Outside the training pattern: every legitimate URL in the training data was a "
                "bare https://www. homepage. Deep links and hosts without www tend to score as "
                "phishing, so treat this score with caution."
            )
        return ScanResult(
            url=url,
            host=host,
            p_phishing=p_phish,
            verdict=verdict,
            risk=risk,
            notes=notes,
            drivers=self._drivers(X, feats),
            flags=self._flags(url, host, feats),
            anatomy=_anatomy(url),
            features=feats,
            reliability=reliability,
        )

    # ---- explanations
    def _drivers(self, X: pd.DataFrame, feats: dict, top: int = 6) -> list[Driver]:
        """Per-feature contribution to the logistic-regression score (coefficient x scaled value)."""
        if not hasattr(self.clf, "coef_"):
            return []
        Z = self.imputer.transform(X)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        # coef_ is for class 1 (legitimate); flip so positive means "toward phishing".
        sign = -1.0 if self.phish_idx == 0 else 1.0
        contrib = sign * Z[0] * self.clf.coef_[0]
        order = np.argsort(-np.abs(contrib))
        out = []
        for i in order:
            if abs(contrib[i]) < 0.15 or len(out) >= top:
                continue
            name = self.columns[i]
            label, value = describe_feature(name, feats[name])
            out.append(Driver(label, value, float(contrib[i]), "phishing" if contrib[i] > 0 else "legitimate"))
        return out

    def _flags(self, url: str, host: str, feats: dict) -> list[Flag]:
        """Plain URL checks that sit beside the model. They are not model output."""
        flags: list[Flag] = []
        s, netloc, _, tld = _split_parts(url)
        low = url.lower()
        if feats["IsDomainIP"]:
            flags.append(Flag("alert", "IP address instead of a domain", "Legitimate sites almost always use a registered domain name."))
        if "@" in s.split("/", 1)[0]:
            flags.append(Flag("alert", "Login details embedded in the address", "Text before “@” is ignored by the browser, so the real site is whatever follows it."))
        if host.lower().startswith("xn--") or ".xn--" in host.lower():
            flags.append(Flag("alert", "Punycode domain", "The name uses an encoded form that can imitate look-alike characters."))
        if not feats["IsHTTPS"]:
            flags.append(Flag("warn", "Not using HTTPS", "The connection is unencrypted. Never enter passwords on a plain http page."))
        if feats["NoOfSubDomain"] >= 3:
            flags.append(Flag("warn", f"{feats['NoOfSubDomain']} subdomains", "Long subdomain chains are often used to bury the real domain."))
        if host.count("-") >= 2:
            flags.append(Flag("warn", "Several hyphens in the domain", "Hyphenated names such as secure-bank-login are common in look-alike domains."))
        hits = sorted({w for w in SUSPICIOUS_WORDS if w in low})
        if hits:
            flags.append(Flag("warn", "Credential-themed words", "Found: " + ", ".join(hits[:4]) + "."))
        if tld in ABUSED_TLDS:
            flags.append(Flag("warn", f"Frequently abused ending (.{tld})", "This ending is cheap or free, and common in throwaway sites."))
        if feats["URLLength"] > 90:
            flags.append(Flag("info", "Very long URL", f"{feats['URLLength']} characters. Long addresses can hide the destination."))
        if feats["HasObfuscation"]:
            flags.append(Flag("info", "Percent-encoded characters", f"{feats['NoOfObfuscatedChar']} encoded character(s) found in the address."))
        return flags


# --------------------------------------------------------------------------- presentation helpers
def _anatomy(url: str) -> Anatomy:
    scheme = url.split("://", 1)[0].lower() + "://"
    s, netloc, host, tld = _split_parts(url)
    rest = s[len(netloc):] if s.startswith(netloc) else ""
    labels = host.split(".")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) or len(labels) < 2:
        return Anatomy(scheme, "", host, "", rest)
    two_part = len(labels) >= 3 and len(tld) == 2 and labels[-2] in SECOND_LEVEL
    tld_labels = 2 if two_part else 1
    domain = labels[-tld_labels - 1]
    sub = ".".join(labels[: -tld_labels - 1])
    return Anatomy(scheme, sub + "." if sub else "", domain, "." + ".".join(labels[-tld_labels:]), rest)


def describe_feature(name: str, v: float) -> tuple[str, str]:
    pct = lambda x: f"{x * 100:.1f}%"
    table = {
        "NoOfLettersInURL": ("Letters in the address", f"{int(v)}"),
        "IsHTTPS": ("Connection", "HTTPS" if v else "Plain HTTP"),
        "DomainLength": ("Domain length", f"{int(v)} characters"),
        "NoOfOtherSpecialCharsInURL": ("Symbols (dots, dashes, slashes)", f"{int(v)}"),
        "NoOfDegitsInURL": ("Digits in the address", f"{int(v)}"),
        "DegitRatioInURL": ("Digit share of the address", pct(v)),
        "LetterRatioInURL": ("Letter share of the address", pct(v)),
        "SpacialCharRatioInURL": ("Symbol share of the address", pct(v)),
        "URLCharProb": ("Character pattern", "typical" if v >= 0.055 else "unusual"),
        "CharContinuationRate": ("Longest unbroken word in the domain", pct(v)),
        "TLDLength": ("Ending length", f"{int(v)} characters"),
        "NoOfSubDomain": ("Subdomains", f"{int(v)}"),
        "NoOfEqualsInURL": ("Equals signs", f"{int(v)}"),
        "NoOfQMarkInURL": ("Question marks", f"{int(v)}"),
        "NoOfAmpersandInURL": ("Ampersands", f"{int(v)}"),
        "IsDomainIP": ("Host is an IP address", "yes" if v else "no"),
        "HasObfuscation": ("Percent-encoded characters", "yes" if v else "none"),
        "NoOfObfuscatedChar": ("Encoded characters", f"{int(v)}"),
        "ObfuscationRatio": ("Encoded share of the address", pct(v)),
        "URLLength": ("Address length", f"{int(v)} characters"),
        "TLDLegitimateProb": ("Ending reputation in training data", pct(v)),
    }
    return table.get(name, (name, f"{v:g}"))
