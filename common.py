"""데이터 불러오기 · 조 색 · 차트 함수 (디자인 담당)

모든 페이지는 여기 함수만 써서 데이터를 읽고 차트를 그린다.
색은 아래 COLORS / GROUP_COLORS 이름으로만 쓴다 (새 #코드 만들지 않기).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

# ── 조 색 (dataviz 기본 팔레트, 순서 고정) ──────────────────────────
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "gray": "#8a8984",
    "grid": "rgba(128,128,128,0.18)",
}
GROUP_COLORS = {"건강 상태": COLORS["blue"], "식습관·영양": COLORS["orange"], "생활 습관": COLORS["aqua"]}
LABEL_COLORS = {"정상": COLORS["aqua"], "비듬성": COLORS["yellow"], "염증성": COLORS["orange"],
                "탈모": COLORS["blue"], "(정답 비공개)": COLORS["gray"]}
PERIOD_COLORS = {"코로나 전": COLORS["blue"], "코로나 후": COLORS["orange"]}
SEQ_BLUE = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]  # 순차(크기) 색: 연한 파랑 → 진한 파랑

# 설문 요인: 열 이름 → (한글 이름, 묶음)
FACTORS = {
    "genetics": ("가족력(유전)", "건강 상태"),
    "hormonal": ("호르몬 변화", "건강 상태"),
    "medical": ("앓는 질환", "건강 상태"),
    "medication": ("복용 약·치료", "건강 상태"),
    "weight_loss": ("급격한 체중 감소", "건강 상태"),
    "age_group": ("나이대", "건강 상태"),
    "nutrition": ("부족한 영양소", "식습관·영양"),
    "stress": ("스트레스", "생활 습관"),
    "poor_care": ("잘못된 모발 관리", "생활 습관"),
    "environment": ("환경 요인(오염 등)", "생활 습관"),
    "smoking": ("흡연", "생활 습관"),
}

DISCLAIMER = "⚠️ 이 대시보드는 공개 데이터를 탐색한 학습용 결과예요. 의학적 진단이 아니며, 탈모가 걱정되면 피부과 전문의와 상담하세요."


# ── 데이터 불러오기 (정제본만 읽는다) ──────────────────────────────
@st.cache_data
def load_main() -> pd.DataFrame:
    return pd.read_csv(DATA / "clean.csv")


@st.cache_data
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DATA / "clean_daily.csv", parse_dates=["date"])


@st.cache_data
def load_scalp() -> pd.DataFrame:
    return pd.read_csv(DATA / "clean_scalp.csv", parse_dates=["taken_at"], dtype={"subject": str})


@st.cache_data
def load_tweets() -> pd.DataFrame:
    return pd.read_csv(DATA / "clean_tweets.csv", parse_dates=["created_at"])


@st.cache_data
def load_clean_log() -> dict:
    import json
    return json.loads((DATA / "clean_log.json").read_text(encoding="utf-8"))


# ── 분석 함수 ───────────────────────────────────────────────────
def factor_rates(df: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    """요인 값별 탈모 비율(%)과 인원."""
    g = df.groupby(col)["hair_loss"].agg(rate="mean", n="size").reset_index()
    g["rate"] = g["rate"] * 100
    return g[g["n"] >= min_n].sort_values("rate", ascending=False)


def factor_gaps(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """요인마다 '가장 높은 값 - 가장 낮은 값' 탈모 비율 차이(%p). 인원 min_n명 미만 값은 뺀다."""
    rows = []
    for col, (name, group) in FACTORS.items():
        r = factor_rates(df, col, min_n)
        hi, lo = r.iloc[0], r.iloc[-1]
        rows.append({"col": col, "factor": name, "group": group, "gap": hi["rate"] - lo["rate"],
                     "high": f"{hi[col]} {hi['rate']:.0f}%", "low": f"{lo[col]} {lo['rate']:.0f}%"})
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


class NaiveBayes:
    """탈모 판별 알고리즘: 범주형 나이브 베이즈 (직접 구현, 라플라스 보정).

    각 요인이 '탈모 있음/없음' 그룹에서 얼마나 자주 나오는지 세어 확률을 곱한다.
    """

    def __init__(self, cols: list[str], alpha: float = 1.0):
        self.cols, self.alpha = cols, alpha

    def fit(self, df: pd.DataFrame) -> "NaiveBayes":
        y = df["hair_loss"]
        self.prior = {c: math.log((y == c).mean()) for c in (0, 1)}
        self.table = {}
        for col in self.cols:
            values = df[col].unique()
            k = len(values)
            for c in (0, 1):
                sub = df.loc[y == c, col].value_counts()
                total = sub.sum()
                for v in values:
                    self.table[(col, v, c)] = math.log((sub.get(v, 0) + self.alpha) / (total + self.alpha * k))
        return self

    def proba(self, df: pd.DataFrame) -> np.ndarray:
        out = []
        for _, row in df.iterrows():
            s = {c: self.prior[c] + sum(self.table.get((col, row[col], c), 0.0) for col in self.cols) for c in (0, 1)}
            m = max(s.values())
            p1 = math.exp(s[1] - m) / (math.exp(s[0] - m) + math.exp(s[1] - m))
            out.append(p1)
        return np.array(out)

    def contributions(self, row: dict) -> pd.DataFrame:
        """요인별로 '탈모 쪽'으로 기우는 정도 (로그 확률비, +면 탈모 쪽)."""
        rows = []
        for col in self.cols:
            v = row[col]
            lr = self.table.get((col, v, 1), 0.0) - self.table.get((col, v, 0), 0.0)
            rows.append({"factor": FACTORS[col][0], "value": v, "log_ratio": lr})
        return pd.DataFrame(rows).sort_values("log_ratio")


@st.cache_data
def evaluate_model(n_rounds: int = 30, test_frac: float = 0.3) -> pd.DataFrame:
    """무작위로 나눠 학습/검증을 n_rounds번 반복한 정확도."""
    df = load_main()
    cols = list(FACTORS)
    rows = []
    for seed in range(n_rounds):
        test = df.sample(frac=test_frac, random_state=seed)
        train = df.drop(test.index)
        model = NaiveBayes(cols).fit(train)
        pred = (model.proba(test) >= 0.5).astype(int)
        majority = int(train["hair_loss"].mean() >= 0.5)
        rows.append({"seed": seed,
                     "model": (pred == test["hair_loss"].to_numpy()).mean() * 100,
                     "baseline": (test["hair_loss"] == majority).mean() * 100})
    return pd.DataFrame(rows)


TOKEN = re.compile(r"(?u)\b\w\w+\b")  # sklearn CountVectorizer 기본 규칙과 같다
STOP = set("""the and to of a in is it i you that my for me be with have on not this so are but
was just your at do all what can like he she they we an or if as about they his her from by
when no up out get one how who will their more people them don know really there has been
would because im it's i'm dont its am than then him our had did were even being too very""".split())


def tokens(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def word_counts(texts: pd.Series) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(tokens(t))
    return c


def cosine(a: Counter, b: Counter) -> float:
    """원 노트북과 같은 방식: 기간별 트윗을 한 문서로 이어 붙여 단어 수 벡터의 코사인 유사도."""
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ── 화면 공통 ───────────────────────────────────────────────────
def page_header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)
    st.info(DISCLAIMER)


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        font=dict(size=13), showlegend=legend, bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def hbar(df: pd.DataFrame, x: str, y: str, color: str | None = None, color_map: dict | None = None,
         text: str | None = None, x_title: str = "", hover: str | None = None, height: int | None = None) -> go.Figure:
    """가로 막대 (정렬된 순서 그대로, 위가 첫 행)."""
    d = df.iloc[::-1]
    fig = go.Figure()
    groups = [(None, d)] if color is None else list(d.groupby(color, sort=False))
    for name, part in groups:
        fig.add_bar(
            x=part[x], y=part[y], orientation="h", name=str(name) if name is not None else "",
            marker_color=(color_map or {}).get(name, COLORS["blue"]),
            text=part[text] if text else None, textposition="outside", cliponaxis=False,
            hovertext=part[hover] if hover else None,
            hovertemplate=("%{y}: %{x:.1f}<br>%{hovertext}<extra></extra>" if hover else "%{y}: %{x:.1f}<extra></extra>"),
        )
    fig.update_yaxes(categoryorder="array", categoryarray=list(d[y]))
    fig.update_xaxes(title_text=x_title)
    return style(fig, height or max(260, 34 * len(d) + 80), legend=color is not None)
