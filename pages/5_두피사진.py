"""두피 사진 데이터 (대회 2022-ai-tp-18011886) — 파일 목록 기반"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

st.set_page_config(page_title="두피 사진", page_icon="📷", layout="wide")
c.page_header("5 · 두피 사진 데이터", "대회 '두피 이미지를 통한 두피 질환 및 탈모 진단 예측'의 사진 12,729장 목록")

s = c.load_scalp()
train = s[s["split"] == "train"]
test = s[s["split"] == "test"]
POS = {"TH": "TH", "LH": "LH", "RH": "RH", "BH": "BH"}

m = st.columns(4)
m[0].metric("학습 사진 (정답 있음)", f"{len(train):,}장")
m[1].metric("시험 사진 (정답 비공개)", f"{len(test):,}장")
m[2].metric("전체 용량", f"{s['mb'].sum() / 1000:.1f} GB")
m[3].metric("촬영 기간", f"{s['taken_at'].min():%Y.%m} ~ {s['taken_at'].max():%Y.%m}")

st.subheader("정답(두피 상태)별 사진 수")
lab = train["label"].value_counts().rename_axis("label").reset_index(name="n")
lab["share"] = lab["n"] / lab["n"].sum() * 100
lab["text"] = lab.apply(lambda x: f"{x['n']:,}장 · {x['share']:.0f}%", axis=1)
fig = c.hbar(lab, x="n", y="label", text="text", x_title="사진 수")
fig.update_traces(marker_color=[c.LABEL_COLORS[l] for l in lab["label"].iloc[::-1]])
st.plotly_chart(fig, width="stretch")
st.caption("네 가지가 크게 치우치지 않아(23~30%) 사진으로 판별 모델을 학습하기에 알맞아요.")

st.subheader("촬영 위치 코드별 구성")
st.caption("파일 이름 끝의 TH·LH·RH·BH는 촬영 위치 코드예요. 정수리(Top)·왼쪽(Left)·오른쪽(Right)·뒤(Back)로 **추정**되지만 대회 설명에서 확인하지 못했어요.")
pos = train.groupby(["pos", "label"]).size().reset_index(name="n")
pos["share"] = pos["n"] / pos.groupby("pos")["n"].transform("sum") * 100
fig = go.Figure()
for label in ["정상", "비듬성", "염증성", "탈모"]:
    p = pos[pos["label"] == label]
    fig.add_bar(y=p["pos"], x=p["share"], name=label, orientation="h", marker_color=c.LABEL_COLORS[label],
                customdata=p["n"], hovertemplate="%{y} · " + label + ": %{x:.0f}% (%{customdata}장)<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_traces(marker_line_color="rgba(0,0,0,0)", marker_line_width=2)
fig.update_xaxes(title_text="위치 안에서의 비율 (%)", range=[0, 100])
fig.update_yaxes(categoryorder="array", categoryarray=["BH", "RH", "LH", "TH"])
st.plotly_chart(c.style(fig, 300), width="stretch")
bh = pos[(pos["pos"] == "BH") & (pos["label"] == "탈모")]["share"].iloc[0]
th = pos[(pos["pos"] == "TH") & (pos["label"] == "탈모")]["share"].iloc[0]
st.caption(f"'탈모' 사진은 BH 위치에서 {bh:.0f}%로 적고, TH에서 {th:.0f}%예요. 탈모는 뒤통수보다 정수리·옆머리에서 먼저 보인다는 상식과 맞는 방향이에요.")

st.subheader("주별 촬영 사진 수")
wk = s.groupby(["split", pd.Grouper(key="taken_at", freq="W")]).size().reset_index(name="n")
fig = go.Figure()
for split, color, name in [("train", c.COLORS["blue"], "학습"), ("test", c.COLORS["orange"], "시험")]:
    p = wk[wk["split"] == split]
    fig.add_bar(x=p["taken_at"], y=p["n"], name=name, marker_color=color,
                hovertemplate="%{x|%Y-%m-%d} 주: %{y}장<extra>" + name + "</extra>")
fig.update_layout(barmode="stack")
st.plotly_chart(c.style(fig, 280), width="stretch")

st.subheader("파일 이름 읽는 법")
st.code("0013_A2LEBJJDE00060O_1602578303771_3_TH.jpg\n"
        "│    │                 │             │ └ 촬영 위치 코드\n"
        "│    │                 │             └── 촬영 순번\n"
        "│    │                 └──────────────── 촬영 시각 (1/1000초, 2020-10-13 17:38)\n"
        "│    └────────────────────────────────── 기기 번호\n"
        "└─────────────────────────────────────── 묶음 번호", language=None)

st.info("**다음 단계:** 대회 페이지에서 규칙에 동의하면 사진 원본을 받을 수 있어요. "
        "그다음 학습 사진 11,417장으로 4가지 두피 상태를 분류하는 이미지 모델을 만들면 '사진 기반 탈모 판별'이 돼요.")
