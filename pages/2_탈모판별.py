"""탈모 판별 알고리즘 · 나도 해 보기"""
import plotly.graph_objects as go
import streamlit as st

import common as c

st.set_page_config(page_title="탈모 판별", page_icon="🔍", layout="wide")
c.page_header("2 · 탈모 판별 알고리즘", "습관·건강·식습관 11개 요인으로 탈모 여부를 맞혀 보면?")

main = c.load_main()
ev = c.evaluate_model()
cols = list(c.FACTORS)

st.markdown("""
**방법 — 나이브 베이즈(직접 구현)**
1. 999명을 무작위로 **학습 70% / 검증 30%**로 나눈다.
2. 학습용에서 "탈모 있음/없음 그룹에 각 요인 값이 얼마나 자주 나오는지" 센다.
3. 검증용 사람마다 두 그룹일 확률을 곱해 비교하고, 50% 이상이면 '탈모'로 판별한다.
4. 1~3을 **30번** 반복해 평균 정확도를 본다. 비교 기준은 '학습용에서 더 많은 쪽으로 모두 찍기'.
""")

m1, m2, m3 = st.columns(3)
m1.metric("판별 모델 평균 정확도", f"{ev['model'].mean():.1f}%")
m2.metric("모두 한쪽으로 찍기", f"{ev['baseline'].mean():.1f}%")
m3.metric("차이", f"{ev['model'].mean() - ev['baseline'].mean():+.1f}%p")

fig = go.Figure()
fig.add_scatter(x=ev["seed"] + 1, y=ev["model"], mode="lines+markers", name="판별 모델",
                line=dict(color=c.COLORS["blue"], width=2), marker=dict(size=8),
                hovertemplate="%{x}번째: %{y:.1f}%<extra>판별 모델</extra>")
fig.add_scatter(x=ev["seed"] + 1, y=ev["baseline"], mode="lines+markers", name="찍기",
                line=dict(color=c.COLORS["orange"], width=2, dash="dot"), marker=dict(size=8),
                hovertemplate="%{x}번째: %{y:.1f}%<extra>찍기</extra>")
fig.update_xaxes(title_text="반복 회차")
fig.update_yaxes(title_text="검증 정확도 (%)", range=[35, 65])
st.plotly_chart(c.style(fig, 340), width="stretch")

st.warning("**해석:** 모델이 찍기보다 낫지 않아요. 이 설문의 요인들은 탈모 여부와 거의 관계가 없게 섞여 있어서, "
           "**요인만 보고 탈모를 판별하는 건 동전 던지기와 같아요.** 판별은 두피 사진(5페이지) 같은 직접 증거가 필요해요.")

st.divider()
st.subheader("나도 해 보기")
st.caption("값을 고르면 모델이 계산한 '탈모 쪽 확률'과, 어떤 요인이 그쪽으로 기울였는지 보여 줘요. 위 결과처럼 **믿을 만한 점수가 아니에요.**")

model = c.NaiveBayes(cols).fit(main)
with st.form("me"):
    grid = st.columns(3)
    choice = {}
    for i, col in enumerate(cols):
        name, group = c.FACTORS[col]
        options = sorted(main[col].unique())
        choice[col] = grid[i % 3].selectbox(f"{name} · {group}", options, key=col)
    sent = st.form_submit_button("판별해 보기", type="primary")

if sent:
    import pandas as pd
    p = model.proba(pd.DataFrame([choice]))[0]
    st.metric("모델이 본 '탈모 쪽' 확률", f"{p * 100:.0f}%", help="학습 데이터 전체(999명)로 학습한 결과")
    contrib = model.contributions(choice)
    contrib["dir"] = contrib["log_ratio"].map(lambda v: "탈모 쪽" if v > 0 else "탈모 아닌 쪽")
    contrib["name"] = contrib["factor"] + " = " + contrib["value"].astype(str)
    fig = go.Figure(go.Bar(
        x=contrib["log_ratio"], y=contrib["name"], orientation="h",
        marker_color=[c.COLORS["blue"] if v > 0 else c.COLORS["gray"] for v in contrib["log_ratio"]],
        hovertemplate="%{y}<br>기울기 %{x:+.2f}<extra></extra>"))
    fig.update_xaxes(title_text="← 탈모 아닌 쪽 · 탈모 쪽 → (로그 확률비)")
    st.plotly_chart(c.style(fig, 420, legend=False), width="stretch")
