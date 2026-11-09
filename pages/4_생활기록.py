"""한 사람의 400일 생활 기록"""
import plotly.graph_objects as go
import streamlit as st

import common as c

st.set_page_config(page_title="생활 기록", page_icon="📅", layout="wide")
c.page_header("4 · 한 사람의 400일 생활 기록", "2021-01-01 ~ 2022-02-04, 매일 적은 탈모량과 생활 습관")

d = c.load_daily()
VARS = {
    "stress_level": "스트레스 (0 낮음~3 매우 높음)",
    "pressure_level": "압박감 (0~3)",
    "coffee": "커피 (잔)",
    "hair_grease": "두피 유분 (1~5)",
    "brain_hours": "공부·머리 쓴 시간 (시간)",
    "stay_up_late": "밤샘 (시간)",
    "libido": "성욕 (0~5)",
}

corr = d[list(VARS) + ["hair_loss_level"]].corr()["hair_loss_level"].drop("hair_loss_level")
cdf = corr.rename("r").reset_index().rename(columns={"index": "var"})
cdf["name"] = cdf["var"].map(VARS)
cdf = cdf.sort_values("r", ascending=False)
cdf["label"] = cdf["r"].map(lambda v: f"{v:.2f}")

st.subheader("탈모량과 함께 움직이는 정도 (상관계수 r)")
st.caption("탈모량은 적음 0 · 보통 1 · 많음 2 · 매우 많음 3으로 바꿨어요. r이 1에 가까울수록 같이 늘어나요.")
st.plotly_chart(c.hbar(cdf, x="r", y="name", text="label", x_title="상관계수 r"), width="stretch")

st.subheader("시간에 따라 보기")
var = st.selectbox("탈모량과 나란히 볼 항목", list(VARS), format_func=VARS.get, index=2)
w = d.set_index("date")[["hair_loss_level", var]].resample("W").mean().reset_index()
for col, title, color in [("hair_loss_level", "탈모량 (주간 평균, 0~3)", c.COLORS["blue"]),
                          (var, f"{VARS[var]} (주간 평균)", c.COLORS["orange"])]:
    fig = go.Figure(go.Scatter(x=w["date"], y=w[col], mode="lines", line=dict(color=color, width=2),
                               hovertemplate="%{x|%Y-%m-%d} 주: %{y:.2f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), hovermode="x unified")
    st.plotly_chart(c.style(fig, 240, legend=False), width="stretch")

st.subheader("샴푸·수영·머리 감기")
cols = st.columns(3)
for box, col, name in zip(cols, ["shampoo", "swimming", "hair_washing"], ["샴푸 브랜드", "수영한 날", "머리 감은 날"]):
    g = d.groupby(col)["hair_loss_level"].agg(["mean", "size"]).reset_index()
    g[col] = g[col].astype(str).replace({"True": "예", "False": "아니오"})
    g["label"] = g.apply(lambda x: f"{x['mean']:.2f} · {x['size']}일", axis=1)
    fig = c.hbar(g.sort_values("mean", ascending=False), x="mean", y=col, text="label", x_title="평균 탈모량 (0~3)", height=220)
    fig.update_layout(title=dict(text=name, font=dict(size=14)))
    box.plotly_chart(fig, width="stretch")

st.warning("**한 사람의 기록이에요.** 시험 기간처럼 스트레스·커피·밤샘이 한꺼번에 늘어나는 시기가 겹쳐 있어서, "
           "어느 하나가 탈모의 '원인'이라고 말할 수 없어요. 내 생활 기록을 이렇게 쌓아 보는 방법으로 참고하세요.")
