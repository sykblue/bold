"""코로나 전후 '낮은 자존감' 트윗 비교 (노트북 cosine-similarities-for-during-and-before-covid 재현)"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

st.set_page_config(page_title="코로나 전후", page_icon="🦠", layout="wide")
c.page_header("6 · 코로나 전후 마음 상태", "'low self esteem'이 들어간 트윗을 코로나 전(2017~18)과 후(2021~22)로 비교")

st.caption("탈모와 직접 관련된 데이터는 아니에요. 스트레스·자존감 같은 마음 상태가 탈모 이야기와 자주 함께 나와서, "
           "노트북 `adeyinkaajagunna/cosine-similarities-for-during-and-before-covid`의 방법을 그대로 재현한 보조 페이지예요.")

t = c.load_tweets()
pre, post = t[t["period"] == "코로나 전"], t[t["period"] == "코로나 후"]


@st.cache_data
def counts():
    return c.word_counts(pre["text"]), c.word_counts(post["text"])


a, b = counts()
sim = c.cosine(a, b)

m = st.columns(3)
m[0].metric("코로나 전 트윗", f"{len(pre):,}건")
m[1].metric("코로나 후 트윗", f"{len(post):,}건")
m[2].metric("두 시기 단어 사용 코사인 유사도", f"{sim:.3f}", help="1이면 단어 쓰는 비율이 완전히 같음")
st.caption("방법(노트북과 동일): 시기별 트윗을 하나의 긴 글로 이어 붙이고, 두 글의 단어 수 벡터 사이 코사인 값을 구했어요. "
           "원 노트북의 결과값은 저장돼 있지 않아 직접 비교하지는 못했어요.")

st.subheader("감정 비율")
sent = t.groupby("period")["sentiment"].value_counts(normalize=True).mul(100).rename("share").reset_index()
fig = go.Figure()
for label, color in [("부정", c.COLORS["orange"]), ("중립", c.COLORS["gray"]), ("긍정", c.COLORS["aqua"])]:
    p = sent[sent["sentiment"] == label]
    fig.add_bar(y=p["period"], x=p["share"], name=label, orientation="h", marker_color=color,
                text=p["share"].map(lambda v: f"{v:.0f}%"), textposition="inside",
                hovertemplate="%{y} · " + label + ": %{x:.1f}%<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_xaxes(range=[0, 100], title_text="비율 (%)")
st.plotly_chart(c.style(fig, 220), width="stretch")
st.caption("감정 값은 데이터 원작자가 붙인 것이에요.")

st.subheader("시기별로 많이 늘어난 단어")
st.caption("자주 쓰는 일상어와 'low·self·esteem'은 뺐어요. 1만 단어당 몇 번 나왔는지로 비교해요.")
skip = c.STOP | {"low", "self", "esteem", "lowselfesteem", "https", "co", "amp", "rt"}
ta, tb = sum(a.values()), sum(b.values())
words = pd.DataFrame({"pre": pd.Series(a), "post": pd.Series(b)}).fillna(0)
words = words[~words.index.isin(skip) & (words.sum(axis=1) >= 30) & ~words.index.str.isdigit()]
words["pre"] = words["pre"] / ta * 1e4
words["post"] = words["post"] / tb * 1e4
words["diff"] = words["post"] - words["pre"]
n = st.slider("단어 수", 5, 20, 10)
col1, col2 = st.columns(2)
for box, d, title, color in [(col1, words.nlargest(n, "diff"), "코로나 후에 늘어난 단어", c.COLORS["orange"]),
                             (col2, words.nsmallest(n, "diff").assign(diff=lambda x: -x["diff"]), "코로나 전에 더 많던 단어", c.COLORS["blue"])]:
    d = d.reset_index(names="word")
    d["label"] = d["diff"].map(lambda v: f"+{v:.1f}")
    fig = c.hbar(d, x="diff", y="word", text="label", x_title="1만 단어당 차이")
    fig.update_traces(marker_color=color)
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    box.plotly_chart(fig, width="stretch")

with st.expander("트윗 예시 보기 (원문 그대로, 거친 표현이 있을 수 있어요)"):
    period = st.radio("시기", ["코로나 전", "코로나 후"], horizontal=True)
    st.dataframe(t[t["period"] == period][["created_at", "sentiment", "text"]].sample(15, random_state=1),
                 hide_index=True, width="stretch")
