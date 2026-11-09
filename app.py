"""홈: 결정 한 줄 · 결론 3줄 · 대표 차트 (PO)

실행: streamlit run app.py
"""
import streamlit as st

import common as c

st.set_page_config(page_title="탈모 판별 대시보드", page_icon="🧑‍🦲", layout="wide")

main = c.load_main()
daily = c.load_daily()
scalp = c.load_scalp()
gaps = c.factor_gaps(main)
ev = c.evaluate_model()
corr = daily.drop(columns=["date", "hair_loss_ko", "school", "shampoo", "dandruff"]).corr(numeric_only=True)["hair_loss_level"]
train = scalp[scalp["split"] == "train"]
bald_share = (train["label"] == "탈모").mean() * 100

c.page_header("탈모 판별 대시보드", "Kaggle 공개 데이터 5개 · 두피 사진 12,729장 목록 · 설문 999명 · 생활 기록 400일 · 트윗 13,628건")

# ── 결정 한 줄 ──
st.success("**결정:** 탈모 판별은 **설문 요인(습관·건강·식습관)이 아니라 '두피 사진'**으로 하고, "
           "생활 요인은 **내 생활 기록으로 추적하는 용도**로 쓴다.")

# ── 결론 3줄 ──
st.subheader("결론 3줄")
top = gaps.iloc[0]
st.markdown(
    f"1. **설문 요인만으로는 판별이 안 된다.** 999명 설문에서 요인 11개 중 차이가 가장 큰 '{top['factor']}'도 "
    f"탈모 비율 차이가 **{top['gap']:.0f}%p**뿐이고, 판별 모델 정확도는 **{ev['model'].mean():.0f}%**로 "
    f"한쪽으로 찍는 방식(**{ev['baseline'].mean():.0f}%**)과 같다.\n"
    f"2. **개인 생활 기록에서는 신호가 뚜렷하다.** 한 사람의 400일 기록에서 탈모량은 스트레스(r={corr['stress_level']:.2f}), "
    f"커피(r={corr['coffee']:.2f}), 두피 유분(r={corr['hair_grease']:.2f})과 함께 움직였다. 다만 1명의 기록이고 원인이라고 단정할 수 없다.\n"
    f"3. **판별의 근거는 사진이다.** 대회 학습 사진 {len(train):,}장 중 '탈모' 정답이 **{bald_share:.0f}%**로, "
    f"정상·비듬성·염증성과 함께 4가지 정답이 붙어 있다. 사진 원본은 대회 규칙 동의 후 받을 수 있다."
)

# ── 대표 차트 ──
st.subheader("대표 차트 · 요인별로 탈모 비율이 얼마나 갈리나")
st.caption("요인 안에서 탈모 비율이 가장 높은 값과 가장 낮은 값의 차이(%p). 30명 미만인 값은 뺐어요. "
           "차이가 클수록 판별에 쓸모가 있는 요인이에요. 전체 탈모 비율은 "
           f"{main['hair_loss'].mean() * 100:.0f}%예요.")
gaps["label"] = gaps["gap"].map(lambda v: f"{v:.1f}%p")
gaps["detail"] = "높음: " + gaps["high"] + " · 낮음: " + gaps["low"]
fig = c.hbar(gaps, x="gap", y="factor", color="group", color_map=c.GROUP_COLORS, text="label",
             hover="detail", x_title="탈모 비율 차이 (%p)")
st.plotly_chart(fig, width="stretch")

with st.expander("표로 보기"):
    st.dataframe(gaps[["factor", "group", "gap", "high", "low"]].rename(columns={
        "factor": "요인", "group": "묶음", "gap": "차이(%p)", "high": "가장 높은 값", "low": "가장 낮은 값"}),
        hide_index=True, width="stretch")

st.markdown("**페이지 안내** · 1 데이터 · 2 탈모 판별 · 3 습관·건강·식습관 · 4 생활 기록 · 5 두피 사진 · 6 코로나 전후 · **7 사진으로 판별** · 9 검산")
