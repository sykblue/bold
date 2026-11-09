"""요인 탐색: 건강 상태 · 식습관·영양 · 생활 습관"""
import streamlit as st

import common as c

st.set_page_config(page_title="습관·건강·식습관", page_icon="🥗", layout="wide")
c.page_header("3 · 습관 · 건강 · 식습관", "999명 설문에서 요인 값마다 탈모 비율이 어떻게 다른가")

main = c.load_main()
overall = main["hair_loss"].mean() * 100

min_n = st.slider("이 인원보다 적은 값은 빼기", 1, 100, 30, help="인원이 적으면 비율이 우연히 크게 흔들려요.")

for group, tab in zip(c.GROUP_COLORS, st.tabs(list(c.GROUP_COLORS))):
    with tab:
        cols = [col for col, (_, g) in c.FACTORS.items() if g == group]
        for col in cols:
            name = c.FACTORS[col][0]
            r = c.factor_rates(main, col, min_n)
            if r.empty:
                st.caption(f"{name}: 조건에 맞는 값이 없어요.")
                continue
            r["label"] = r.apply(lambda x: f"{x['rate']:.0f}% · {x['n']}명", axis=1)
            r["hover"] = r["n"].map(lambda n: f"{n}명")
            st.markdown(f"**{name}**")
            fig = c.hbar(r, x="rate", y=col, text="label", hover="hover", x_title="탈모 비율 (%)")
            fig.update_traces(marker_color=c.GROUP_COLORS[group])
            fig.add_vline(x=overall, line_dash="dot", line_color=c.COLORS["gray"],
                          annotation_text=f"전체 {overall:.0f}%", annotation_position="top")
            fig.update_xaxes(range=[0, 80])
            st.plotly_chart(fig, width="stretch")

st.info("모든 값이 전체 평균(점선) 근처에 몰려 있어요. 설문 데이터 안에서는 **어떤 습관·건강·식습관 요인도 탈모 비율을 크게 바꾸지 않아요.** "
        "실제 의학 연구 결과와 다를 수 있고, 이 데이터가 합성 데이터일 가능성도 있어요(1페이지 '주의할 점').")
