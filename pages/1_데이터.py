"""데이터 카드 · 정제 기록 · 미리 보기"""
import streamlit as st

import common as c

st.set_page_config(page_title="데이터", page_icon="🗂️", layout="wide")
c.page_header("1 · 데이터", "어떤 데이터를 어디서 받아, 어떻게 다듬었는지")

log = c.load_clean_log()
cards = [
    ("두피 사진 목록", "clean_scalp.csv", "2022-ai-tp-18011886 (대회)",
     "두피 이미지를 통한 두피 질환 및 탈모 진단 예측. 사진 원본(3.1GB)이 아니라 **파일 목록**만 사용. "
     "학습 사진은 폴더 이름이 정답(정상·비듬성·염증성·탈모).", c.load_scalp),
    ("탈모 요인 설문", "clean.csv", "amitvkulkarni/hair-health",
     "999명의 유전·호르몬·질환·약·영양 결핍·스트레스·나이·모발 관리·환경·흡연·체중 감소 + 탈모 여부(0/1).", c.load_main),
    ("한 사람의 생활 기록", "clean_daily.csv", "lukexun/luke-hair-loss-dataset",
     "한 사람이 400일 동안 매일 적은 탈모량·밤샘·커피·공부 시간·스트레스·샴푸·두피 유분 등.", c.load_daily),
    ("'낮은 자존감' 트윗", "clean_tweets.csv",
     "kareemrasheed89/precovid-10000-tweets-for-analysis + …/5000-tweets-with-keyword-lowselfesteem-postcovid",
     "노트북 adeyinkaajagunna/cosine-similarities-for-during-and-before-covid 가 쓰는 '코로나 전/후' 트윗. "
     "탈모와 직접 관련은 없고, 코로나 전후 마음 상태 비교용 보조 자료.", c.load_tweets),
]

for name, file, src, desc, loader in cards:
    with st.container(border=True):
        df = loader()
        a, b = st.columns([3, 1])
        a.markdown(f"#### {name} · `data/{file}`")
        a.markdown(f"출처: `{src}`  \n{desc}")
        b.metric("정제 후 행", f"{len(df):,}")
        with st.expander("정제 기록 · 미리 보기"):
            st.json(log.get(file, {}))
            st.dataframe(df.head(20), width="stretch")

st.subheader("주의할 점")
st.markdown("""
- **설문 데이터(hair-health)**는 출처·수집 방법이 공개돼 있지 않고, 탈모 비율이 정확히 절반(49.8%)이에요. 실제 조사가 아니라 **만든(합성) 데이터일 수 있어요.**
- **생활 기록**은 **한 사람**의 기록이에요. 다른 사람에게 그대로 적용할 수 없어요.
- **두피 사진**은 대회 규칙 동의 전이라 사진은 받지 못했고(403), 파일 이름·폴더·크기만 썼어요.
- **트윗**의 감정(긍정·부정·중립)과 점수는 원작자가 붙인 값을 그대로 썼어요.
""")
