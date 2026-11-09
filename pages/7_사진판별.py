"""사진으로 탈모 판별 (학습 데이터: train_photo.py가 고른 데이터셋)"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import photo

st.set_page_config(page_title="사진 판별", page_icon="📸", layout="wide")
ready = photo.model_ready()
ev = photo.load_eval() if ready else {}
CLS = ev.get("classes", ["정상", "탈모"])
c.page_header("7 · 사진으로 탈모 판별", " · ".join(CLS) + " 중 어디에 가까운지 사진 한 장으로 판별")

# ── 판별해 보기 (모델이 없어도 찍기·올리기는 항상 보인다) ──
st.subheader("내 사진 판별해 보기")
kind = ev.get("photo_kind", "머리 사진")
st.caption(f"학습한 사진은 **{kind}**이에요. 비슷하게 찍어야 결과가 맞아요"
           + (" — 머리 전체(특히 정수리·이마 라인)가 화면 가운데 오게, 밝은 곳에서 찍어 주세요." if ev.get("source") != "scalp" else ".")
           + " 사진은 이 컴퓨터 안에서만 계산하고 저장하지 않아요.")
tab_cam, tab_up = st.tabs(["📷 카메라로 찍기", "🖼️ 사진 올리기"])
with tab_cam:
    shot = st.camera_input("사진 찍기", help="브라우저가 카메라 권한을 물으면 '허용'을 눌러 주세요.")
with tab_up:
    upload = st.file_uploader("사진 (jpg·png·webp)", type=["jpg", "jpeg", "png", "webp"],
                              help="휴대폰으로 접속했다면 이 버튼을 눌러 휴대폰 카메라로 바로 찍을 수 있어요.")
src = shot or upload

if not ready:
    if src is not None:
        st.image(src, width=320, caption="찍은 사진")
    st.warning("사진은 찍혔지만, **판별 모델이 아직 학습 전**이라 결과를 낼 수 없어요. 터미널에서 한 번 실행해 주세요:")
    st.code(".\\.venv\\Scripts\\python.exe train_photo.py", language=None)
    st.stop()

model = photo.MLP.load()
if src is not None:
    a, b = st.columns([1, 2])
    a.image(src, width="stretch")
    feat = photo.features(src)
    if feat.shape[0] != model.mu.shape[0]:
        st.error(f"모델(특징 {model.mu.shape[0]}개)과 지금 코드(특징 {feat.shape[0]}개)가 맞지 않아요. "
                 "photo.py를 고친 뒤라면 대시보드를 껐다 켜고, 그래도 같으면 train_photo.py를 다시 실행해 주세요.")
        st.stop()
    p = model.proba(feat)[0]
    top = int(p.argmax())
    b.metric("판별 결과", CLS[top], f"확률 {p[top] * 100:.0f}%", delta_color="off")
    order = np.argsort(p)[::-1]
    d = pd.DataFrame({"cls": [CLS[i] for i in order], "p": p[order] * 100})
    d["label"] = d["p"].map(lambda v: f"{v:.0f}%")
    fig = c.hbar(d, x="p", y="cls", text="label", x_title="확률 (%)", height=200)
    fig.update_traces(marker_color=[c.LABEL_COLORS.get(x, c.COLORS["blue"]) for x in d["cls"].iloc[::-1]])
    fig.update_xaxes(range=[0, 110])
    b.plotly_chart(fig, width="stretch")
    if p[top] < 0.65:
        b.info("두 쪽 확률이 비슷해요. 각도나 밝기를 바꿔 다시 찍어 보세요.")
    if CLS[top] == "탈모":
        b.warning("탈모 사진과 비슷하게 나왔어요. 정확한 판단은 피부과 전문의에게 받아 보세요.")

st.divider()

# ── 모델이 얼마나 맞히나 ──
st.subheader("이 판별기는 얼마나 맞히나")
r = ev["rounds"]
log = ev.get("clean_log", {})
st.markdown(f"""
**학습 데이터:** {ev['source_title']} — 정리 후 **{ev['n_labeled']:,}장** ({', '.join(f'{k} {v}장' for k, v in ev['label_counts'].items())})
**방법:** 사진마다 색·밝기·결(머리카락 선)·피부색 위치 특징 **{ev['n_features']}개**를 뽑아, 1층 신경망(numpy 직접 구현)으로 분류해요.
정답별로 **80% 학습 / 20% 검증**으로 나누고 **{len(r)}번** 반복한 평균이에요. 크기·자르기만 다른 거의 같은 사진은 한 묶음으로 묶어 학습·검증에 나뉘어 들어가지 않게 했어요.
""")
m = st.columns(4)
m[0].metric("정확도", f"{ev['accuracy']:.1f}%", help=f"{len(r)}회 검증 평균, 표준편차 ±{ev.get('accuracy_sd', 0):.1f}%p")
m[1].metric("가장 많은 쪽으로 찍기", f"{ev['baseline']:.1f}%")
m[2].metric("탈모 사진을 찾아낸 비율", f"{np.mean([x['bald_recall'] for x in r]):.0f}%", help="실제 탈모 사진 중 탈모로 맞힌 비율 (재현율)")
m[3].metric("'탈모'라고 한 것 중 맞은 비율", f"{np.mean([x['bald_precision'] for x in r]):.0f}%", help="정밀도")

cm = np.array(ev["confusion"], dtype=float)
rate = cm / cm.sum(1, keepdims=True) * 100
fig = go.Figure(go.Heatmap(
    z=rate, x=[f"판별: {x}" for x in CLS], y=[f"실제: {x}" for x in CLS],
    colorscale=c.SEQ_BLUE, zmin=0, zmax=100,
    text=[[f"{v:.0f}%<br>({int(n)}장)" for v, n in zip(rv, rn)] for rv, rn in zip(rate, cm)],
    texttemplate="%{text}", hovertemplate="%{y} → %{x}: %{z:.1f}%<extra></extra>", xgap=2, ygap=2,
    colorbar=dict(title="%")))
fig.update_yaxes(autorange="reversed")
st.markdown(f"**헷갈림표** · 줄마다 실제 정답 사진이 무엇으로 판별됐는지 (검증 {len(r)}회 합계)")
st.plotly_chart(c.style(fig, 120 + 90 * len(CLS), legend=False), width="stretch")

with st.expander("데이터 정리 기록"):
    st.json(log)

# ── 예시 ──
ex = pd.DataFrame(ev.get("examples", []))
if not ex.empty:
    ex["full"] = ex["path"].map(lambda p: photo.DATA / p)
    ex = ex[ex["full"].map(lambda p: p.exists())]
if not ex.empty:
    st.subheader("검증 사진 예시")
    pick = st.radio("보기", ["맞힌 사진", "틀린 사진"], horizontal=True)
    show = ex[(ex["label"] == ex["pred"]) == (pick == "맞힌 사진")].head(8)
    cols = st.columns(4)
    for i, row in enumerate(show.itertuples()):
        cols[i % 4].image(str(row.full), caption=f"실제 {row.label} → 판별 {row.pred} ({row.conf * 100:.0f}%)", width="stretch")

st.info("**한계:** 딥러닝이 아니라 색·질감 숫자만 보는 가벼운 판별기예요. 검산해 보니 **사진 가장자리 밝기(배경)만으로도 약 64%를 맞혀서**, "
        "머리카락이 아니라 사진 분위기·배경을 단서로 쓰는 부분이 있어요. 인터넷에서 모은 사진이라 실제 내 사진과 다를 수 있고, 의학적 진단이 아니에요. "
        "이 PC는 보안 정책 때문에 scikit-learn·딥러닝 라이브러리를 쓸 수 없어 numpy로 직접 만들었어요.")
