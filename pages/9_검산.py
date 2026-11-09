"""검산: 화면에 나온 숫자를 원본에서 다른 방법으로 다시 계산해 맞는지 확인"""
import math

import numpy as np
import pandas as pd
import streamlit as st

import common as c

st.set_page_config(page_title="검산", page_icon="✅", layout="wide")
c.page_header("9 · 검산", "정제본·분석 결과를 원본에서 다른 방법으로 다시 계산해 맞춰 봐요")

checks = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append({"결과": "✅ 통과" if ok else "❌ 실패", "항목": name, "근거": detail})


# 1. 설문: 행 수·탈모 비율이 원본과 같은가
raw = pd.read_csv(c.DATA / "raw.csv")
raw.columns = raw.columns.str.strip()
main = c.load_main()
check("설문 행 수 = 원본 행 수", len(main) == len(raw), f"정제 {len(main)} / 원본 {len(raw)} (지운 행 없음)")
check("설문 탈모 비율 = 원본", math.isclose(main["hair_loss"].mean(), raw["Hair Loss"].mean()),
      f"정제 {main['hair_loss'].mean():.4f} / 원본 {raw['Hair Loss'].mean():.4f}")
check("설문 빈칸 없음", int(main.isna().sum().sum()) == 0, f"빈칸 {int(main.isna().sum().sum())}개")
same = raw[raw["Id"].duplicated(keep=False)].drop(columns="Id").duplicated(keep=False).sum()
check("겹치는 id는 서로 다른 사람", same == 0, f"id 겹침 {raw['Id'].duplicated(keep=False).sum()}행, 그중 내용이 같은 행 {same}개")

# 2. 대표 차트 숫자: 가장 큰 요인 차이를 pandas crosstab으로 다시 계산
gaps = c.factor_gaps(main)
top = gaps.iloc[0]
ct = pd.crosstab(main[top["col"]], main["hair_loss"], normalize="index")[1] * 100
ct = ct[main[top["col"]].value_counts().reindex(ct.index) >= 30]
check(f"대표 차트 1위({top['factor']}) 차이", math.isclose(ct.max() - ct.min(), top["gap"]),
      f"crosstab {ct.max() - ct.min():.2f}%p / 화면 {top['gap']:.2f}%p")

# 3. 판별 모델: 한 회차를 정확도 공식으로 다시 계산
test = main.sample(frac=0.3, random_state=0)
train = main.drop(test.index)
pred = (c.NaiveBayes(list(c.FACTORS)).fit(train).proba(test) >= 0.5).astype(int)
acc = (pred == test["hair_loss"].to_numpy()).sum() / len(test) * 100
ev = c.evaluate_model()
check("판별 모델 0회차 정확도", math.isclose(acc, ev.loc[0, "model"]), f"다시 계산 {acc:.2f}% / 화면 {ev.loc[0, 'model']:.2f}%")
check("학습·검증 겹치는 사람 없음", len(set(train.index) & set(test.index)) == 0, f"학습 {len(train)} + 검증 {len(test)} = {len(main)}")

# 4. 생활 기록: 날짜가 하루씩 빠짐없이 이어지는가
d = c.load_daily()
days = (d["date"].max() - d["date"].min()).days + 1
check("생활 기록 날짜 연속", days == len(d) and d["date"].is_unique, f"{d['date'].min():%Y-%m-%d}~{d['date'].max():%Y-%m-%d} = {days}일, 기록 {len(d)}행")
r1 = d["hair_loss_level"].corr(d["coffee"])
r2 = np.corrcoef(d["hair_loss_level"], d["coffee"])[0, 1]
check("커피 상관계수 (pandas vs numpy)", math.isclose(r1, r2), f"{r1:.4f} / {r2:.4f}")

# 5. 두피 사진: 학습+시험 = 전체, 폴더 정답 4가지
s = c.load_scalp()
raw_s = pd.read_csv(c.DATA / "raw_scalp_files.csv")
check("사진 수 = 원본 jpg 수", len(s) == raw_s["name"].str.endswith(".jpg").sum(), f"정제 {len(s)} / 원본 jpg {raw_s['name'].str.endswith('.jpg').sum()}")
check("학습 정답 4가지", set(s.loc[s["split"] == "train", "label"]) == {"정상", "비듬성", "염증성", "탈모"},
      ", ".join(sorted(s.loc[s["split"] == "train", "label"].unique())))
check("시험 사진은 정답 없음", (s.loc[s["split"] == "test", "label"] == "(정답 비공개)").all(), f"시험 {int((s['split'] == 'test').sum())}장")

# 6. 트윗: 시기 구분이 날짜와 맞는가, 코사인 값을 numpy로 다시 계산
t = c.load_tweets()
pre_ok = (t.loc[t["period"] == "코로나 전", "created_at"] < "2020-01-01").all()
post_ok = (t.loc[t["period"] == "코로나 후", "created_at"] >= "2020-03-01").all()
check("코로나 전/후 날짜 구분", pre_ok and post_ok,
      f"전 {t.loc[t['period'] == '코로나 전', 'created_at'].max():%Y-%m-%d}까지 / 후 {t.loc[t['period'] == '코로나 후', 'created_at'].min():%Y-%m-%d}부터")
a = c.word_counts(t.loc[t["period"] == "코로나 전", "text"])
b = c.word_counts(t.loc[t["period"] == "코로나 후", "text"])
vocab = sorted(set(a) | set(b))
va = np.array([a.get(w, 0) for w in vocab], dtype=float)
vb = np.array([b.get(w, 0) for w in vocab], dtype=float)
np_cos = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))
check("코사인 유사도 (직접 구현 vs numpy)", math.isclose(np_cos, c.cosine(a, b)), f"{np_cos:.6f} / {c.cosine(a, b):.6f}")

# 7. 사진 판별 (모델이 있을 때만)
import photo  # noqa: E402

if photo.model_ready():
    pev = photo.load_eval()
    for i, rd in enumerate(pev["rounds"]):
        cm = np.array(rd["confusion"])
        check(f"사진 검증 {i}회차: 헷갈림표 → 정확도", math.isclose(np.trace(cm) / cm.sum() * 100, rd["accuracy"]),
              f"표 대각선 {np.trace(cm)}/{cm.sum()} = {np.trace(cm) / cm.sum() * 100:.2f}% / 기록 {rd['accuracy']:.2f}%")
    check("사진 검증 사진 수 합 = 학습 사진 수", all(r["n_train"] + r["n_val"] == pev["n_labeled"] for r in pev["rounds"]),
          f"학습+검증 = {pev['rounds'][0]['n_train']}+{pev['rounds'][0]['n_val']} / 정답 있는 사진 {pev['n_labeled']}")
else:
    checks.append({"결과": "⏸️ 건너뜀", "항목": "사진 판별 모델", "근거": "아직 학습 전 (train_photo.py 실행 필요)"})

res = pd.DataFrame(checks)
passed = (res["결과"] == "✅ 통과").sum()
st.metric("통과", f"{passed} / {(res['결과'] != '⏸️ 건너뜀').sum()}")
st.dataframe(res, hide_index=True, width="stretch")
