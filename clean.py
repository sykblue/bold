"""공통 정제 (개발 담당)

원본(data/raw*.csv)을 읽어 정제본(data/clean*.csv)을 만든다.
모든 페이지는 정제본만 읽는다. 원본은 고치지 않는다.

    python clean.py              # 정제만
    python clean.py --download   # Kaggle에서 원본을 다시 받은 뒤 정제 (Kaggle 토큰 필요)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
LOG: dict[str, dict] = {}

# ── 1. 탈모 요인 설문 (raw.csv → clean.csv) ──────────────────────────
MAIN_COLS = {
    "Id": "id",
    "Genetics": "genetics",
    "Hormonal Changes": "hormonal",
    "Medical Conditions": "medical",
    "Medications & Treatments": "medication",
    "Nutritional Deficiencies": "nutrition",
    "Stress": "stress",
    "Age": "age",
    "Poor Hair Care Habits": "poor_care",
    "Environmental Factors": "environment",
    "Smoking": "smoking",
    "Weight Loss": "weight_loss",
    "Hair Loss": "hair_loss",
}

KO_VALUES = {
    "Yes": "예", "No": "아니오", "No Data": "정보 없음",
    "Low": "낮음", "Moderate": "보통", "High": "높음",
    # 질환
    "Alopecia Areata": "원형 탈모증", "Androgenetic Alopecia": "남성형(안드로겐성) 탈모",
    "Dermatitis": "피부염", "Dermatosis": "피부병", "Eczema": "습진", "Psoriasis": "건선",
    "Ringworm": "백선(곰팡이)", "Scalp Infection": "두피 감염",
    "Seborrheic Dermatitis": "지루성 피부염", "Thyroid Problems": "갑상선 질환",
    # 약물·치료
    "Accutane": "여드름약(이소트레티노인)", "Antibiotics": "항생제", "Antidepressants": "항우울제",
    "Antifungal Cream": "항진균 크림", "Blood Pressure Medication": "혈압약",
    "Chemotherapy": "항암 치료", "Heart Medication": "심장약", "Immunomodulators": "면역조절제",
    "Rogaine": "미녹시딜(로게인)", "Steroids": "스테로이드",
    # 영양 결핍
    "Biotin Deficiency": "비오틴 결핍", "Iron deficiency": "철분 결핍",
    "Magnesium deficiency": "마그네슘 결핍", "Omega-3 fatty acids": "오메가3 부족",
    "Protein deficiency": "단백질 결핍", "Selenium deficiency": "셀레늄 결핍",
    "Vitamin A Deficiency": "비타민A 결핍", "Vitamin D Deficiency": "비타민D 결핍",
    "Vitamin E deficiency": "비타민E 결핍", "Zinc Deficiency": "아연 결핍",
}


def clean_main() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "raw.csv")
    log = {"원본 행": len(raw)}
    raw.columns = raw.columns.str.strip()  # 원본 열 이름 끝에 공백이 붙은 것이 있음
    df = raw.rename(columns=MAIN_COLS)
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip().map(lambda v: KO_VALUES.get(v, v))
    log["빈칸"] = int(df.isna().sum().sum())
    # id가 겹치는 4쌍은 나머지 값이 모두 달라 서로 다른 사람으로 본다 → 지우지 않고 행 번호로 새 키를 만든다
    log["겹치는 id(유지)"] = int(df["id"].duplicated(keep=False).sum())
    log["완전히 같은 행 제거"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df.insert(0, "row_id", df.index + 1)
    df["age_group"] = pd.cut(df["age"], [0, 24, 34, 44, 120],
                             labels=["~24세", "25~34세", "35~44세", "45세~"]).astype(str)
    log["정제 후 행"] = len(df)
    log["탈모 비율"] = round(float(df["hair_loss"].mean()), 4)
    df.to_csv(DATA / "clean.csv", index=False, encoding="utf-8-sig")
    LOG["clean.csv"] = log
    return df


# ── 2. 한 사람의 400일 생활 기록 (raw_daily.csv → clean_daily.csv) ──────
LEVEL4 = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
HAIR4 = {"Few": 0, "Medium": 1, "Many": 2, "A lot": 3}
HAIR4_KO = {"Few": "적음", "Medium": "보통", "Many": "많음", "A lot": "매우 많음"}


def clean_daily() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "raw_daily.csv")
    log = {"원본 행": len(raw)}
    raw.columns = raw.columns.str.strip()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"].str.strip(), format="%d/%m/%Y"),
        "hair_loss_level": raw["hair_loss"].map(HAIR4),
        "hair_loss_ko": raw["hair_loss"].map(HAIR4_KO),
        "stay_up_late": raw["stay_up_late"],
        "coffee": raw["coffee_consumed"],
        "brain_hours": raw["brain_working_duration"],
        "pressure_level": raw["pressure_level"].map(LEVEL4),
        "stress_level": raw["stress_level"].map(LEVEL4),
        "school": raw["school_assesssment"].fillna("없음"),
        "shampoo": raw["shampoo_brand"],
        "swimming": raw["swimming"].eq("Yes"),
        "hair_washing": raw["hair_washing"].eq("Y"),
        "hair_grease": raw["hair_grease"],
        "dandruff": raw["dandruff"].fillna("기록 없음"),
        "libido": raw["libido"],
    })
    log["날짜 범위"] = f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}"
    log["두피 유분 빈칸(그대로 둠)"] = int(df["hair_grease"].isna().sum())
    log["시험·과제 빈칸 → '없음'"] = int(raw["school_assesssment"].isna().sum())
    log["비듬 빈칸 → '기록 없음'"] = int(raw["dandruff"].isna().sum())
    log["매핑 실패"] = int(df[["hair_loss_level", "pressure_level", "stress_level"]].isna().sum().sum())
    log["정제 후 행"] = len(df)
    df.to_csv(DATA / "clean_daily.csv", index=False, encoding="utf-8-sig")
    LOG["clean_daily.csv"] = log
    return df


# ── 3. 두피 사진 파일 목록 (raw_scalp_files.csv → clean_scalp.csv) ──────
NAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<device>[A-Z0-9]+)_(?P<ts>\d{13})_(?P<shot>\d)_(?P<pos>[A-Z]{2})\.jpg$")


def clean_scalp() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "raw_scalp_files.csv")
    log = {"원본 파일": len(raw)}
    img = raw[raw["name"].str.endswith(".jpg")].copy()
    log["사진 아닌 파일 제외"] = len(raw) - len(img)
    parts = img["name"].str.split("/")
    img["split"] = parts.str[0]
    img["label"] = parts.map(lambda p: p[1] if len(p) == 3 else "(정답 비공개)")
    img["file"] = parts.str[-1]
    m = img["file"].str.extract(NAME_RE)
    log["이름 형식 불일치"] = int(m["subject"].isna().sum())
    img = pd.concat([img, m], axis=1).dropna(subset=["subject"])
    img["taken_at"] = (pd.to_datetime(img["ts"].astype("int64"), unit="ms", utc=True)
                       .dt.tz_convert("Asia/Seoul").dt.tz_localize(None))
    img["shot"] = img["shot"].astype(int)
    img["mb"] = (img["total_bytes"] / 1e6).round(3)
    out = img[["split", "label", "subject", "device", "taken_at", "shot", "pos", "mb", "file"]]
    log["정제 후 사진"] = len(out)
    out.to_csv(DATA / "clean_scalp.csv", index=False, encoding="utf-8-sig")
    LOG["clean_scalp.csv"] = log
    return out


# ── 4. '낮은 자존감' 트윗, 코로나 전/후 (raw_tweets_* → clean_tweets.csv) ──
def clean_tweets() -> pd.DataFrame:
    pre = pd.read_csv(DATA / "raw_tweets_pre.csv", encoding_errors="replace")
    post = pd.read_excel(DATA / "raw_tweets_post.xlsx")
    log = {"코로나 전 원본": len(pre), "코로나 후 원본": len(post)}
    frames = []
    for period, d in [("코로나 전", pre), ("코로나 후", post)]:
        d = d[["text", "created_at", "scores", "sentiment", "stars"]].copy()
        d["period"] = period
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["text"] = df["text"].astype("string").str.strip()
    bad = df["text"].isna() | df["text"].eq("") | df["created_at"].isna()
    log["본문·날짜 빈칸 제거"] = int(bad.sum())
    df = df[~bad]
    dup = df.duplicated(["period", "text"])
    log["같은 기간 중복 본문 제거"] = int(dup.sum())
    df = df[~dup].copy()
    df["created_at"] = df["created_at"].dt.tz_localize(None)
    df["sentiment"] = df["sentiment"].map({"positive": "긍정", "negative": "부정", "neutral": "중립"}).fillna("기타")
    for period in ["코로나 전", "코로나 후"]:
        log[f"{period} 정제 후"] = int((df["period"] == period).sum())
    df.to_csv(DATA / "clean_tweets.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
    LOG["clean_tweets.csv"] = log
    return df


def download() -> None:
    """Kaggle 원본 다시 받기. 대회 파일은 사진(3.1GB)이라 목록만 받는다."""
    import shutil
    import tempfile
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    tmp = Path(tempfile.mkdtemp())
    src = {
        "amitvkulkarni/hair-health": ("Predict Hair Fall.csv", "raw.csv"),
        "lukexun/luke-hair-loss-dataset": ("Luke_hair_loss_documentation.csv", "raw_daily.csv"),
        "kareemrasheed89/precovid-10000-tweets-for-analysis": ("10000 Low Self Esteem Tweets.csv", "raw_tweets_pre.csv"),
        "kareemrasheed89/5000-tweets-with-keyword-lowselfesteem-postcovid": ("lowselfesteempostcovid.xlsx", "raw_tweets_post.xlsx"),
    }
    for ref, (name, target) in src.items():
        api.dataset_download_files(ref, path=tmp / target, unzip=True, quiet=True)
        shutil.copy(tmp / target / name, DATA / target)
    files, token = [], None
    while True:
        r = api.competition_list_files("2022-ai-tp-18011886", page_token=token, page_size=200)
        files += r.files
        token = r.next_page_token
        if not token:
            break
    with open(DATA / "raw_scalp_files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "total_bytes", "creation_date"])
        for x in files:
            w.writerow([x.name, x.total_bytes, x.creation_date])


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    clean_main()
    clean_daily()
    clean_scalp()
    clean_tweets()
    (DATA / "clean_log.json").write_text(json.dumps(LOG, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(LOG, ensure_ascii=False, indent=2, default=str))
