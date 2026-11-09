# 데이터 카드 + 정제 기록

모든 페이지는 `clean*.csv`만 읽는다. `raw*`는 Kaggle에서 받은 그대로 두고 고치지 않는다.
정제는 `python clean.py`로 다시 만들 수 있고, 숫자 기록은 `clean_log.json`에 남는다.

## 파일 한눈에

| 원본 (raw) | 정제본 (clean) | 출처 (Kaggle) | 행 | 쓰는 페이지 |
|---|---|---|---|---|
| `raw.csv` | `clean.csv` | `amitvkulkarni/hair-health` (Predict Hair Fall.csv) | 999 | 홈·2·3 |
| `raw_daily.csv` | `clean_daily.csv` | `lukexun/luke-hair-loss-dataset` | 400 | 홈·4 |
| `raw_scalp_files.csv` | `clean_scalp.csv` | 대회 `2022-ai-tp-18011886`의 **파일 목록** | 12,729 | 홈·5 |
| `raw_tweets_pre.csv` + `raw_tweets_post.xlsx` | `clean_tweets.csv` | `kareemrasheed89/precovid-10000-tweets-for-analysis`, `kareemrasheed89/5000-tweets-with-keyword-lowselfesteem-postcovid` | 13,628 | 6 |

- 주제 데이터: 사용자 지정 대회 `2022-ai-tp-18011886` ("두피 이미지를 통한 두피 질환 및 탈모 진단 예측"), 노트북 `adeyinkaajagunna/cosine-similarities-for-during-and-before-covid`
- 대회 사진(3.1GB)은 **대회 규칙 동의 전이라 받지 못했다(403)**. 파일 목록 API로 이름·폴더·크기만 받았다.
- 노트북이 읽는 원래 트윗 데이터는 노트북 메타데이터에 출처가 비어 있다. 같은 작성자(kareemrasheed89)의 '낮은 자존감' 코로나 전/후 데이터셋으로 대신했다. 노트북에서 쓴 '높은 자존감' 데이터는 찾지 못했다.
- '습관·건강상태·식습관' 요인은 대회 데이터에 없어서, 탈모 요인 공개 데이터 2개(hair-health, luke)를 추가했다.

## 정제 기록

### clean.csv (탈모 요인 설문)
- 열 이름 끝 공백 제거 (`Nutritional Deficiencies `, `Poor Hair Care Habits `, `Weight Loss `), 영어 snake_case로 바꿈
- 값 한글화 (예/아니오, 질환·약·영양소 이름), `No Data` → `정보 없음`(빈칸 아님, 그대로 범주로 둠)
- **id가 겹치는 8행(4쌍)**: 나머지 값이 모두 달라 다른 사람으로 보고 **지우지 않음**. 새 키 `row_id` 추가
- 완전히 같은 행: 0개
- `age_group` 추가 (~24 / 25~34 / 35~44 / 45~)
- 결과 999행, 탈모 비율 49.75%

### clean_daily.csv (한 사람의 400일 기록)
- `date ` 열 이름 공백 제거, `일/월/년` 날짜 해석 → 2021-01-01 ~ 2022-02-04 (빠진 날 없음)
- 탈모량 Few/Medium/Many/A lot → 0~3, 스트레스·압박감 Low~Very High → 0~3
- 시험·과제 빈칸 305개 → `없음`, 비듬 빈칸 289개 → `기록 없음` (빈칸이 '없음'인지 확실하지 않아 구분해 둠)
- 두피 유분 빈칸 4개는 그대로 둠 (상관계수 계산 때 자동 제외)

### clean_scalp.csv (두피 사진 목록)
- `sample_submit.csv` 1개 제외, jpg 12,729개
- 경로 `train/<정답>/파일.jpg` → `split`, `label` / 시험 사진은 `(정답 비공개)`
- 파일 이름 `묶음번호_기기_촬영시각(ms)_순번_위치.jpg` 분해, 형식 불일치 0개
- 촬영 시각은 한국 시간으로 변환. 위치 코드(TH/LH/RH/BH)의 뜻은 **추정**(정수리/왼쪽/오른쪽/뒤)

### clean_tweets.csv (코로나 전/후 트윗)
- 쓰는 열만 남김: text, created_at, scores, sentiment, stars (+ period)
- 본문·날짜 빈칸 2행 제거, **같은 시기 안에서 본문이 똑같은 1,371행 제거**(리트윗 등)
- 결과: 코로나 전 8,730건(2017-03~2018-12), 코로나 후 4,898건(2021-02~2022-06)
- 감정 값(positive/negative/neutral)은 원작자가 붙인 값 → 긍정/부정/중립

### 사진 판별 (`python train_photo.py`)
- 대회 사진은 토큰 계정이 대회에 참가하지 않은 상태라 받을 수 없어서(403), **`sithukaungset/hairlossdataset`**(머리 사진 bald 560 / notbald 554)으로 학습했다.
  대회 사진이 열리면 `python train_photo.py --source scalp`로 바꿔 학습할 수 있다.
- `data/photo_sets/<source>/` ← 사진 원본 (git 제외)
- 정리: 내용이 완전히 같은 사진 중 정답이 엇갈린 14장 제외, 같은 정답 안 중복 3장 제거,
  **크기·자르기만 다른 거의 같은 사진**(16x16 평균 해시 차이 ≤10)을 한 묶음으로 묶음(286묶음), 묶음 안 정답이 엇갈린 6장 제외 → **1,091장**
- 특징 172개 (`photo.features`: 색·밝기·결·4x4 위치별 밝기/결/피부색) → 1층 신경망(numpy)
- 검증: 정답별·묶음 단위 80/20 × 5회 → **정확도 74.8% (±2.3)**, 찍기 48.8%
  - 거의 같은 사진을 묶지 않으면 86%로 부풀려졌다
  - 가장자리 밝기(배경) 12개만으로도 약 64% → 배경 단서를 일부 쓰는 한계
- `photo_model.npz`(모델) · `photo_eval.json`(검증 결과, 7·9페이지에서 사용)

## 알려진 한계
- hair-health는 수집 방법이 공개되지 않았고 탈모 비율이 정확히 절반이라 **합성 데이터일 가능성**이 있다.
- luke 데이터는 **1명**의 자기 기록이다.
- 어떤 데이터도 의학적 진단에 쓸 수 없다.
