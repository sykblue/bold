"""사진 판별 모델 학습 (개발 담당)

    python train_photo.py                 # 기본: 머리 사진 탈모/정상 (sithukaungset/hairlossdataset)
    python train_photo.py --source scalp  # 대회 두피 확대 사진 4종 (대회 규칙 동의 필요)
    python train_photo.py --no-download   # 이미 받은 사진으로만

결과:
    data/photo_sets/<source>/   사진 원본 (git에 올리지 않음)
    data/photo_model.npz        판별 모델
    data/photo_eval.json        검증 결과 (페이지 7·9에서 사용)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import photo

DATA = photo.DATA
SETS = DATA / "photo_sets"

SOURCES = {
    "hairloss": {
        "kaggle": "sithukaungset/hairlossdataset",
        "title": "머리 사진 탈모/정상 (Kaggle sithukaungset/hairlossdataset)",
        "classes": ["정상", "탈모"],
        "folders": {"notbald": "정상", "bald": "탈모"},
        "photo_kind": "머리 전체가 보이는 일반 사진",
    },
    "scalp": {
        "kaggle": "2022-ai-tp-18011886",
        "title": "두피 확대 사진 4종 (Kaggle 대회 2022-ai-tp-18011886)",
        "classes": ["정상", "비듬성", "염증성", "탈모"],
        "folders": {"정상": "정상", "비듬성": "비듬성", "염증성": "염증성", "탈모": "탈모"},
        "photo_kind": "두피 진단기로 가까이 찍은 확대 사진",
    },
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def download(source: str) -> Path:
    cfg = SOURCES[source]
    root = SETS / source
    if root.exists() and any(root.rglob("*.*")):
        print(f"{root} 에 사진이 이미 있어요 → 다운로드 건너뜀")
        return root
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    root.mkdir(parents=True, exist_ok=True)
    if source == "scalp":
        api.competition_download_files(cfg["kaggle"], path=root, quiet=False)
        z = next(root.glob("*.zip"))
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():  # 안전: train/ 아래 사진만, 상위 폴더로 나가는 경로 무시
                parts = Path(info.filename).parts
                if Path(info.filename).suffix.lower() in IMG_EXT and ".." not in parts and parts[0] == "train":
                    target = root / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(info))
        z.unlink()
    else:
        api.dataset_download_files(cfg["kaggle"], path=root, unzip=True, quiet=False)
    return root


def collect(source: str, root: Path) -> tuple[pd.DataFrame, dict]:
    """사진 목록 + 정답. 똑같은 사진(내용 해시가 같음)을 정리한다."""
    cfg = SOURCES[source]
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.parent.name in cfg["folders"]:
            rows.append({"path": p, "label": cfg["folders"][p.parent.name],
                         "hash": hashlib.md5(p.read_bytes()).hexdigest()})
    df = pd.DataFrame(rows)
    log = {"원본 사진": len(df)}
    n_labels = df.groupby("hash")["label"].nunique()
    conflict = df["hash"].isin(n_labels[n_labels > 1].index)
    log["정답이 엇갈린 같은 사진(제외)"] = int(conflict.sum())
    df = df[~conflict]
    dup = df.duplicated("hash")
    log["같은 정답 안 중복(1장만 남김)"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    # 묶음: 대회 사진은 같은 촬영 회차('묶음_기기_촬영시각'),
    # 그 밖에는 '거의 같은 사진'(크기·자르기만 다른 사진)끼리 묶는다 → 학습·검증에 나뉘어 들어가지 않게
    if source == "scalp":
        df["group"] = df["path"].map(lambda p: p.name.rsplit("_", 2)[0])
    else:
        df["group"] = near_duplicate_groups(list(df["path"]))
        log["거의 같은 사진 묶음(2장 이상)"] = int((df["group"].value_counts() > 1).sum())
        mixed = df.groupby("group")["label"].transform("nunique") > 1
        log["정답이 엇갈린 거의 같은 사진(제외)"] = int(mixed.sum())
        df = df[~mixed].reset_index(drop=True)
    log["학습에 쓴 사진"] = len(df)
    return df, log


def near_duplicate_groups(paths: list[Path], max_diff: int = 10) -> np.ndarray:
    """16x16 흑백 평균 해시가 256칸 중 max_diff칸 이하로 다르면 같은 묶음."""
    from PIL import Image

    def ahash(p):
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((16, 16)), float)
        return (a > a.mean()).ravel()

    H = np.stack([ahash(p) for p in paths])
    d = (H[:, None, :] != H[None, :, :]).sum(-1)
    parent = list(range(len(paths)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in np.argwhere(np.triu(d <= max_diff, 1)):
        parent[root(a)] = root(b)
    return np.array([root(i) for i in range(len(paths))])


def extract(paths: list[Path]) -> np.ndarray:
    t = time.time()
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(photo.features, paths, chunksize=16))
    print(f"특징 {len(paths):,}장 · {time.time() - t:.0f}초")
    return np.stack(out)


def group_split(groups: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """정답마다 묶음(group)의 frac만큼을 검증용으로 고른다.

    같은 회차에 찍은 사진이 학습·검증에 나뉘어 들어가면 점수가 부풀려지므로 묶음 단위로 나눈다.
    정답별로 따로 고르기 때문에 어느 정답도 학습에서 빠지지 않는다.
    """
    rng = np.random.default_rng(seed)
    val = np.zeros(len(groups), bool)
    for c in np.unique(y):
        uniq = rng.permutation(np.unique(groups[y == c]))
        pick = set(uniq[: max(1, int(len(uniq) * frac))])
        val |= (y == c) & np.isin(groups, list(pick))
    return val


def metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    bald = classes.index("탈모")
    tp = cm[bald, bald]
    return {
        "accuracy": float((y == pred).mean() * 100),
        "recall": {c: float(cm[i, i] / max(cm[i].sum(), 1) * 100) for i, c in enumerate(classes)},
        "confusion": cm.tolist(),
        "bald_precision": float(tp / max(cm[:, bald].sum(), 1) * 100),
        "bald_recall": float(tp / max(cm[bald].sum(), 1) * 100),
    }


def main() -> None:
    source = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "hairloss"
    cfg = SOURCES[source]
    classes = cfg["classes"]
    root = SETS / source if "--no-download" in sys.argv else download(source)
    files, log = collect(source, root)
    if files.empty:
        sys.exit("사진이 없어요. 다운로드부터 확인하세요.")

    X = extract(list(files["path"]))
    y = files["label"].map(classes.index).to_numpy()
    groups = files["group"].to_numpy()

    # 검증: 정답별·묶음 단위 80/20을 5번 바꿔 가며 평균
    rounds, examples = [], []
    for seed in range(5):
        val = group_split(groups, y, 0.2, seed)
        model = photo.MLP(X.shape[1], 64, len(classes), seed=seed).fit(X[~val], y[~val], epochs=150, seed=seed)
        p = model.proba(X[val])
        m = metrics(y[val], p.argmax(1), classes)
        majority = np.bincount(y[~val], minlength=len(classes)).argmax()
        m["baseline"] = float((y[val] == majority).mean() * 100)
        m["n_train"], m["n_val"] = int((~val).sum()), int(val.sum())
        rounds.append(m)
        print(f"검증 {seed}: 정확도 {m['accuracy']:.1f}% (찍기 {m['baseline']:.1f}%)")
        if seed == 0:
            vf = files[val].reset_index(drop=True)
            examples = [{"path": str(Path(r.path).relative_to(DATA)).replace("\\", "/"), "label": r.label,
                         "pred": classes[int(i)], "conf": float(pp.max())}
                        for r, i, pp in zip(vf.itertuples(), p.argmax(1), p)]

    final = photo.MLP(X.shape[1], 64, len(classes), seed=0).fit(X, y, epochs=150)
    final.save()
    ev = {
        "source": source,
        "source_title": cfg["title"],
        "photo_kind": cfg["photo_kind"],
        "classes": classes,
        "clean_log": log,
        "rounds": rounds,
        "accuracy": float(np.mean([r["accuracy"] for r in rounds])),
        "accuracy_sd": float(np.std([r["accuracy"] for r in rounds])),
        "baseline": float(np.mean([r["baseline"] for r in rounds])),
        "confusion": np.sum([r["confusion"] for r in rounds], 0).tolist(),
        "n_labeled": len(files),
        "n_features": int(X.shape[1]),
        "label_counts": files["label"].value_counts().to_dict(),
        "examples": pd.DataFrame(examples).sample(min(80, len(examples)), random_state=0).to_dict("records"),
    }
    photo.EVAL_PATH.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False))
    print(f"평균 정확도 {ev['accuracy']:.1f}% (±{ev['accuracy_sd']:.1f}) · 찍기 {ev['baseline']:.1f}% → {photo.MODEL_PATH.name} 저장")


if __name__ == "__main__":
    main()
