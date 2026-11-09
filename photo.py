"""사진 판별 공통 모듈 (학습 train_photo.py · 페이지 7_사진판별.py가 함께 씀)

딥러닝 없이 numpy로 만든 판별기:
1. 사진 한 장에서 색·밝기·결(질감)·피부색 위치 특징 숫자를 뽑는다.
2. 1층 신경망(MLP)으로 정답 종류(예: 정상 / 탈모) 중 하나로 분류한다.
정답 종류는 학습한 데이터셋에 따라 달라지고, photo_eval.json의 "classes"에 저장된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DATA = Path(__file__).parent / "data"
MODEL_PATH = DATA / "photo_model.npz"
EVAL_PATH = DATA / "photo_eval.json"
SIZE = 192
GRID = 4


# ── 특징 뽑기 ───────────────────────────────────────────────────
def _load(src) -> np.ndarray:
    img = Image.open(src)
    img.draft("RGB", (SIZE * 2, SIZE * 2))  # JPEG는 작게 풀어서 빠르게 (다른 형식은 그냥 넘어감)
    img = ImageOps.exif_transpose(img)  # 휴대폰 사진 회전 정보 반영
    img = ImageOps.fit(img.convert("RGB"), (SIZE, SIZE), Image.BILINEAR)  # 가운데 기준 정사각형
    return np.asarray(img, dtype=np.float32) / 255.0


def _hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    return np.stack([h, s, mx], -1)


def _skin(rgb: np.ndarray) -> np.ndarray:
    """피부색 화소 (YCrCb 범위 규칙). 머리카락이 없으면 정수리에 피부색이 넓게 보인다."""
    r, g, b = rgb[..., 0] * 255, rgb[..., 1] * 255, rgb[..., 2] * 255
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135) & (y > 60)).astype(np.float32)


def _stats(x: np.ndarray) -> list[float]:
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return [x.mean(), x.std(), *p]


def _grid(x: np.ndarray) -> list[float]:
    s = SIZE // GRID
    return list(x[: s * GRID, : s * GRID].reshape(GRID, s, GRID, s).mean((1, 3)).ravel())


def features(src) -> np.ndarray:
    rgb = _load(src)
    hsv = _hsv(rgb)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    f: list[float] = []
    for ch in range(3):  # RGB·HSV 채널 통계
        f += _stats(rgb[..., ch])
        f += _stats(hsv[..., ch])
    f += list(np.histogram(hsv[..., 0], 12, (0, 1))[0] / gray.size)
    f += list(np.histogram(hsv[..., 1], 8, (0, 1))[0] / gray.size)
    f += list(np.histogram(gray, 12, (0, 1))[0] / gray.size)
    redness = rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) / 2
    f += _stats(redness)
    # 결: 기울기(머리카락 선), 밝은 점, 어두운 비율(머리카락 밀도)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    f += _stats(mag)
    f += [(mag > t).mean() for t in (0.02, 0.05, 0.1, 0.2)]
    ang = np.arctan2(gy, gx)[mag > 0.05]
    f += list(np.histogram(ang, 8, (-np.pi, np.pi))[0] / max(len(ang), 1))
    m = gray.mean()
    f += [(gray < m - t).mean() for t in (0.1, 0.2, 0.3)] + [(gray > m + t).mean() for t in (0.1, 0.2, 0.3)]
    # LBP(주변 8칸보다 밝은지) 16구간 히스토그램
    c = gray[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.int32)
    for i, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        code |= (gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx] >= c).astype(np.int32) << i
    f += list(np.histogram(code, 16, (0, 256))[0] / code.size)
    # 위치별 정보 (4x4 칸): 밝기, 결의 세기, 피부색 비율
    skin = _skin(rgb)
    f += _grid(gray) + _grid(mag) + _grid(skin)
    f += [skin.mean(), skin[: SIZE // 2].mean() - skin[SIZE // 2:].mean()]  # 위쪽(정수리) 피부색이 더 많은지
    return np.asarray(f, dtype=np.float32)


# ── 1층 신경망 (numpy) ─────────────────────────────────────────
class MLP:
    def __init__(self, n_in: int, n_hidden: int = 64, n_out: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.mu = np.zeros(n_in, np.float32)
        self.sd = np.ones(n_in, np.float32)

    def _forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return h, p / p.sum(1, keepdims=True)

    def fit(self, X, y, epochs=300, lr=0.05, l2=1e-3, batch=256, seed=0, class_weight=True):
        k = self.W2.shape[1]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        Y = np.eye(k, dtype=np.float32)[y]
        w = (len(y) / (k * np.maximum(np.bincount(y, minlength=k), 1))).astype(np.float32) if class_weight else np.ones(k, np.float32)
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                xb, yb = X[b], Y[b]
                h, p = self._forward(xb)
                g = (p - yb) * w[y[b]][:, None] / len(b)
                gW2 = h.T @ g + l2 * self.W2
                gb2 = g.sum(0)
                gh = (g @ self.W2.T) * (h > 0)
                gW1 = xb.T @ gh + l2 * self.W1
                gb1 = gh.sum(0)
                t += 1
                for i, gr in enumerate([gW1, gb1, gW2, gb2]):  # Adam
                    m[i] = 0.9 * m[i] + 0.1 * gr
                    v[i] = 0.999 * v[i] + 0.001 * gr * gr
                    params[i] -= lr * 0.1 * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def proba(self, X):
        return self._forward((np.atleast_2d(X) - self.mu) / self.sd)[1]

    def save(self, path=None):
        path = path or MODEL_PATH
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path=None) -> "MLP":
        path = path or MODEL_PATH
        d = np.load(path)
        m = cls(d["W1"].shape[0], d["W1"].shape[1], d["W2"].shape[1])
        for k in ("W1", "b1", "W2", "b2", "mu", "sd"):
            setattr(m, k, d[k])
        return m


def model_ready() -> bool:
    return MODEL_PATH.exists() and EVAL_PATH.exists()


def load_eval() -> dict:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
