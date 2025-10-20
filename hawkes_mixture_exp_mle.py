from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import optax
from quadax import quadgk  # 数値積分のフォールバック

Array = jnp.ndarray

# ---------------------------
# 型定義: baseline のコールバック
# ---------------------------
# mu_fn: (d, t, mu_params) -> μ_d(t)   （非負はユーザー側 or 内部でsoftplus可）
MuFn = Callable[[int, Array, Any], Array]
# mu_int_fn: (d, t0, t1, mu_params) -> ∫_{t0}^{t1} μ_d(s) ds
MuIntFn = Optional[Callable[[int, Array, Array, Any], Array]]


# ---------------------------
# データ構造
# ---------------------------
@dataclass
class Sequence:
    # 各次元dのイベント時刻（昇順, 形は任意の長さ）
    events_by_dim: List[Array]
    # 観測区間 [t0, t1]
    t0: float
    t1: float


# ---------------------------
# 補助: イベントの結合・整列
# ---------------------------
def _merge_and_sort_events(events_by_dim: List[Array]) -> Tuple[Array, Array]:
    """
    与えられた D 個の時刻配列を1列にまとめ, (time, mark) を時刻順にソートして返す.
    time: shape (N,), mark: shape (N,) int32
    """

    times_list: List[Array] = []
    marks_list: List[Array] = []
    for d, arr in enumerate(events_by_dim):
        arr = jnp.asarray(arr)
        if arr.ndim != 1:
            raise ValueError("各次元の時刻は1次元配列である必要があります")
        times_list.append(arr)
        marks_list.append(jnp.full(arr.shape, d, dtype=jnp.int32))

    if len(times_list) == 0:
        return jnp.zeros((0,)), jnp.zeros((0,), dtype=jnp.int32)

    times = jnp.concatenate(times_list, axis=0) if len(times_list) > 0 else jnp.zeros((0,))
    marks = (
        jnp.concatenate(marks_list, axis=0) if len(marks_list) > 0 else jnp.zeros((0,), dtype=jnp.int32)
    )
    order = jnp.argsort(times, kind="stable")
    return times[order], marks[order]


# ---------------------------
# baseline 積分: 手動 or quadax にフォールバック
# ---------------------------
def _baseline_integral_sum_over_dims(
    mu_fn: MuFn,
    mu_int_fn: MuIntFn,
    mu_params: Any,
    D: int,
    t0: Array,
    t1: Array,
    *,
    quad_epsabs: float = 1e-6,
    quad_epsrel: float = 1e-6,
) -> Array:
    """
    Σ_d ∫_{t0}^{t1} μ_d(s) ds を返す.
    """

    def _one_dim_int(d: int) -> Array:
        d_idx = int(d)
        if mu_int_fn is not None:
            return mu_int_fn(d_idx, t0, t1, mu_params)

        # フォールバック: quadax.quadgk を使用（JAX可微分）
        def f(s: Array) -> Array:
            return mu_fn(d_idx, s, mu_params)

        val, _info = quadgk(f, jnp.array([t0, t1]), epsabs=quad_epsabs, epsrel=quad_epsrel)
        return val

    # ループで合計（D は小さいはず）
    total = jnp.array(0.0)
    for d in range(D):
        total = total + _one_dim_int(d)
    return total


# ---------------------------
# 片系列の対数尤度（負）を JIT 化したコア
# ---------------------------
def _single_sequence_nll_core(
    times: Array,  # shape (N,)
    marks: Array,  # shape (N,) int32 in {0,...,D-1}
    t0: Array,
    t1: Array,
    W_pos: Array,  # shape (D, D, K)  非負
    beta_pos: Array,  # shape (K,)      非負
    mu_fn: MuFn,
    mu_params: Any,
    *,
    eps: float = 1e-12,
) -> Tuple[Array, Dict[str, Array]]:
    """
    基底積分は別関数で計算して足し合わせる（コアは合計ログ強度とカーネル補正を出す）
    戻り値: (負の対数尤度の一部, 診断情報)
    """

    D, _, K = W_pos.shape
    N = times.shape[0]

    # 減衰メモリ m[r,k] を保持（直前イベント時刻からの減衰）
    m0 = jnp.zeros((D, K), dtype=times.dtype)

    def scan_step(carry, idx):
        last_t, m, loglik = carry
        t = times[idx]
        d = marks[idx]  # イベントの「起こった」次元

        dt = t - last_t
        # 減衰
        m = m * jnp.exp(-beta_pos[None, :] * dt)

        # 強度 λ_d(t) を計算
        # 刺激項: sum_{r,k} W[d,r,k] * β_k * m[r,k]
        excite = jnp.sum(W_pos[d, :, :] * (beta_pos[None, :] * m), axis=(0, 1))
        lam = mu_fn(d, t, mu_params) + excite
        lam = jnp.clip(lam, a_min=eps)  # 数値安定化（baseline が0でもOK）

        loglik = loglik + jnp.log(lam)

        # 新イベントをメモリに追加
        m = m.at[d, :].add(1.0)

        return (t, m, loglik), None

    # scan（N=0 でも問題なく初期値が返る）
    (last_t, m, sum_loglam), _ = lax.scan(
        scan_step,
        (t0, m0, jnp.array(0.0, dtype=times.dtype)),
        jnp.arange(N),
    )
    # 観測終了まで減衰
    m_T = m * jnp.exp(-beta_pos[None, :] * (t1 - last_t))

    # N_r（各次元のイベント数）
    N_r = jnp.bincount(marks, length=D)

    # カーネル補正: Σ_{d,r,k} W[d,r,k] * (N_r - m_T[r,k])
    tmp = N_r[:, None] - m_T  # shape (D,K)
    kernel_compensator = jnp.sum(W_pos * tmp[None, :, :])  # (D,D,K) * (1,D,K)

    return -(sum_loglam) + kernel_compensator, {
        "sum_loglam": sum_loglam,
        "kernel_compensator": kernel_compensator,
        "N_r": N_r,
        "m_T": m_T,
    }


_single_sequence_nll_core_jit = jax.jit(_single_sequence_nll_core, static_argnames=("mu_fn",))


# ---------------------------
# 総負対数尤度（複数系列の和）
# ---------------------------
def make_total_nll(
    sequences: List[Sequence],
    mu_fn: MuFn,
    mu_int_fn: MuIntFn,
    D: int,
    K: int,
    *,
    quad_epsabs: float = 1e-6,
    quad_epsrel: float = 1e-6,
    stability_penalty_weight: float = 0.0,  # 0: 無効
):
    """
    params を受け取り, 総負対数尤度（任意で安定性ペナルティ付）を返す関数を生成.
    params 構造:
        {
          "W_uncon": (D,D,K),
          "beta_uncon": (K,),
          "mu_params": pytree（mu_fn が解釈）
        }
    非負化: softplus を適用して W_pos, beta_pos を使用.
    """

    def total_nll(params: Dict[str, Any]) -> Array:
        W_pos = jax.nn.softplus(params["W_uncon"])
        beta_pos = jax.nn.softplus(params["beta_uncon"]) + 1e-6  # β>0 を確実に

        nll_sum = jnp.array(0.0)
        # baseline 積分のトータル
        base_int_sum = jnp.array(0.0)

        for seq in sequences:
            times, marks = _merge_and_sort_events(seq.events_by_dim)
            # 片系列コア
            nll_core, _diag = _single_sequence_nll_core_jit(
                times,
                marks,
                jnp.array(seq.t0),
                jnp.array(seq.t1),
                W_pos,
                beta_pos,
                mu_fn,
                params["mu_params"],
            )
            nll_sum = nll_sum + nll_core
            # baseline 積分
            base_int = _baseline_integral_sum_over_dims(
                mu_fn,
                mu_int_fn,
                params["mu_params"],
                D,
                jnp.array(seq.t0),
                jnp.array(seq.t1),
                quad_epsabs=quad_epsabs,
                quad_epsrel=quad_epsrel,
            )
            base_int_sum = base_int_sum + base_int

        # 安定性の緩い正則化（∞ノルム上界で 1 未満を促す）
        penalty = jnp.array(0.0)
        if stability_penalty_weight > 0.0:
            A = jnp.sum(W_pos, axis=2)  # (D,D) = Σ_k W[:,:,k]
            row_sums = jnp.sum(jnp.abs(A), axis=1)
            inf_norm = jnp.max(row_sums)
            # max(0, ∥A∥_∞ - 1 + margin)^2
            margin = 1e-3
            penalty = stability_penalty_weight * jnp.square(jnp.maximum(0.0, inf_norm - (1.0 - margin)))

        return nll_sum + base_int_sum + penalty

    return total_nll


# ---------------------------
# 例: 区分一様 baseline の helper（任意）
# ---------------------------
@dataclass
class PiecewiseConstBaseline:
    # 共通ビン境界（昇順, shape (B+1,)）
    edges: Array
    # 各次元の値（非負にしたい場合は softplus を適用）
    values_uncon: Array  # shape (D,B)


def pc_mu_fn(d: int, t: Array, params: PiecewiseConstBaseline) -> Array:
    edges = params.edges
    vals = jax.nn.softplus(params.values_uncon)[d]  # 非負化
    # bin index
    idx = jnp.clip(jnp.searchsorted(edges, t, side="right") - 1, 0, edges.shape[0] - 2)
    return vals[idx]


def pc_mu_int_fn(d: int, t0: Array, t1: Array, params: PiecewiseConstBaseline) -> Array:
    edges = params.edges
    vals = jax.nn.softplus(params.values_uncon)[d]  # (B,)
    # 各ビンの区間 [edges[b], edges[b+1]] と [t0,t1] の重なり長を足し合わせる
    left = edges[:-1]
    right = edges[1:]
    over_len = jnp.clip(jnp.minimum(right, t1) - jnp.maximum(left, t0), a_min=0.0)
    return jnp.sum(over_len * vals)


# ---------------------------
# 学習ループ（optax.adam の例）
# ---------------------------
@dataclass
class FitConfig:
    lr: float = 1e-2
    steps: int = 1000
    quad_epsabs: float = 1e-6
    quad_epsrel: float = 1e-6
    stability_penalty_weight: float = 0.0
    log_every: int = 100


def fit_hawkes_mixture_exp(
    sequences: List[Sequence],
    D: int,
    K: int,
    mu_fn: MuFn,
    mu_int_fn: MuIntFn,
    init_params: Dict[str, Any],
    cfg: FitConfig = FitConfig(),
) -> Dict[str, Any]:
    """
    MLE を Adam で最適化（必要に応じて jaxopt の L-BFGS 等に置換可）
    """

    total_nll = make_total_nll(
        sequences,
        mu_fn,
        mu_int_fn,
        D,
        K,
        quad_epsabs=cfg.quad_epsabs,
        quad_epsrel=cfg.quad_epsrel,
        stability_penalty_weight=cfg.stability_penalty_weight,
    )

    params = init_params
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(total_nll)(params)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for it in range(cfg.steps):
        params, opt_state, loss = step(params, opt_state)
        if (it % cfg.log_every) == 0 or (it == cfg.steps - 1):
            # print は JIT 外で
            print(f"[{it:05d}] nll={float(loss):.6f}")

    return params


# ---------------------------
# 使い方イメージ（最小例）
# ---------------------------
if __name__ == "__main__":
    # ダミーデータ: D=2 次元, 2系列
    seqs = [
        Sequence(events_by_dim=[jnp.array([0.3, 0.9, 1.2]), jnp.array([0.5, 1.1])], t0=0.0, t1=2.0),
        Sequence(events_by_dim=[jnp.array([0.2, 0.4, 1.5]), jnp.array([0.7])], t0=0.0, t1=2.0),
    ]
    D, K = 2, 3

    # baseline を区分一様で持たせる例（B=4）
    edges = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    mu_params = PiecewiseConstBaseline(
        edges=edges,
        values_uncon=jnp.zeros((D, edges.shape[0] - 1)),  # softplus 後に初期値 ~0
    )

    # 初期パラメータ
    init = {
        "W_uncon": jnp.full((D, D, K), -3.0),  # softplus(-3) ~ 0.05
        "beta_uncon": jnp.zeros((K,)),  # softplus(0)=~0.693 → β~0.693
        "mu_params": mu_params,
    }

    cfg = FitConfig(lr=1e-2, steps=500, log_every=100)

    # 学習
    learned = fit_hawkes_mixture_exp(
        sequences=seqs,
        D=D,
        K=K,
        mu_fn=pc_mu_fn,  # 閉形式積分あり
        mu_int_fn=pc_mu_int_fn,  # ←無ければ None にして quadax にフォールバック
        init_params=init,
        cfg=cfg,
    )

    # 学習後パラメータ（非負化後）を取り出す例
    W_hat = jax.nn.softplus(learned["W_uncon"])
    beta_hat = jax.nn.softplus(learned["beta_uncon"]) + 1e-6
    print("W_hat:", W_hat)
    print("beta_hat:", beta_hat)
