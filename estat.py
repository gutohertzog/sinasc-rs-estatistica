"""Funções estatísticas usadas nos painéis.

Seguem as convenções da disciplina (notebooks OpenIntro adaptados):
- IC para média: distribuição t com n-1 g.l. e desvio padrão amostral (ddof=1);
- IC para proporção: Wald (com Wilson como conferência);
- teste para proporção: z com erro padrão calculado sob H0, sqrt(p0(1-p0)/n);
- condição sucesso/falha: n·p >= 10 e n·(1-p) >= 10.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- tamanho amostral
def tamanho_amostra_proporcao(N: int, conf: float = 0.95, E: float = 0.03, p: float = 0.5):
    """Tamanho de amostra para estimar uma proporção.

    n0 = z² p(1-p) / E²       (população infinita)
    n  = n0 / (1 + (n0-1)/N)  (correção para população finita, como na calculadora da USP)
    """
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n0 = z**2 * p * (1 - p) / E**2
    n = n0 / (1 + (n0 - 1) / N)
    return {"z": z, "n0": n0, "n0_arred": math.ceil(n0), "n": n, "n_arred": math.ceil(n)}


# ---------------------------------------------------------------- médias
@dataclass
class ResultadoMedia:
    n: int
    media: float
    dp: float
    ep: float
    gl: int
    t_crit: float
    ic: tuple
    mu0: float
    alternativa: str
    t: float
    p_valor: float
    alpha: float
    rejeita: bool
    assimetria: float
    d_cohen: float


def inferencia_media(x, mu0: float, alternativa: str = "two-sided", conf: float = 0.95,
                     alpha: float = 0.05) -> ResultadoMedia:
    """IC t bilateral para a média e teste t para H0: mu = mu0."""
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    ep = s / np.sqrt(n)
    gl = n - 1
    t_crit = stats.t.ppf(1 - (1 - conf) / 2, df=gl)
    ic = (m - t_crit * ep, m + t_crit * ep)
    t = (m - mu0) / ep
    if alternativa == "two-sided":
        p = 2 * stats.t.sf(abs(t), df=gl)
    elif alternativa == "greater":
        p = stats.t.sf(t, df=gl)
    else:  # less
        p = stats.t.cdf(t, df=gl)
    return ResultadoMedia(n=n, media=m, dp=s, ep=ep, gl=gl, t_crit=t_crit, ic=ic, mu0=mu0,
                          alternativa=alternativa, t=t, p_valor=p, alpha=alpha,
                          rejeita=p < alpha, assimetria=float(x.skew()), d_cohen=(m - mu0) / s)


# ---------------------------------------------------------------- proporções
@dataclass
class ResultadoProporcao:
    n: int
    x: int
    p_hat: float
    ep: float
    z_crit: float
    ic: tuple
    ic_wilson: tuple
    p0: float
    alternativa: str
    ep0: float
    z: float
    p_valor: float
    alpha: float
    rejeita: bool
    cond_ic: bool      # n·p̂ >= 10 e n·(1-p̂) >= 10
    cond_teste: bool   # n·p0 >= 10 e n·(1-p0) >= 10


def wilson_ci(x: int, n: int, conf: float = 0.95):
    p = x / n
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - meia, centro + meia


def inferencia_proporcao(evento, p0: float, alternativa: str = "two-sided", conf: float = 0.95,
                         alpha: float = 0.05) -> ResultadoProporcao:
    """IC de Wald para p e teste z para H0: p = p0 (EP calculado sob H0).

    `evento` é uma série booleana (True = evento de interesse); ausentes são descartados.
    """
    e = pd.Series(evento).dropna().astype(bool)
    n = len(e)
    x = int(e.sum())
    p_hat = x / n
    z_crit = stats.norm.ppf(1 - (1 - conf) / 2)
    ep = np.sqrt(p_hat * (1 - p_hat) / n)
    ic = (p_hat - z_crit * ep, p_hat + z_crit * ep)
    ep0 = np.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / ep0
    if alternativa == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternativa == "greater":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    return ResultadoProporcao(n=n, x=x, p_hat=p_hat, ep=ep, z_crit=z_crit, ic=ic,
                              ic_wilson=wilson_ci(x, n, conf), p0=p0, alternativa=alternativa,
                              ep0=ep0, z=z, p_valor=p, alpha=alpha, rejeita=p < alpha,
                              cond_ic=(x >= 10 and n - x >= 10),
                              cond_teste=(n * p0 >= 10 and n * (1 - p0) >= 10))


# ---------------------------------------------------------------- descritiva
def resumo_quantitativo(x) -> pd.Series:
    x = pd.Series(x)
    v = x.dropna().astype(float)
    q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return pd.Series({
        "n válidos": len(v),
        "ausentes": int(x.isna().sum()),
        "média": v.mean(),
        "mediana": q2,
        "moda": v.mode().iloc[0] if len(v) else np.nan,
        "desvio padrão": v.std(ddof=1),
        "variância": v.var(ddof=1),
        "coef. de variação (%)": v.std(ddof=1) / v.mean() * 100,
        "mínimo": v.min(),
        "Q1": q1,
        "Q3": q3,
        "máximo": v.max(),
        "amplitude interquartil (IQR)": iqr,
        "assimetria": v.skew(),
        "curtose (excesso)": v.kurt(),
        "atípicos (1,5·IQR)": int(((v < lim_inf) | (v > lim_sup)).sum()),
    })


def tabela_frequencias(x) -> pd.DataFrame:
    s = pd.Series(x)
    freq = s.value_counts(sort=False, dropna=True)
    if not isinstance(s.dtype, pd.CategoricalDtype):
        freq = freq.sort_values(ascending=False)
    tab = pd.DataFrame({"frequência": freq})
    tab["proporção"] = tab["frequência"] / tab["frequência"].sum()
    if isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered:
        tab["proporção acumulada"] = tab["proporção"].cumsum()
    tab.index.name = s.name
    return tab


# ---------------------------------------------------------------- variabilidade amostral
def simular_amostras(populacao: pd.Series, n: int, estatistica, reps: int = 500, seed: int = 1):
    """Aplica `estatistica` (função de uma Series -> dict) em `reps` AAS de tamanho n."""
    rng = np.random.default_rng(seed)
    valores = populacao.to_numpy()
    N = len(valores)
    linhas = []
    for _ in range(reps):
        idx = rng.choice(N, size=n, replace=False)
        linhas.append(estatistica(pd.Series(valores[idx])))
    return pd.DataFrame(linhas)
