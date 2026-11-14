"""Recursos compartilhados pelas páginas: dados, dicionário de variáveis, hipóteses e gráficos."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from estat import resumo_quantitativo, tabela_frequencias

PASTA_DADOS = Path(__file__).parent / "data"

# cores (paleta validada: slot 1 = azul, slot 2 = laranja)
COR_POP = "#2a78d6"
COR_AMO = "#eb6834"
COR_REF = "#52514e"     # linha de referência (H0)
COR_TEXTO = "#0b0b0b"
COR_GRADE = "#e6e5e0"

FONTE_NOME = "SINASC – Sistema de Informações sobre Nascidos Vivos (Ministério da Saúde/DATASUS)"
FONTE_LINK = "https://datasus.saude.gov.br/transferencia-de-arquivos/"
FONTE_FTP = "ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/NOV/DNRES/"
FONTE_DOC = "https://opendatasus.saude.gov.br/dataset/sistema-de-informacao-sobre-nascidos-vivos-sinasc"

# ------------------------------------------------------------------ dicionário de variáveis
QUANTITATIVAS = {
    "peso_g": ("Peso ao nascer", "g", "Peso do recém-nascido em gramas (PESO)."),
    "semanas_gestacao": ("Duração da gestação", "semanas", "Semanas de gestação (SEMAGESTAC)."),
    "idade_mae": ("Idade da mãe", "anos", "Idade da mãe em anos (IDADEMAE)."),
    "consultas_prenatal": ("Consultas de pré-natal", "consultas",
                           "Número de consultas de pré-natal (CONSPRENAT)."),
    "apgar5": ("Apgar no 5º minuto", "pontos", "Índice de Apgar no 5º minuto, de 0 a 10 (APGAR5)."),
}
QUALITATIVAS = {
    "tipo_parto": ("Tipo de parto", "Nominal: vaginal ou cesáreo (PARTO)."),
    "baixo_peso": ("Baixo peso ao nascer", "Nominal derivada: peso < 2.500 g (critério da OMS)."),
    "prematuro": ("Prematuridade", "Nominal derivada: gestação < 37 semanas (critério da OMS)."),
    "sexo": ("Sexo do recém-nascido", "Nominal (SEXO)."),
    "raca_cor": ("Raça/cor do recém-nascido", "Nominal (RACACOR)."),
    "escolaridade_mae": ("Escolaridade da mãe", "Ordinal, em anos de estudo (ESCMAE)."),
    "faixa_idade_mae": ("Faixa etária da mãe", "Ordinal derivada da idade da mãe."),
    "faixa_consultas": ("Faixa de consultas de pré-natal", "Ordinal (CONSULTAS)."),
    "estado_civil_mae": ("Estado civil da mãe", "Nominal (ESTCIVMAE)."),
    "tipo_gravidez": ("Tipo de gravidez", "Nominal: única, dupla, tripla ou mais (GRAVIDEZ)."),
}


def rotulo(var: str) -> str:
    if var in QUANTITATIVAS:
        nome, unid, _ = QUANTITATIVAS[var]
        return f"{nome} ({unid})"
    return QUALITATIVAS[var][0]


# ------------------------------------------------------------------ hipóteses
# Os valores de referência foram definidos ANTES de olhar os dados, a partir de fontes externas.
HIPOTESES = [
    {
        "id": "P1", "tipo": "proporcao", "titulo": "Proporção de cesáreas",
        "variavel": "tipo_parto", "evento": "Cesáreo", "p0": 0.60, "alternativa": "greater",
        "pergunta": "A proporção de nascimentos por cesárea no RS é maior que a do Brasil (cerca de 60%)?",
        "h0": "p = 0,60", "h1": "p > 0,60",
        "parametro": "p = proporção de nascidos vivos de mães residentes no RS em 2024 cujo parto foi cesáreo",
        "justificativa": (
            "O Brasil é um dos países com maior taxa de cesáreas do mundo: em 2024 a taxa nacional passou de 60%, "
            "enquanto a OMS considera ideal algo entre 10% e 15%. O teste é unilateral à direita porque a pergunta "
            "de interesse é se o RS está **acima** do patamar nacional."),
    },
    {
        "id": "P2", "tipo": "proporcao", "titulo": "Proporção de baixo peso ao nascer",
        "variavel": "baixo_peso", "evento": "Sim", "p0": 0.09, "alternativa": "two-sided",
        "pergunta": "A proporção de recém-nascidos com baixo peso (< 2.500 g) no RS difere de 9%?",
        "h0": "p = 0,09", "h1": "p ≠ 0,09",
        "parametro": "p = proporção de nascidos vivos de mães residentes no RS em 2024 com peso < 2.500 g",
        "justificativa": (
            "O baixo peso ao nascer é um dos principais fatores de risco para a mortalidade infantil. Estudos com o "
            "SINASC estimam a prevalência nacional em torno de 9%. Sem expectativa prévia de direção, o teste é "
            "bilateral."),
    },
    {
        "id": "M1", "tipo": "media", "titulo": "Média de consultas de pré-natal",
        "variavel": "consultas_prenatal", "mu0": 8.0, "alternativa": "greater",
        "pergunta": "As gestantes do RS realizam, em média, mais do que as 8 consultas de pré-natal recomendadas pela OMS?",
        "h0": "μ = 8", "h1": "μ > 8",
        "parametro": "μ = número médio de consultas de pré-natal das mães de nascidos vivos residentes no RS em 2024",
        "justificativa": (
            "A OMS recomenda no mínimo 8 contatos de pré-natal (o Ministério da Saúde adota um mínimo de 6 a 7). "
            "Testar se a média supera 8 avalia se o acompanhamento típico no RS atinge o padrão mais exigente."),
    },
    {
        "id": "M2", "tipo": "media", "titulo": "Peso médio ao nascer",
        "variavel": "peso_g", "mu0": 3200.0, "alternativa": "two-sided",
        "pergunta": "O peso médio ao nascer no RS difere de 3.200 g?",
        "h0": "μ = 3.200 g", "h1": "μ ≠ 3.200 g",
        "parametro": "μ = peso médio ao nascer (g) dos nascidos vivos de mães residentes no RS em 2024",
        "justificativa": (
            "3.200 g é um valor de referência usual para o peso médio de recém-nascidos no Brasil. "
            "O teste bilateral verifica se o RS se afasta desse patamar em qualquer direção."),
    },
]

ALTERNATIVA_TXT = {"two-sided": "bilateral", "greater": "unilateral à direita", "less": "unilateral à esquerda"}


# ------------------------------------------------------------------ dados
@st.cache_data
def carregar():
    pop = pd.read_parquet(PASTA_DADOS / "populacao.parquet")
    amo = pd.read_parquet(PASTA_DADOS / "amostra.parquet")
    meta = json.loads((PASTA_DADOS / "meta.json").read_text())
    return pop, amo, meta


# ------------------------------------------------------------------ formatação pt-BR
def fmt(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, casas: int = 1) -> str:
    return fmt(100 * x, casas) + "%"


def fmt_p(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else fmt(p, 4)


def tabela_br(df: pd.DataFrame, casas: int = 2, colunas_pct=()) -> pd.DataFrame:
    """Converte números em texto pt-BR para exibição."""
    out = df.copy()
    for c in out.columns:
        if c in colunas_pct:
            out[c] = out[c].map(lambda v: pct(v, 1))
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: fmt(v, 0) if float(v).is_integer() else fmt(v, casas))
    return out


# ------------------------------------------------------------------ gráficos
def layout(fig: go.Figure, titulo: str = "", altura: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=COR_TEXTO), x=0, xanchor="left"),
        height=altura, margin=dict(l=10, r=10, t=60 if titulo else 20, b=10),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(color=COR_TEXTO, size=12),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0),
        separators=",.", **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", ticks="outside")
    fig.update_yaxes(gridcolor=COR_GRADE, zeroline=False)
    return fig


def histograma(series_por_nome: dict, cores: dict, bins: int, titulo: str, xlabel: str,
               densidade: bool = False, linhas: dict | None = None):
    fig = go.Figure()
    todos = pd.concat([s.dropna() for s in series_por_nome.values()])
    inicio, fim = float(todos.min()), float(todos.max())
    tamanho = (fim - inicio) / bins if fim > inicio else 1
    for nome, s in series_por_nome.items():
        fig.add_histogram(x=s.dropna(), name=nome, marker_color=cores[nome],
                          marker_line=dict(color="white", width=1), opacity=0.75 if len(series_por_nome) > 1 else 1,
                          xbins=dict(start=inicio, end=fim + tamanho, size=tamanho),
                          histnorm="probability density" if densidade else "",
                          hovertemplate="%{x}<br>" + ("densidade: %{y:.4f}" if densidade else "frequência: %{y}")
                          + f"<extra>{nome}</extra>")
    fig.update_layout(barmode="overlay")
    for i, (nome, (valor, cor, dash)) in enumerate((linhas or {}).items()):
        fig.add_vline(x=valor, line=dict(color=cor, dash=dash, width=2),
                      annotation_text=f"{nome} = {fmt(valor, 1)}",
                      annotation_position="top right" if i == 0 else "top left")
    layout(fig, titulo)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="densidade" if densidade else "frequência")
    return fig


def boxplot(series_por_nome: dict, cores: dict, xlabel: str, altura: int = 220):
    fig = go.Figure()
    for nome, s in series_por_nome.items():
        fig.add_box(x=s.dropna(), name=nome, marker_color=cores[nome], boxmean=True,
                    marker=dict(size=3, opacity=0.4), line=dict(width=1.5))
    layout(fig, altura=altura, showlegend=False)
    fig.update_xaxes(title=xlabel, showgrid=True, gridcolor=COR_GRADE)
    return fig


def barras_proporcao(tabelas: dict, cores: dict, titulo: str):
    """tabelas: nome -> DataFrame com coluna 'proporção' (índice = categorias)."""
    fig = go.Figure()
    for nome, tab in tabelas.items():
        cats = [str(c) for c in tab.index]
        fig.add_bar(y=cats, x=tab["proporção"], name=nome, orientation="h", marker_color=cores[nome],
                    text=[pct(v) for v in tab["proporção"]], textposition="outside",
                    customdata=tab["frequência"],
                    hovertemplate="%{y}: %{x:.1%} (n = %{customdata:,})" + f"<extra>{nome}</extra>")
    layout(fig, titulo, altura=max(260, 70 + 45 * len(next(iter(tabelas.values()))) * len(tabelas) ** 0.6))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=COR_GRADE, range=[0, 1.12 * max(
        t["proporção"].max() for t in tabelas.values())])
    return fig


# ------------------------------------------------------------------ painel descritivo (1 e 2)
def kpis(df: pd.DataFrame, ref: pd.DataFrame | None = None):
    def media(d, c):
        return d[c].mean()

    def prop(d, c, ev):
        s = d[c].dropna()
        return (s == ev).mean()

    itens = [
        ("Nascidos vivos (registros)", len(df), None, lambda v: fmt(v, 0)),
        ("Peso médio ao nascer", media(df, "peso_g"), media(ref, "peso_g") if ref is not None else None,
         lambda v: fmt(v, 0) + " g"),
        ("Cesáreas", prop(df, "tipo_parto", "Cesáreo"),
         prop(ref, "tipo_parto", "Cesáreo") if ref is not None else None, pct),
        ("Baixo peso (< 2.500 g)", prop(df, "baixo_peso", "Sim"),
         prop(ref, "baixo_peso", "Sim") if ref is not None else None, pct),
        ("Prematuros (< 37 sem.)", prop(df, "prematuro", "Sim"),
         prop(ref, "prematuro", "Sim") if ref is not None else None, pct),
        ("Idade média da mãe", media(df, "idade_mae"), media(ref, "idade_mae") if ref is not None else None,
         lambda v: fmt(v, 1) + " anos"),
    ]
    cols = st.columns(len(itens))
    for col, (nome, v, r, f) in zip(cols, itens):
        delta = None
        if r is not None:
            d = v - r
            delta = ("+" if d >= 0 else "") + (pct(d, 2).replace("%", " p.p.") if f is pct else fmt(d, 1))
        col.metric(nome, f(v), delta=delta, delta_color="off",
                   help="O valor menor indica a diferença amostra − população." if delta else None)


def painel_descritivo(df: pd.DataFrame, nome: str, cor: str, ref: pd.DataFrame | None = None):
    """Painel descritivo genérico. Se `ref` (a população) for passado, mostra comparações."""
    comparar = ref is not None
    nome_ref = "População"
    cores = {nome: cor, nome_ref: COR_POP}

    kpis(df, ref)
    st.divider()

    aba_quant, aba_qual, aba_cruz = st.tabs(
        ["Variáveis quantitativas", "Variáveis qualitativas", "Relações entre variáveis"])

    # ---------------- quantitativas
    with aba_quant:
        st.markdown("##### Medidas descritivas de todas as variáveis quantitativas")
        resumo = pd.DataFrame({rotulo(v): resumo_quantitativo(df[v]) for v in QUANTITATIVAS}).T
        st.dataframe(tabela_br(resumo), width="stretch")

        st.markdown("##### Distribuição de uma variável")
        c1, c2, c3 = st.columns([2, 1, 1])
        var = c1.selectbox("Variável", list(QUANTITATIVAS), format_func=rotulo, key=f"q_{nome}")
        bins = c2.slider("Número de classes (bins)", 5, 80, 30, key=f"b_{nome}")
        sobrepor = c3.toggle("Sobrepor a população", value=False, key=f"s_{nome}",
                             disabled=not comparar,
                             help="Compara as formas das distribuições (eixo em densidade).")
        x = df[var]
        series = {nome: x}
        if comparar and sobrepor:
            series = {nome_ref: ref[var], nome: x}
        linhas = {"média": (x.mean(), COR_TEXTO, "dash"), "mediana": (x.median(), COR_REF, "dot")}
        st.plotly_chart(histograma(series, cores, bins, f"Histograma – {rotulo(var)}", rotulo(var),
                                   densidade=comparar and sobrepor, linhas=linhas),
                        width="stretch")
        series_box = {nome_ref: ref[var], nome: x} if comparar else {nome: x}
        st.plotly_chart(boxplot(series_box, cores, rotulo(var)), width="stretch")

        r = resumo_quantitativo(x)
        if comparar:
            rr = resumo_quantitativo(ref[var])
            comp = pd.DataFrame({"População": rr, "Amostra": r})
            comp["diferença (amostra − população)"] = comp["Amostra"] - comp["População"]
            st.markdown(f"**{rotulo(var)} – população × amostra**")
            st.dataframe(tabela_br(comp.drop(index=["ausentes"])), width="stretch")
        st.caption(QUANTITATIVAS[var][2] + " Registros com código de “ignorado” foram tratados como ausentes. "
                   f"Assimetria = {fmt(r['assimetria'])}: "
                   + ("cauda mais longa à esquerda." if r["assimetria"] < -0.5 else
                      "cauda mais longa à direita." if r["assimetria"] > 0.5 else "distribuição aproximadamente simétrica."))

    # ---------------- qualitativas
    with aba_qual:
        c1, _ = st.columns([2, 2])
        var = c1.selectbox("Variável", list(QUALITATIVAS), format_func=rotulo, key=f"c_{nome}")
        tab = tabela_frequencias(df[var])
        tabelas = {nome: tab}
        if comparar:
            tab_ref = tabela_frequencias(ref[var])
            tabelas = {nome_ref: tab_ref, nome: tab}
        st.plotly_chart(barras_proporcao(tabelas, cores, f"Distribuição de frequências – {rotulo(var)}"),
                        width="stretch")
        if comparar:
            comp = pd.DataFrame({"freq. população": tab_ref["frequência"], "prop. população": tab_ref["proporção"],
                                 "freq. amostra": tab["frequência"], "prop. amostra": tab["proporção"]})
            comp["diferença (p.p.)"] = (comp["prop. amostra"] - comp["prop. população"]) * 100
            st.dataframe(tabela_br(comp, colunas_pct=("prop. população", "prop. amostra")),
                         width="stretch")
        else:
            st.dataframe(tabela_br(tab, colunas_pct=("proporção", "proporção acumulada")),
                         width="stretch")
        ausentes = int(df[var].isna().sum())
        st.caption(QUALITATIVAS[var][1] + f" Ausentes/ignorados: {fmt(ausentes, 0)} "
                   f"({pct(ausentes / len(df))}), excluídos do cálculo das proporções.")

    # ---------------- cruzamentos
    with aba_cruz:
        c1, c2 = st.columns(2)
        vnum = c1.selectbox("Variável quantitativa", list(QUANTITATIVAS), format_func=rotulo,
                            index=0, key=f"xn_{nome}")
        vcat = c2.selectbox("Agrupar por", list(QUALITATIVAS), format_func=rotulo, index=0, key=f"xc_{nome}")
        fig = go.Figure()
        for cat in df[vcat].cat.categories:
            s = df.loc[df[vcat] == cat, vnum].dropna()
            if len(s):
                fig.add_box(y=s, name=str(cat), marker_color=cor, boxmean=True,
                            marker=dict(size=2, opacity=0.3), line=dict(width=1.5))
        layout(fig, f"{rotulo(vnum)} por {rotulo(vcat).lower()} – {nome.lower()}", altura=420, showlegend=False)
        fig.update_yaxes(title=rotulo(vnum))
        st.plotly_chart(fig, width="stretch")
        g = df.groupby(vcat, observed=True)[vnum].agg(["count", "mean", "median", "std"])
        g.columns = ["n", "média", "mediana", "desvio padrão"]
        st.dataframe(tabela_br(g), width="stretch")
        st.caption("Associação não implica causalidade: diferenças entre grupos podem refletir outros fatores.")
