import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from comum import (ALTERNATIVA_TXT, COR_AMO, COR_GRADE, COR_POP, COR_REF, COR_TEXTO, HIPOTESES, carregar,
                   fmt, fmt_p, layout, pct, rotulo, tabela_br)
from estat import inferencia_media, inferencia_proporcao, simular_amostras

pop, amo, meta = carregar()

st.title("Painel 3 · Inferência estatística e comparação dos resultados")
st.markdown(
    "Os procedimentos de inferência usam **apenas a amostra** "
    f"(n = {fmt(meta['n'], 0)}). Os parâmetros populacionais são calculados diretamente "
    "na população e servem só para **avaliar** as estimativas.")

with st.sidebar:
    st.markdown("### Parâmetros da inferência")
    conf = st.select_slider("Nível de confiança dos ICs", options=[0.90, 0.95, 0.99], value=0.95,
                            format_func=lambda v: pct(v, 0))
    alpha = st.select_slider("Nível de significância (α)", options=[0.01, 0.05, 0.10], value=0.05,
                             format_func=lambda v: fmt(v, 2))


# ------------------------------------------------------------------ cálculos
def evento(df, h):
    s = df[h["variavel"]]
    return (s == h["evento"]).where(s.notna())


def calcular(h, df):
    if h["tipo"] == "proporcao":
        return inferencia_proporcao(evento(df, h), h["p0"], h["alternativa"], conf, alpha)
    return inferencia_media(df[h["variavel"]], h["mu0"], h["alternativa"], conf, alpha)


def parametro_pop(h):
    if h["tipo"] == "proporcao":
        return float(evento(pop, h).dropna().astype(bool).mean())
    return float(pop[h["variavel"]].mean())


def lado_h1(h, valor):
    ref = h["p0"] if h["tipo"] == "proporcao" else h["mu0"]
    return {"two-sided": valor != ref, "greater": valor > ref, "less": valor < ref}[h["alternativa"]]


def f_val(h, v):
    if h["tipo"] == "proporcao":
        return pct(v, 2)
    return fmt(v, 0 if h["variavel"] == "peso_g" else 2) + (" g" if h["variavel"] == "peso_g" else "")


def dif_txt(h, d):
    sinal = "+" if d >= 0 else "−"
    if h["tipo"] == "proporcao":
        return sinal + fmt(abs(100 * d), 2) + " p.p."
    return sinal + fmt(abs(d), 1 if h["variavel"] == "peso_g" else 2) + (" g" if h["variavel"] == "peso_g" else "")


# ------------------------------------------------------------------ gráficos
def grafico_ic(h, r, theta):
    est = r.p_hat if h["tipo"] == "proporcao" else r.media
    ref = h["p0"] if h["tipo"] == "proporcao" else h["mu0"]
    lo, hi = r.ic
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[est], y=["Amostra"], mode="markers", marker=dict(size=13, color=COR_AMO),
        error_x=dict(type="data", symmetric=False, array=[hi - est], arrayminus=[est - lo],
                     color=COR_AMO, thickness=3, width=12),
        name=f"estimativa e IC de {pct(conf, 0)}",
        hovertemplate=f"estimativa: {f_val(h, est)}<br>IC: [{f_val(h, lo)} ; {f_val(h, hi)}]<extra></extra>"))
    fig.add_vline(x=theta, line=dict(color=COR_POP, width=3),
                  annotation_text=f"parâmetro populacional = {f_val(h, theta)}",
                  annotation_position="top left" if theta >= ref else "top right")
    fig.add_vline(x=ref, line=dict(color=COR_REF, width=2, dash="dash"),
                  annotation_text=f"H₀: {f_val(h, ref)}",
                  annotation_position="bottom left" if theta >= ref else "bottom right")
    fig.add_annotation(x=est, y=0, yshift=22, text=f"amostra: IC de {pct(conf, 0)}", showarrow=False,
                       font=dict(color=COR_AMO))
    xs = [lo, hi, theta, ref]
    folga = (max(xs) - min(xs)) * 0.25
    layout(fig, "Intervalo de confiança × parâmetro populacional × valor de H₀", altura=260,
           showlegend=False)
    fig.update_xaxes(range=[min(xs) - folga, max(xs) + folga], showgrid=True, gridcolor=COR_GRADE,
                     tickformat=".1%" if h["tipo"] == "proporcao" else None)
    fig.update_yaxes(showticklabels=False)
    return fig


def grafico_teste(h, r):
    if h["tipo"] == "proporcao":
        dist, est, nome = stats.norm(), r.z, "z"
    else:
        dist, est, nome = stats.t(df=r.gl), r.t, "t"
    alt = h["alternativa"]
    lim = max(4.0, min(abs(est) + 1, 8.0))
    xs = np.linspace(-lim, lim, 600)
    ys = dist.pdf(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=COR_TEXTO, width=2),
                             name=f"distribuição de {nome} sob H₀", hoverinfo="skip"))
    if alt == "two-sided":
        c = dist.ppf(1 - alpha / 2)
        regioes = [(xs <= -c), (xs >= c)]
    elif alt == "greater":
        c = dist.ppf(1 - alpha)
        regioes = [(xs >= c)]
    else:
        c = dist.ppf(alpha)
        regioes = [(xs <= c)]
    for i, m in enumerate(regioes):
        fig.add_trace(go.Scatter(x=xs[m], y=ys[m], fill="tozeroy", mode="none",
                                 fillcolor="rgba(227,73,72,0.35)", name=f"região de rejeição (α = {fmt(alpha, 2)})",
                                 showlegend=i == 0, hoverinfo="skip"))
    x_marca = float(np.clip(est, -lim, lim))
    texto = f"{nome} observado = {fmt(est, 2)}" + ("  (fora da escala →)" if abs(est) > lim else "")
    fig.add_vline(x=x_marca, line=dict(color=COR_AMO, width=3), annotation_text=texto,
                  annotation_position="top left" if est > 0 else "top right")
    layout(fig, f"Estatística de teste na distribuição de referência ({ALTERNATIVA_TXT[alt]})", altura=320)
    fig.update_yaxes(title="densidade")
    fig.update_xaxes(title=nome)
    return fig


def grafico_qq(x):
    x = pd.Series(x).dropna().astype(float)
    (osm, osr), (slope, inter, _) = stats.probplot(x, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(size=5, color=COR_AMO, opacity=0.6),
                             name="amostra", hovertemplate="teórico %{x:.2f}<br>observado %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[inter + slope * osm.min(), inter + slope * osm.max()],
                             mode="lines", line=dict(color=COR_TEXTO, dash="dash"), name="referência normal"))
    layout(fig, "Q-Q plot normal da amostra", altura=320, showlegend=False)
    fig.update_xaxes(title="quantis teóricos")
    fig.update_yaxes(title="quantis observados")
    return fig


def grafico_hist_amostra(x, titulo):
    fig = go.Figure(go.Histogram(x=pd.Series(x).dropna(), nbinsx=30, marker_color=COR_AMO,
                                 marker_line=dict(color="white", width=1)))
    layout(fig, titulo, altura=320, showlegend=False)
    return fig


# ------------------------------------------------------------------ texto de interpretação
def interpretacao(h, r, theta):
    est = r.p_hat if h["tipo"] == "proporcao" else r.media
    lo, hi = r.ic
    contem = lo <= theta <= hi
    erro = est - theta
    decisao = (f"Como o valor-p ({fmt_p(r.p_valor)}) é **menor** que α = {fmt(alpha, 2)}, **rejeitamos H₀**. "
               "Há evidência estatística a favor de H₁: " + h["h1"] + "."
               if r.rejeita else
               f"Como o valor-p ({fmt_p(r.p_valor)}) é **maior ou igual** a α = {fmt(alpha, 2)}, **não "
               "rejeitamos H₀**. A amostra não traz evidência suficiente contra " + h["h0"]
               + ". Isso **não prova** que H₀ é verdadeira.")
    txt = [
        f"**Estimativa:** com {pct(conf, 0)} de confiança, o parâmetro está entre **{f_val(h, lo)}** e "
        f"**{f_val(h, hi)}**, com estimativa pontual de **{f_val(h, est)}**. Se repetíssemos o sorteio muitas "
        f"vezes, cerca de {pct(conf, 0)} dos intervalos construídos assim conteriam o parâmetro verdadeiro.",
        "**Teste:** " + decisao,
        f"**Comparação com a população:** o parâmetro populacional é **{f_val(h, theta)}**. O erro amostral "
        f"(estimativa − parâmetro) foi de **{dif_txt(h, erro)}**, e o IC "
        + ("**contém** o parâmetro, ou seja, este intervalo está entre os que “acertam”."
           if contem else "**não contém** o parâmetro: esta amostra caiu entre os casos (cerca de "
           f"{pct(1 - conf, 0)}) em que o intervalo não captura o valor verdadeiro."),
    ]
    na_h1 = lado_h1(h, theta)
    if r.rejeita and na_h1:
        txt.append("Na população, o parâmetro de fato está do lado de H₁, então a decisão do teste foi **correta**.")
    elif r.rejeita and not na_h1:
        txt.append("Na população, o parâmetro **não** está do lado de H₁: a amostra levou a um **erro do tipo I**.")
    elif not r.rejeita and na_h1:
        ref = h["p0"] if h["tipo"] == "proporcao" else h["mu0"]
        txt.append(f"Na população, o parâmetro ({f_val(h, theta)}) está do lado de H₁, mas a diferença em relação "
                   f"a {f_val(h, ref)} é pequena demais para ser detectada com n = {fmt(r.n, 0)} e α = "
                   f"{fmt(alpha, 2)}: isso é um **erro do tipo II**, consequência do poder limitado do teste.")
    else:
        txt.append("Na população, o parâmetro também não está do lado de H₁. Não rejeitar H₀ foi a decisão correta.")
    return txt


# ------------------------------------------------------------------ uma hipótese
def mostrar(h):
    r = calcular(h, amo)
    theta = parametro_pop(h)
    eh_prop = h["tipo"] == "proporcao"

    st.subheader(h["pergunta"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Parâmetro:** {h['parametro']}.")
        if eh_prop:
            st.markdown(f"**Variável qualitativa:** {rotulo(h['variavel'])} · **evento de interesse:** "
                        f"“{h['evento']}”")
        else:
            st.markdown(f"**Variável quantitativa:** {rotulo(h['variavel'])}")
        st.markdown(f"**Hipóteses:** H₀: {h['h0']}  ×  H₁: {h['h1']}. Teste {ALTERNATIVA_TXT[h['alternativa']]}, "
                    f"α = {fmt(alpha, 2)}.")
        st.markdown("**Por que esta hipótese?** " + h["justificativa"])
    with c2:
        st.markdown("**Verificação das condições**")
        cond = [("Independência", True, f"AAS sem reposição e n = {fmt(r.n, 0)} < 10% de N "
                 f"({pct(r.n / meta['N'], 2)})")]
        if eh_prop:
            cond += [
                ("Sucesso/falha no IC", r.cond_ic, f"n·p̂ = {fmt(r.x, 0)} e n·(1−p̂) = {fmt(r.n - r.x, 0)} (≥ 10)"),
                ("Sucesso/falha no teste", r.cond_teste,
                 f"n·p₀ = {fmt(r.n * h['p0'], 1)} e n·(1−p₀) = {fmt(r.n * (1 - h['p0']), 1)} (≥ 10)"),
            ]
        else:
            cond += [
                ("Amostra grande (TCL)", r.n >= 30, f"n = {fmt(r.n, 0)} ≥ 30"),
                ("Assimetria moderada", abs(r.assimetria) < 2,
                 f"assimetria = {fmt(r.assimetria, 2)}; com n grande, a média amostral é aproximadamente normal"),
            ]
        for nome, ok, det in cond:
            st.markdown(f"{'✅' if ok else '⚠️'} **{nome}:** {det}")
        if eh_prop:
            st.caption(f"Conferência pelo IC de Wilson: [{pct(r.ic_wilson[0], 2)} ; {pct(r.ic_wilson[1], 2)}].")

    st.markdown("---")
    est = r.p_hat if eh_prop else r.media
    m1, m2, m3, m4 = st.columns([1, 1.6, 1, 1])
    m1.metric("Estimativa pontual" + (" (p̂)" if eh_prop else " (x̄)"), f_val(h, est))
    m2.metric(f"IC de {pct(conf, 0)}", f"{f_val(h, r.ic[0])} a {f_val(h, r.ic[1])}")
    m3.metric("Estatística de teste", ("z = " if eh_prop else "t = ") + fmt(r.z if eh_prop else r.t, 3))
    m4.metric("Valor-p", fmt_p(r.p_valor))
    msg = (f"**Decisão ao nível α = {fmt(alpha, 2)}: {'rejeitar H₀' if r.rejeita else 'não rejeitar H₀'}** "
           f"(valor-p {'<' if r.rejeita else '≥'} α)")
    st.info(msg, icon=":material/gavel:")

    with st.expander("Ver os cálculos"):
        if eh_prop:
            st.latex(rf"\hat p = \frac{{{r.x}}}{{{r.n}}} = {r.p_hat:.4f} \qquad "
                     rf"EP = \sqrt{{\frac{{\hat p(1-\hat p)}}{{n}}}} = {r.ep:.4f}")
            st.latex(rf"IC_{{{int(conf * 100)}\%}} = \hat p \pm z^* EP = {r.p_hat:.4f} \pm {r.z_crit:.3f}"
                     rf"\times {r.ep:.4f} = [{r.ic[0]:.4f};\ {r.ic[1]:.4f}]")
            st.latex(rf"z = \frac{{\hat p - p_0}}{{\sqrt{{p_0(1-p_0)/n}}}} = \frac{{{r.p_hat:.4f} - {h['p0']}}}"
                     rf"{{{r.ep0:.4f}}} = {r.z:.3f}")
            st.caption("No teste, o erro padrão usa p₀ (o valor sob H₀); no IC, usa p̂.")
        else:
            st.latex(rf"\bar x = {r.media:.3f} \qquad s = {r.dp:.3f} \qquad EP = \frac{{s}}{{\sqrt n}} = "
                     rf"\frac{{{r.dp:.3f}}}{{\sqrt{{{r.n}}}}} = {r.ep:.3f}")
            st.latex(rf"IC_{{{int(conf * 100)}\%}} = \bar x \pm t^*_{{{r.gl}}}\, EP = {r.media:.3f} \pm "
                     rf"{r.t_crit:.3f}\times {r.ep:.3f} = [{r.ic[0]:.3f};\ {r.ic[1]:.3f}]")
            st.latex(rf"t = \frac{{\bar x - \mu_0}}{{EP}} = \frac{{{r.media:.3f} - {h['mu0']:g}}}{{{r.ep:.3f}}} "
                     rf"= {r.t:.3f} \quad (gl = {r.gl})")
            st.caption("σ é desconhecido e estimado por s (ddof = 1), por isso a distribuição t com n − 1 g.l. "
                       f"Tamanho de efeito (d de Cohen) = {fmt(r.d_cohen, 2)}.")

    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_ic(h, r, theta), width="stretch")
    g2.plotly_chart(grafico_teste(h, r), width="stretch")
    if not eh_prop:
        g3, g4 = st.columns(2)
        g3.plotly_chart(grafico_hist_amostra(amo[h["variavel"]], f"Distribuição na amostra – {rotulo(h['variavel'])}"),
                        width="stretch")
        g4.plotly_chart(grafico_qq(amo[h["variavel"]]), width="stretch")

    st.markdown("#### Interpretação e comparação com a população")
    comp = pd.DataFrame({
        "": ["Parâmetro populacional", "Estimativa amostral", "Erro amostral", "IC contém o parâmetro?",
             "Parâmetro está do lado de H₁?", "Decisão do teste"],
        "valor": [f_val(h, theta), f_val(h, est),
                  dif_txt(h, est - theta),
                  "Sim" if r.ic[0] <= theta <= r.ic[1] else "Não", "Sim" if lado_h1(h, theta) else "Não",
                  "Rejeitar H₀" if r.rejeita else "Não rejeitar H₀"],
    })
    c1, c2 = st.columns([1, 2])
    c1.dataframe(comp, hide_index=True, width="stretch")
    with c2:
        for t in interpretacao(h, r, theta):
            st.markdown(t)


# ------------------------------------------------------------------ síntese e variabilidade
@st.cache_data(show_spinner="Sorteando amostras…")
def simular(hid, reps, seed, conf, alpha):
    h = next(x for x in HIPOTESES if x["id"] == hid)
    if h["tipo"] == "proporcao":
        base = evento(pop, h).dropna().astype(bool).reset_index(drop=True)

        def f(s):
            r = inferencia_proporcao(s, h["p0"], h["alternativa"], conf, alpha)
            return {"estimativa": r.p_hat, "inf": r.ic[0], "sup": r.ic[1], "rejeita": r.rejeita}
    else:
        base = pop[h["variavel"]].dropna().reset_index(drop=True)

        def f(s):
            r = inferencia_media(s, h["mu0"], h["alternativa"], conf, alpha)
            return {"estimativa": r.media, "inf": r.ic[0], "sup": r.ic[1], "rejeita": r.rejeita}
    n_valido = int(round(meta["n"] * len(base) / meta["N"]))   # mesmo tamanho efetivo da amostra real
    return simular_amostras(base, n_valido, f, reps=reps, seed=seed), n_valido


def sintese():
    st.subheader("Resumo das quatro análises")
    linhas = []
    for h in HIPOTESES:
        r = calcular(h, amo)
        theta = parametro_pop(h)
        eh_prop = h["tipo"] == "proporcao"
        est = r.p_hat if eh_prop else r.media
        linhas.append({
            "hipótese": f"{h['id']} · {h['titulo']}", "H₀ × H₁": f"{h['h0']} × {h['h1']}",
            "estimativa": f_val(h, est), f"IC {pct(conf, 0)}": f"{f_val(h, r.ic[0])} a {f_val(h, r.ic[1])}",
            "estatística": ("z = " if eh_prop else "t = ") + fmt(r.z if eh_prop else r.t, 2),
            "valor-p": fmt_p(r.p_valor), "decisão": "Rejeita H₀" if r.rejeita else "Não rejeita H₀",
            "parâmetro (pop.)": f_val(h, theta),
            "IC contém o parâmetro?": "Sim" if r.ic[0] <= theta <= r.ic[1] else "Não",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

    st.subheader("Efeito da variabilidade amostral")
    st.markdown(
        "A amostra usada nas análises é **uma** entre inúmeras possíveis. Aqui sorteamos muitas outras AAS do "
        "mesmo tamanho na população e repetimos todo o procedimento. Assim dá para ver quanto as estimativas "
        "variam, com que frequência o IC captura o parâmetro (**cobertura**) e com que frequência o teste "
        "rejeita H₀.")
    c1, c2, c3 = st.columns(3)
    hid = c1.selectbox("Análise", [h["id"] for h in HIPOTESES],
                       format_func=lambda i: next(f"{h['id']} · {h['titulo']}" for h in HIPOTESES if h["id"] == i))
    reps = c2.select_slider("Número de amostras simuladas", options=[100, 250, 500, 1000], value=500)
    seed = c3.number_input("Semente da simulação", min_value=1, value=7, step=1)
    h = next(x for x in HIPOTESES if x["id"] == hid)
    sim, n_valido = simular(hid, reps, int(seed), conf, alpha)
    theta = parametro_pop(h)
    ref = h["p0"] if h["tipo"] == "proporcao" else h["mu0"]
    sim["contem"] = (sim["inf"] <= theta) & (theta <= sim["sup"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Média das estimativas", f_val(h, sim["estimativa"].mean()),
              help="Próxima do parâmetro: o estimador é não viesado.")
    k2.metric("Desvio padrão das estimativas (EP empírico)",
              pct(sim["estimativa"].std(), 2) if h["tipo"] == "proporcao" else fmt(sim["estimativa"].std(), 2))
    k3.metric("Cobertura dos ICs", pct(sim["contem"].mean(), 1), help=f"Nominal: {pct(conf, 0)}")
    k4.metric("Proporção de rejeições de H₀", pct(sim["rejeita"].mean(), 1))

    g1, g2 = st.columns(2)
    fig = go.Figure(go.Histogram(x=sim["estimativa"], nbinsx=35, marker_color=COR_AMO,
                                 marker_line=dict(color="white", width=1), name="estimativas"))
    fig.add_vline(x=theta, line=dict(color=COR_POP, width=3), annotation_text="parâmetro populacional",
                  annotation_position="top right" if theta >= ref else "top left")
    fig.add_vline(x=ref, line=dict(color=COR_REF, width=2, dash="dash"), annotation_text="H₀",
                  annotation_position="top left" if theta >= ref else "top right")
    layout(fig, f"Distribuição amostral simulada ({reps} amostras de n = {fmt(n_valido, 0)})", altura=380,
           showlegend=False)
    if h["tipo"] == "proporcao":
        fig.update_xaxes(tickformat=".1%")
    g1.plotly_chart(fig, width="stretch")

    k = sim.head(100).reset_index(drop=True)
    fig = go.Figure()
    for ok, cor, nome in [(True, COR_POP, "contém o parâmetro"), (False, "#e34948", "não contém")]:
        sub = k[k["contem"] == ok]
        xs, ys = [], []
        for i, row in sub.iterrows():
            xs += [row["inf"], row["sup"], None]
            ys += [i, i, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cor, width=2), name=nome))
    fig.add_vline(x=theta, line=dict(color=COR_TEXTO, width=2, dash="dash"))
    layout(fig, f"Os primeiros 100 ICs de {pct(conf, 0)} ({int((~k['contem']).sum())} não contêm o parâmetro)",
           altura=380)
    fig.update_yaxes(title="amostra", showticklabels=False)
    if h["tipo"] == "proporcao":
        fig.update_xaxes(tickformat=".1%")
    g2.plotly_chart(fig, width="stretch")

    na_h1 = lado_h1(h, theta)
    st.markdown(
        f"Neste caso o parâmetro populacional ({f_val(h, theta)}) " +
        ("está do lado de H₁, então a proporção de rejeições estima o **poder** do teste, ou seja, a chance "
         "de detectar a diferença com amostras deste tamanho." if na_h1 else
         "não está do lado de H₁, então a proporção de rejeições estima a taxa de **erro do tipo I**, "
         f"que deve ficar perto de α = {fmt(alpha, 2)} ou abaixo dele."))

    st.subheader("Limitações da análise")
    st.markdown(f"""
- **Uma única amostra.** Todas as conclusões da inferência dependem da amostra sorteada. Outra semente produziria
  outras estimativas, ICs e possivelmente outras decisões, como mostra a simulação acima.
- **Ausentes.** Registros com informação ignorada foram excluídos variável a variável, então o *n* efetivo de cada
  análise fica um pouco abaixo de {fmt(meta['n'], 0)}. Se os ausentes não forem aleatórios, isso pode introduzir viés.
- **Hipóteses pontuais.** Com amostras grandes, até diferenças pequenas e sem relevância prática se tornam
  “significativas”. Por isso olhamos também o IC e o tamanho do efeito, e não só o valor-p.
- **Valores de referência.** Os valores de H₀ vêm de fontes externas (média nacional e recomendações da OMS) e são
  aproximações. Outra referência mudaria o teste, mas não a estimativa.
- **Múltiplos testes.** Com quatro testes a α = {fmt(alpha, 2)}, a chance de ao menos um falso positivo, se todas as H₀
  fossem verdadeiras, seria de cerca de {pct(1 - (1 - alpha) ** 4, 0)}.
- **Qualidade do registro.** O SINASC é um registro administrativo: erros de preenchimento da DNV e
  diferenças entre estabelecimentos afetam a população e a amostra da mesma forma.
- **Associação ≠ causalidade.** As análises descrevem a população de 2024 e não identificam causas.
""")


abas = st.tabs([f"{h['id']} · {h['titulo']}" for h in HIPOTESES] + ["Síntese e variabilidade amostral"])
for aba, h in zip(abas, HIPOTESES):
    with aba:
        mostrar(h)
with abas[-1]:
    sintese()
