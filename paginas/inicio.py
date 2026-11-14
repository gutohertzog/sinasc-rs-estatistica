import pandas as pd
import streamlit as st

from comum import (FONTE_DOC, FONTE_FTP, FONTE_LINK, FONTE_NOME, QUALITATIVAS, QUANTITATIVAS, carregar,
                   fmt, pct)
from estat import tamanho_amostra_proporcao

pop, amo, meta = carregar()

st.title("Nascidos vivos no Rio Grande do Sul (2024)")
st.markdown("Comparação entre **população** e **amostra aleatória simples**: estimação por intervalos de "
            "confiança e testes de hipóteses.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observações (população)", fmt(meta["N"], 0))
c2.metric("Variáveis analisadas", len(QUANTITATIVAS) + len(QUALITATIVAS))
c3.metric("Tamanho da amostra", fmt(meta["n"], 0))
c4.metric("Fração amostral", pct(meta["fracao_amostral"], 2))

st.header("1. A base de dados")
st.markdown(f"""
**Fonte:** {FONTE_NOME}.
- Transferência de arquivos do DATASUS: <{FONTE_LINK}> (Fonte *SINASC* → Modalidade *Dados* → Tipo *DN* → Ano *2024* → UF *RS*)
- Arquivo utilizado: `{meta['arquivo_origem']}` (MD5 `{meta['md5_origem']}`), no diretório FTP `{FONTE_FTP}`
- Documentação e dicionário de dados: <{FONTE_DOC}>

**Contexto.** O SINASC registra **todos os nascidos vivos do país** a partir da *Declaração de Nascido Vivo (DNV)*,
documento obrigatório preenchido pelo estabelecimento de saúde. A base existe para subsidiar a vigilância e o
planejamento em saúde materno-infantil: cobertura do pré-natal, tipo de parto, prematuridade, peso ao nascer e
perfil das mães. Os microdados são públicos e anonimizados.

- **População de interesse:** os nascidos vivos de mães **residentes no Rio Grande do Sul** em **2024**
  (arquivo *DNRES*, organizado por UF de residência da mãe).
- **Unidade amostral (e de observação):** uma Declaração de Nascido Vivo, ou seja, **um nascido vivo**.
- **Tamanho:** {fmt(meta['N'], 0)} registros. O arquivo original tem mais de 60 colunas, das quais
  {len(QUANTITATIVAS) + len(QUALITATIVAS)} foram selecionadas (abaixo).
- **Tratamento:** códigos de “ignorado” (9, 99) e valores fisiologicamente implausíveis viraram ausentes. Cada
  análise usa os registros válidos daquela variável.
""")

with st.expander("Variáveis selecionadas", expanded=True):
    linhas = [{"variável": v, "descrição": d, "tipo": "Quantitativa " + ("discreta" if v in
              ("consultas_prenatal", "apgar5", "idade_mae", "semanas_gestacao") else "contínua"),
               "unidade": u, "ausentes (pop.)": pct(pop[v].isna().mean(), 2)}
              for v, (_, u, d) in QUANTITATIVAS.items()]
    linhas += [{"variável": v, "descrição": d, "tipo": "Qualitativa " + ("ordinal" if "Ordinal" in d
               else "nominal"), "unidade": "–", "ausentes (pop.)": pct(pop[v].isna().mean(), 2)}
               for v, (_, d) in QUALITATIVAS.items()]
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

st.header("2. Dimensionamento da amostra")
st.markdown(r"""
O tamanho foi calculado para estimar **uma proporção**, pela mesma fórmula da calculadora da USP/Bauru, com
correção para população finita:
""")
st.latex(r"n_0 = \frac{z_{\alpha/2}^2\, p(1-p)}{E^2} \qquad\qquad n = \frac{n_0}{1 + \dfrac{n_0 - 1}{N}}")
st.markdown(f"""
| Parâmetro | Valor | Justificativa |
|---|---|---|
| Nível de confiança | {pct(meta['confianca'], 0)} (z = {fmt(meta['z'], 3)}) | Padrão em saúde pública e o mesmo usado nas aulas. |
| Margem de erro (E) | ± {fmt(100 * meta['margem_erro'], 0)} pontos percentuais | Precisão usual em pesquisas de opinião (Gallup: ±3 a 5%), suficiente para distinguir os patamares de referência das hipóteses. |
| Proporção de planejamento (p) | {fmt(meta['p_planejamento'], 2)} | Pior caso: maximiza p(1−p). Assim a margem fica garantida para **qualquer** proporção estudada. |
| Tamanho da população (N) | {fmt(meta['N'], 0)} | Total de registros da base. |

**Resultado:** n₀ = {fmt(meta['n0'], 1)} e, com a correção, n = {fmt(meta['n_corrigido'], 1)}, arredondado para
cima: **n = {fmt(meta['n'], 0)}**. Como N é grande, a correção quase não altera o resultado.
""")

st.markdown("**Sensibilidade:** quanto custa mais precisão?")
sens = []
for E in (0.05, 0.03, 0.02, 0.01):
    linha = {"margem de erro": f"± {fmt(100 * E, 0)} p.p."}
    for c in (0.90, 0.95, 0.99):
        linha[f"{int(c * 100)}% de confiança"] = fmt(tamanho_amostra_proporcao(meta["N"], c, E, 0.5)["n_arred"], 0)
    sens.append(linha)
st.dataframe(pd.DataFrame(sens), hide_index=True, width="stretch")
st.caption("Reduzir a margem de erro pela metade exige aproximadamente quadruplicar o tamanho da amostra.")

st.header("3. Seleção da amostra aleatória simples (AAS)")
st.markdown(f"""
1. Cada registro da população recebeu um identificador `id_registro` (1 a {fmt(meta['N'], 0)}).
2. Foram sorteados **{fmt(meta['n'], 0)} registros sem reposição**, de modo que cada nascido vivo teve a mesma
   probabilidade de seleção, n/N = {pct(meta['fracao_amostral'], 3)}, e todo subconjunto de tamanho n era igualmente provável.
3. O sorteio usa um gerador pseudoaleatório com **semente fixa ({meta['semente']})**, o que torna a amostra
   reproduzível. O sorteio foi feito uma única vez, **antes** de qualquer análise inferencial.
""")
st.code(f"amostra = populacao.sample(n={meta['n']}, replace=False, random_state={meta['semente']})",
        language="python")
st.markdown("Os registros sorteados estão no arquivo `data/amostra.parquet` do repositório, e o script "
            "`preparar_dados.py` reproduz todo o processo a partir do arquivo original do DATASUS.")
