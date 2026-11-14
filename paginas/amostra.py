import streamlit as st

from comum import COR_AMO, carregar, fmt, painel_descritivo, pct

pop, amo, meta = carregar()

st.title("Painel 2 · Análise descritiva da amostra")
st.markdown(
    f"**Amostra aleatória simples** de **n = {fmt(meta['n'], 0)}** registros "
    f"({pct(meta['fracao_amostral'], 2)} da população), sorteada sem reposição com semente {meta['semente']}. "
    "Os valores abaixo são **estatísticas amostrais**. Os indicadores mostram a diferença em relação à "
    "população, e as tabelas trazem as duas lado a lado.")

painel_descritivo(amo, "Amostra", COR_AMO, ref=pop)
