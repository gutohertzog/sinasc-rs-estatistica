import streamlit as st

from comum import COR_POP, carregar, fmt, painel_descritivo

pop, amo, meta = carregar()

st.title("Painel 1 · Análise descritiva da população")
st.markdown(
    f"**População:** todos os **{fmt(meta['N'], 0)}** nascidos vivos de mães residentes no Rio Grande do Sul "
    "em 2024 (SINASC). Como temos a população inteira, os valores abaixo são **parâmetros**, "
    "e não estimativas.")

painel_descritivo(pop, "População", COR_POP)
