import streamlit as st

st.set_page_config(page_title="Nascidos vivos no RS – Estatística", page_icon=":material/child_care:",
                   layout="wide")

paginas = [
    st.Page("paginas/inicio.py", title="Sobre o trabalho", icon=":material/info:", default=True),
    st.Page("paginas/populacao.py", title="Painel 1 · População", icon=":material/groups:",
            url_path="populacao"),
    st.Page("paginas/amostra.py", title="Painel 2 · Amostra", icon=":material/scatter_plot:",
            url_path="amostra"),
    st.Page("paginas/inferencia.py", title="Painel 3 · Inferência", icon=":material/functions:",
            url_path="inferencia"),
]
st.navigation(paginas).run()

st.sidebar.caption("Atividade Avaliativa Final · Probabilidade e Estatística para IA · "
                   "Especialização em IA Avançada – UFRGS · Augusto Hertzog")
