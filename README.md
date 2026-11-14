# Nascidos vivos no RS (2024): população × amostra

Atividade Avaliativa Final de **Probabilidade e Estatística para IA**, da Especialização em IA Avançada (UFRGS).
Autor: Augusto Hertzog.

Painéis interativos em Streamlit que comparam a **população** de nascidos vivos de mães residentes no Rio Grande do
Sul em 2024 (SINASC/DATASUS) com uma **amostra aleatória simples**. Os painéis incluem intervalos de confiança e
testes de hipóteses para duas proporções e duas médias.

| Página | Conteúdo |
|---|---|
| Sobre o trabalho | Base de dados, população, unidade amostral, variáveis, dimensionamento e sorteio da AAS |
| Painel 1 · População | Análise descritiva da população (parâmetros) |
| Painel 2 · Amostra | Análise descritiva da amostra, comparada à população |
| Painel 3 · Inferência | ICs e testes (P1, P2, M1, M2), comparação com a população, simulação da variabilidade amostral e limitações |

## Fonte dos dados

- **SINASC – Sistema de Informações sobre Nascidos Vivos**, Ministério da Saúde / DATASUS.
- Download: <https://datasus.saude.gov.br/transferencia-de-arquivos/> → SINASC → Dados → DN → 2024 → RS
  (arquivo `DNRS2024.dbc`, em `ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/NOV/DNRES/`).
- Documentação: <https://opendatasus.saude.gov.br/dataset/sistema-de-informacao-sobre-nascidos-vivos-sinasc>

## Reproduzir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt datasus-dbc dbfread
python preparar_dados.py DNRS2024.dbc     # gera data/populacao.parquet, data/amostra.parquet e data/meta.json
streamlit run app.py
```

Os parâmetros do dimensionamento (95% de confiança, margem de erro de 3 p.p., p = 0,5) e a semente do sorteio
(2026) ficam no início de `preparar_dados.py`.

## Estrutura

```
app.py               navegação entre as páginas
comum.py             dados, dicionário de variáveis, hipóteses, gráficos e painel descritivo
estat.py             tamanho amostral, ICs, testes e simulação
preparar_dados.py    leitura do arquivo do DATASUS, limpeza e sorteio da AAS
paginas/             as quatro páginas do app
data/                população, amostra e metadados (gerados por preparar_dados.py)
```
