"""Prepara a população (SINASC RS) e sorteia a amostra aleatória simples.

Uso:
    python preparar_dados.py DNRS2024.dbc        # arquivo original do DATASUS (.dbc ou .dbf)
    python preparar_dados.py SINASC_2024.csv     # CSV do OpenDataSUS (separador ';')

Gera, na pasta data/:
    populacao.parquet  - todos os nascidos vivos de mães residentes no RS (a população)
    amostra.parquet    - a AAS sorteada da população
    meta.json          - parâmetros do dimensionamento e do sorteio (documentação)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from estat import tamanho_amostra_proporcao

# ------------------------------------------------------------------ parâmetros do sorteio
CONFIANCA = 0.95   # nível de confiança
MARGEM_ERRO = 0.03  # margem de erro absoluta (3 pontos percentuais)
P_PLANEJ = 0.50    # proporção de planejamento (pior caso: maximiza p(1-p))
SEMENTE = 2026     # semente do gerador -> sorteio reproduzível

COLUNAS = ["CODMUNRES", "DTNASC", "LOCNASC", "PESO", "SEMAGESTAC", "IDADEMAE", "CONSPRENAT",
           "CONSULTAS", "APGAR5", "PARTO", "SEXO", "RACACOR", "ESCMAE", "ESTCIVMAE", "GRAVIDEZ"]

ROTULOS = {
    "PARTO": {"1": "Vaginal", "2": "Cesáreo"},
    "SEXO": {"1": "Masculino", "2": "Feminino", "M": "Masculino", "F": "Feminino"},
    "RACACOR": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"},
    "ESCMAE": {"1": "Nenhuma", "2": "1 a 3 anos", "3": "4 a 7 anos", "4": "8 a 11 anos",
               "5": "12 anos ou mais"},
    "ESTCIVMAE": {"1": "Solteira", "2": "Casada", "3": "Viúva", "4": "Separada/divorciada",
                  "5": "União estável"},
    "GRAVIDEZ": {"1": "Única", "2": "Dupla", "3": "Tripla ou mais"},
    "CONSULTAS": {"1": "Nenhuma", "2": "1 a 3", "3": "4 a 6", "4": "7 ou mais"},
    "LOCNASC": {"1": "Hospital", "2": "Outro estab. de saúde", "3": "Domicílio", "4": "Outros",
                "5": "Aldeia indígena"},
}
ORDINAIS = {"ESCMAE", "CONSULTAS"}


def ler_bruto(caminho: Path) -> pd.DataFrame:
    suf = caminho.suffix.lower()
    if suf in (".dbc", ".dbf"):
        from dbfread import DBF
        if suf == ".dbc":
            from datasus_dbc import decompress
            tmp = Path(tempfile.mkdtemp()) / (caminho.stem + ".dbf")
            decompress(str(caminho), str(tmp))
            caminho_dbf = tmp
        else:
            caminho_dbf = caminho
        tabela = DBF(str(caminho_dbf), encoding="latin-1", char_decode_errors="replace",
                     load=False)
        cols = [c for c in COLUNAS if c in tabela.field_names]
        df = pd.DataFrame(({c: r.get(c) for c in cols} for r in tabela), columns=cols)
    elif suf == ".csv":
        df = pd.read_csv(caminho, sep=";", dtype=str, usecols=lambda c: c in COLUNAS,
                         encoding="latin-1")
    else:
        raise SystemExit(f"Formato não suportado: {suf}")
    return df.astype("string").apply(lambda s: s.str.strip())


def numero(s: pd.Series, ignorados=(), minimo=None, maximo=None) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    v = v.mask(v.isin(ignorados))
    if minimo is not None:
        v = v.mask(v < minimo)
    if maximo is not None:
        v = v.mask(v > maximo)
    return v.astype(float)


def categoria(s: pd.Series, var: str) -> pd.Series:
    mapa = ROTULOS[var]
    rotulos = list(dict.fromkeys(mapa.values()))
    return pd.Categorical(s.map(mapa), categories=rotulos, ordered=var in ORDINAIS)


def montar_populacao(bruto: pd.DataFrame) -> pd.DataFrame:
    # população: nascidos vivos de mães residentes no RS (código IBGE do município começa com 43)
    bruto = bruto[bruto["CODMUNRES"].str.startswith("43", na=False)].reset_index(drop=True)

    df = pd.DataFrame(index=bruto.index)
    df["id_registro"] = np.arange(1, len(bruto) + 1)          # unidade amostral = 1 DNV
    df["data_nascimento"] = pd.to_datetime(bruto["DTNASC"], format="%d%m%Y", errors="coerce")
    df["cod_municipio_res"] = bruto["CODMUNRES"]
    # --- quantitativas (códigos de "ignorado" viram ausentes)
    df["peso_g"] = numero(bruto["PESO"], minimo=200, maximo=7000)
    df["semanas_gestacao"] = numero(bruto["SEMAGESTAC"], ignorados=(99,), minimo=18, maximo=45)
    df["idade_mae"] = numero(bruto["IDADEMAE"], ignorados=(99,), minimo=10, maximo=60)
    df["consultas_prenatal"] = numero(bruto["CONSPRENAT"], ignorados=(99,), minimo=0, maximo=60)
    df["apgar5"] = numero(bruto["APGAR5"], ignorados=(99,), minimo=0, maximo=10)
    # --- qualitativas
    df["tipo_parto"] = categoria(bruto["PARTO"], "PARTO")
    df["sexo"] = categoria(bruto["SEXO"], "SEXO")
    df["raca_cor"] = categoria(bruto["RACACOR"], "RACACOR")
    df["escolaridade_mae"] = categoria(bruto["ESCMAE"], "ESCMAE")
    df["estado_civil_mae"] = categoria(bruto["ESTCIVMAE"], "ESTCIVMAE")
    df["tipo_gravidez"] = categoria(bruto["GRAVIDEZ"], "GRAVIDEZ")
    df["faixa_consultas"] = categoria(bruto["CONSULTAS"], "CONSULTAS")
    df["local_nascimento"] = categoria(bruto["LOCNASC"], "LOCNASC")
    # --- derivadas (definições da OMS / Ministério da Saúde)
    def sim_nao(cond, base):
        return pd.Categorical(np.where(base.isna(), None, np.where(cond, "Sim", "Não")),
                              categories=["Sim", "Não"])
    df["baixo_peso"] = sim_nao(df["peso_g"] < 2500, df["peso_g"])
    df["prematuro"] = sim_nao(df["semanas_gestacao"] < 37, df["semanas_gestacao"])
    df["faixa_idade_mae"] = pd.cut(df["idade_mae"], bins=[0, 19, 34, 99],
                                   labels=["Até 19 anos", "20 a 34 anos", "35 anos ou mais"],
                                   ordered=True)
    return df


def md5(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo", type=Path)
    ap.add_argument("--saida", type=Path, default=Path(__file__).parent / "data")
    args = ap.parse_args()
    args.saida.mkdir(exist_ok=True)

    pop = montar_populacao(ler_bruto(args.arquivo))
    N = len(pop)

    dim = tamanho_amostra_proporcao(N, CONFIANCA, MARGEM_ERRO, P_PLANEJ)
    n = dim["n_arred"]
    # AAS sem reposição: cada um dos N registros tem a mesma probabilidade n/N de ser sorteado
    amostra = pop.sample(n=n, replace=False, random_state=SEMENTE).sort_values("id_registro")

    pop.to_parquet(args.saida / "populacao.parquet", index=False)
    amostra.to_parquet(args.saida / "amostra.parquet", index=False)
    meta = {
        "arquivo_origem": args.arquivo.name,
        "md5_origem": md5(args.arquivo),
        "data_preparo": date.today().isoformat(),
        "N": N,
        "confianca": CONFIANCA,
        "margem_erro": MARGEM_ERRO,
        "p_planejamento": P_PLANEJ,
        "z": dim["z"],
        "n0": dim["n0"],
        "n_corrigido": dim["n"],
        "n": n,
        "fracao_amostral": n / N,
        "semente": SEMENTE,
        "metodo": "pandas.DataFrame.sample(n, replace=False, random_state=semente)",
    }
    (args.saida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
