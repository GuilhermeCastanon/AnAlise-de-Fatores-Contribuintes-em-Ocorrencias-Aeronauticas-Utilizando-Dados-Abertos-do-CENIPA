# bibliotecas para exploração de dados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# caminho dos arquivos
# =========================
path = "Recursos/"


# =========================
# leitura dos dados
# =========================
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")
df_aeronave = pd.read_csv(path + "aeronave.csv", sep=";", encoding="latin1")
df_fator = pd.read_csv(path + "fator_contribuinte.csv", sep=";", encoding="latin1")

# recomendação não é usada neste código, mas deixei a leitura opcional
try:
    df_recomendacao = pd.read_csv(path + "recomendacao.csv", sep=";", encoding="latin1")
except FileNotFoundError:
    df_recomendacao = None


# =========================
# limpeza dos nomes das colunas
# =========================
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()
df_aeronave.columns = df_aeronave.columns.str.strip()
df_fator.columns = df_fator.columns.str.strip()

if df_recomendacao is not None:
    df_recomendacao.columns = df_recomendacao.columns.str.strip()


# =========================
# funções auxiliares
# =========================
def valor_valido(valor):
    if pd.isna(valor):
        return False

    valor = str(valor).strip()

    valores_ignorar = [
        "",
        "NAN",
        "***",
        "DESCONHECIDO",
        "NÃO INFORMADO",
        "NAO INFORMADO",
        "NI",
        "NULL",
        "NONE"
    ]

    return valor.upper() not in valores_ignorar


def limpar_texto_coluna(df, coluna):
    df[coluna] = df[coluna].astype(str).str.strip()
    return df


# =========================
# seleção de colunas principais
# =========================
df_ocorrencia_sel = df_ocorrencia[
    [
        "codigo_ocorrencia",
        "ocorrencia_classificacao",
        "ocorrencia_uf",
        "ocorrencia_dia",
        "ocorrencia_hora"
    ]
].copy()

df_aeronave_sel = df_aeronave[
    [
        "codigo_ocorrencia2",
        "aeronave_modelo",
        "aeronave_fabricante"
    ]
].copy()

df_fator_sel = df_fator[
    [
        "codigo_ocorrencia3",
        "fator_nome",
        "fator_area"
    ]
].copy()

df_tipo_sel = df_tipo[
    [
        "codigo_ocorrencia1",
        "ocorrencia_tipo",
        "taxonomia_tipo_icao"
    ]
].copy()


# =========================
# padronização dos nomes das chaves
# =========================
df_aeronave_sel = df_aeronave_sel.rename(
    columns={
        "codigo_ocorrencia2": "codigo_ocorrencia"
    }
)

df_fator_sel = df_fator_sel.rename(
    columns={
        "codigo_ocorrencia3": "codigo_ocorrencia"
    }
)

df_tipo_sel = df_tipo_sel.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia"
    }
)


# =========================
# limpeza de valores inválidos
# =========================
df_aeronave_sel = df_aeronave_sel[
    df_aeronave_sel["aeronave_modelo"].apply(valor_valido)
].copy()

df_fator_sel = df_fator_sel[
    df_fator_sel["fator_nome"].apply(valor_valido)
].copy()

df_tipo_sel = df_tipo_sel[
    df_tipo_sel["ocorrencia_tipo"].apply(valor_valido)
].copy()

df_aeronave_sel = limpar_texto_coluna(df_aeronave_sel, "aeronave_modelo")
df_aeronave_sel = limpar_texto_coluna(df_aeronave_sel, "aeronave_fabricante")
df_fator_sel = limpar_texto_coluna(df_fator_sel, "fator_nome")
df_fator_sel = limpar_texto_coluna(df_fator_sel, "fator_area")
df_tipo_sel = limpar_texto_coluna(df_tipo_sel, "ocorrencia_tipo")
df_tipo_sel = limpar_texto_coluna(df_tipo_sel, "taxonomia_tipo_icao")


# =========================
# remoção de duplicatas
# =========================
# uma ocorrência pode ter mais de uma aeronave e mais de um fator.
# por isso, removemos duplicatas para contar cada associação apenas uma vez.

df_aeronave_sel = df_aeronave_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "aeronave_modelo",
        "aeronave_fabricante"
    ]
)

df_fator_sel = df_fator_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "fator_nome"
    ]
)

df_tipo_sel = df_tipo_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "ocorrencia_tipo",
        "taxonomia_tipo_icao"
    ]
)


# =========================
# inspeção inicial da base
# =========================
print("\n==============================")
print("RESUMO DA BASE")
print("==============================")

print("Total de ocorrências:", df_ocorrencia["codigo_ocorrencia"].nunique())
print("Total de registros em aeronave:", len(df_aeronave))
print("Total de registros em ocorrência_tipo:", len(df_tipo))
print("Total de registros em fator_contribuinte:", len(df_fator))
print("Ocorrências com fator contribuinte:", df_fator_sel["codigo_ocorrencia"].nunique())


# ======================================================
# ANÁLISE 1: MODELOS DE AERONAVE MAIS COMUNS
# ======================================================

# conta ocorrências únicas por modelo de aeronave
top_modelos = (
    df_aeronave_sel
    .groupby("aeronave_modelo")["codigo_ocorrencia"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)

print("\n==============================")
print("10 MODELOS DE AERONAVE MAIS COMUNS")
print("==============================")
print(top_modelos.to_string())


# gráfico - modelos de aeronave
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_modelos.values,
    y=top_modelos.index
)

plt.title("10 modelos de aeronave mais comuns", fontsize=14)
plt.xlabel("Quantidade de ocorrências", fontsize=12)
plt.ylabel("Modelo", fontsize=12)

plt.tight_layout()
plt.show()


# ======================================================
# ANÁLISE 2: FATORES CONTRIBUINTES MAIS COMUNS
# ======================================================

# conta ocorrências únicas por fator contribuinte
top_fatores = (
    df_fator_sel
    .groupby("fator_nome")["codigo_ocorrencia"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)

print("\n==============================")
print("10 FATORES CONTRIBUINTES MAIS COMUNS")
print("==============================")
print(top_fatores.to_string())


# gráfico - fatores contribuintes
plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_fatores.values,
    y=top_fatores.index
)

plt.title("10 fatores contribuintes mais comuns", fontsize=14)
plt.xlabel("Quantidade de ocorrências", fontsize=12)
plt.ylabel("Fator contribuinte", fontsize=12)

plt.tight_layout()
plt.show()


# ======================================================
# ANÁLISE 3: RELAÇÃO ENTRE MODELOS E FATORES
# ======================================================

# merge entre aeronaves e fatores
# cada linha representa uma associação:
# ocorrência + modelo de aeronave + fator contribuinte

df_modelo_fator = pd.merge(
    df_aeronave_sel,
    df_fator_sel,
    on="codigo_ocorrencia",
    how="inner"
)

print("\n==============================")
print("MERGE MODELO + FATOR")
print("==============================")
print(df_modelo_fator.head())
print("Linhas:", len(df_modelo_fator))
print("Ocorrências únicas:", df_modelo_fator["codigo_ocorrencia"].nunique())


# contagem de ocorrências por modelo, usando a tabela de aeronaves
contagem_modelos = (
    df_aeronave_sel
    .groupby("aeronave_modelo")["codigo_ocorrencia"]
    .nunique()
    .sort_values(ascending=False)
)

# mantém apenas modelos com pelo menos 20 ocorrências
modelos_validos = contagem_modelos[contagem_modelos >= 20].index

df_filtrado = df_modelo_fator[
    df_modelo_fator["aeronave_modelo"].isin(modelos_validos)
].copy()

print("\nModelos mantidos:", len(modelos_validos))
print("Linhas após filtro:", len(df_filtrado))


# tabela cruzada modelo x fator
tabela = pd.crosstab(
    df_filtrado["aeronave_modelo"],
    df_filtrado["fator_nome"]
)

# normalização por modelo
tabela_normalizada = tabela.div(tabela.sum(axis=1), axis=0)

# seleciona os 10 modelos e 10 fatores mais comuns dentro do conjunto filtrado
top_modelos_heatmap = (
    df_filtrado
    .groupby("aeronave_modelo")["codigo_ocorrencia"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .index
)

top_fatores_heatmap = (
    df_filtrado
    .groupby("fator_nome")["codigo_ocorrencia"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
    .index
)

tabela_reduzida = tabela_normalizada.loc[
    top_modelos_heatmap,
    top_fatores_heatmap
]


# gráfico - heatmap modelo x fator
plt.figure(figsize=(14, 8))

sns.heatmap(
    tabela_reduzida,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={
        "label": "Proporção"
    }
)

plt.title(
    "Proporção de fatores contribuintes por modelo de aeronave",
    fontsize=14,
    pad=20
)

plt.xlabel("Fator contribuinte", fontsize=12)
plt.ylabel("Modelo de aeronave", fontsize=12)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()
plt.show()

