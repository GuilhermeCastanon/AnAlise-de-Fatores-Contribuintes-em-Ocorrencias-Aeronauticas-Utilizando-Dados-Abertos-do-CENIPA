# =========================
# BIBLIOTECAS
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CONFIG
# =========================
path = "Recursos/"
sns.set_theme(style="whitegrid", context="talk")


# =========================
# LEITURA DOS DADOS
# =========================
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")


# =========================
# LIMPEZA DAS COLUNAS
# =========================
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()


# =========================
# SELEÇÃO DE COLUNAS
# =========================
df_ocorrencia_sel = df_ocorrencia[
    [
        "codigo_ocorrencia",
        "ocorrencia_dia",
        "ocorrencia_hora"
    ]
].copy()

if "ocorrencia_tipo_categoria" in df_tipo.columns:
    col_tipo = "ocorrencia_tipo_categoria"
else:
    col_tipo = "ocorrencia_tipo"

df_tipo_sel = df_tipo[
    [
        "codigo_ocorrencia1",
        col_tipo,
        "taxonomia_tipo_icao"
    ]
].copy()


# =========================
# PADRONIZAÇÃO DOS NOMES DAS CHAVES
# =========================
df_tipo_sel = df_tipo_sel.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia"
    }
)


# =========================
# LIMPEZA DOS TEXTOS
# =========================
df_tipo_sel[col_tipo] = (
    df_tipo_sel[col_tipo]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

df_tipo_sel["taxonomia_tipo_icao"] = (
    df_tipo_sel["taxonomia_tipo_icao"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# =========================
# REMOÇÃO DE DUPLICATAS
# =========================
df_tipo_sel = df_tipo_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        col_tipo,
        "taxonomia_tipo_icao"
    ]
)


# =========================
# MERGE OCORRÊNCIA + TIPO
# =========================
df = pd.merge(
    df_ocorrencia_sel,
    df_tipo_sel,
    on="codigo_ocorrencia",
    how="inner"
)


# =========================
# FILTRO BIRD STRIKE
# =========================
df_bird = df[
    (df[col_tipo] == "COLISÃO COM AVE") |
    (df["taxonomia_tipo_icao"] == "BIRD")
].copy()

# evita contar a mesma ocorrência mais de uma vez
df_bird = df_bird.drop_duplicates(
    subset=[
        "codigo_ocorrencia"
    ]
)

print("\n==============================")
print("BIRD STRIKE")
print("==============================")
print("Total de ocorrências únicas de bird strike:", df_bird["codigo_ocorrencia"].nunique())


# =========================
# TRATAMENTO DE DATA
# =========================
df_bird["ocorrencia_dia_original"] = df_bird["ocorrencia_dia"].copy()

# primeira tentativa: formato brasileiro
df_bird["ocorrencia_dia"] = pd.to_datetime(
    df_bird["ocorrencia_dia_original"],
    format="%d/%m/%Y",
    errors="coerce"
)

# segunda tentativa: caso alguma data venha em outro formato
mask_data_invalida = df_bird["ocorrencia_dia"].isna()

df_bird.loc[mask_data_invalida, "ocorrencia_dia"] = pd.to_datetime(
    df_bird.loc[mask_data_invalida, "ocorrencia_dia_original"],
    errors="coerce",
    dayfirst=True
)

df_bird["ano"] = df_bird["ocorrencia_dia"].dt.year
df_bird["mes"] = df_bird["ocorrencia_dia"].dt.month


# =========================
# TRATAMENTO DE HORA
# =========================
df_bird["hora"] = pd.to_datetime(
    df_bird["ocorrencia_hora"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour


# =========================
# CONFIGURAÇÃO DE MESES
# =========================
nomes_meses = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Abr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Out",
    11: "Nov",
    12: "Dez"
}

ordem_meses_num = list(range(1, 13))

ordem_meses_nome = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]

df_bird["mes_nome"] = df_bird["mes"].map(nomes_meses)


# =========================
# ANÁLISE POR MÊS - BASE INTEIRA
# =========================
bird_por_mes = (
    df_bird
    .dropna(subset=["mes"])
    .groupby("mes")["codigo_ocorrencia"]
    .nunique()
    .reindex(ordem_meses_num, fill_value=0)
)

print("\nOcorrências de bird strike por mês:")
for mes_num, qtd in bird_por_mes.items():
    print(f"{nomes_meses[mes_num]}: {qtd}")

print("\nConferência: soma por mês na base inteira")
print("Soma mensal:", bird_por_mes.sum())
print(
    "Ocorrências únicas com data válida:",
    df_bird.dropna(subset=["mes"])["codigo_ocorrencia"].nunique()
)


plt.figure(figsize=(12, 6))

ax = sns.lineplot(
    x=ordem_meses_nome,
    y=bird_por_mes.values,
    marker="o",
    linewidth=2.5,
    markersize=8
)

ax.set_title("Ocorrências de colisão com ave por mês", fontsize=16, pad=15)
ax.set_xlabel("Mês", fontsize=13)
ax.set_ylabel("Quantidade de ocorrências", fontsize=13)

ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("ocorrencias_bird_por_mes.png", dpi=300)
plt.show()


# =========================
# ANÁLISE POR MÊS - ANOS 2023, 2024 E 2025
# =========================
anos_analisados = [
    2023,
    2024,
    2025
]

df_bird_anos = df_bird[
    df_bird["ano"].isin(anos_analisados)
].copy()

bird_por_mes_ano = (
    df_bird_anos
    .dropna(subset=["mes"])
    .groupby(["ano", "mes"])["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias")
)

bird_por_mes_ano = (
    bird_por_mes_ano
    .pivot(
        index="ano",
        columns="mes",
        values="qtd_ocorrencias"
    )
    .reindex(
        index=anos_analisados,
        columns=ordem_meses_num
    )
    .fillna(0)
    .astype(int)
)

print("\nOcorrências únicas de bird strike por mês - 2023 a 2025:")
print(bird_por_mes_ano)

print("\nConferência: total por ano")
print(bird_por_mes_ano.sum(axis=1))

print("\nConferência: total 2023 a 2025")
print("Soma da tabela:", bird_por_mes_ano.values.sum())
print(
    "Ocorrências únicas filtradas:",
    df_bird_anos.dropna(subset=["mes"])["codigo_ocorrencia"].nunique()
)


plt.figure(figsize=(12, 6))

for ano in anos_analisados:
    plt.plot(
        ordem_meses_nome,
        bird_por_mes_ano.loc[ano].values,
        marker="o",
        linewidth=2.5,
        label=str(ano)
    )

plt.title("Ocorrências de colisão com ave por mês - 2023 a 2025", fontsize=16, pad=15)
plt.xlabel("Mês", fontsize=13)
plt.ylabel("Quantidade de ocorrências", fontsize=13)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Ano")

plt.tight_layout()
plt.savefig("ocorrencias_bird_por_mes_2023_2025.png", dpi=300)
plt.show()


# =========================
# ANÁLISE POR HORA
# =========================
horas = list(range(24))

bird_por_hora = (
    df_bird
    .dropna(subset=["hora"])
    .groupby("hora")["codigo_ocorrencia"]
    .nunique()
    .reindex(horas, fill_value=0)
)

print("\nOcorrências de bird strike por hora:")
for hora, qtd in bird_por_hora.items():
    print(f"{int(hora):02d}h: {qtd}")

print("\nConferência: soma por hora")
print("Soma por hora:", bird_por_hora.sum())
print(
    "Ocorrências únicas com hora válida:",
    df_bird.dropna(subset=["hora"])["codigo_ocorrencia"].nunique()
)


plt.figure(figsize=(12, 6))

ax = sns.lineplot(
    x=bird_por_hora.index,
    y=bird_por_hora.values,
    marker="o",
    linewidth=2.5,
    markersize=7
)

ax.set_title("Ocorrências de colisão com ave por hora do dia", fontsize=16, pad=15)
ax.set_xlabel("Hora do dia", fontsize=13)
ax.set_ylabel("Quantidade de ocorrências", fontsize=13)

ax.set_xticks(horas)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("ocorrencias_bird_por_hora.png", dpi=300)
plt.show()


# =========================
# HEATMAP MÊS x HORA
# =========================
df_heat = df_bird.dropna(
    subset=[
        "mes",
        "hora"
    ]
).copy()

heatmap_data = (
    df_heat
    .groupby(["mes", "hora"])["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias")
    .pivot(
        index="mes",
        columns="hora",
        values="qtd_ocorrencias"
    )
    .reindex(
        index=range(1, 13),
        columns=range(24)
    )
    .fillna(0)
    .astype(int)
)

heatmap_data.index = ordem_meses_nome


# =========================
# CONFERÊNCIA DO HEATMAP
# =========================
print("\nConferência: soma do heatmap por mês")
print(heatmap_data.sum(axis=1))

print("\nConferência: totais do gráfico mensal")
for mes_nome, qtd in zip(ordem_meses_nome, bird_por_mes.values):
    print(f"{mes_nome}: {qtd}")

print("\nConferência: soma total do heatmap")
print("Soma heatmap:", heatmap_data.values.sum())
print(
    "Ocorrências únicas com data e hora válidas:",
    df_heat["codigo_ocorrencia"].nunique()
)


plt.figure(figsize=(16, 7))

ax = sns.heatmap(
    heatmap_data,
    cmap="magma_r",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={
        "label": "Quantidade de ocorrências"
    }
)

ax.set_title(
    "Distribuição conjunta de colisões com ave por mês e hora",
    fontsize=16,
    pad=15
)

ax.set_xlabel("Hora do dia", fontsize=13)
ax.set_ylabel("Mês", fontsize=13)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("heatmap_bird_mes_hora.png", dpi=300)
plt.show()

print("\nOcorrências de bird strike por ano:")
bird_por_ano = (
    df_bird
    .dropna(subset=["ano"])
    .groupby("ano")["codigo_ocorrencia"]
    .nunique()
    .sort_index()
)

print(bird_por_ano)