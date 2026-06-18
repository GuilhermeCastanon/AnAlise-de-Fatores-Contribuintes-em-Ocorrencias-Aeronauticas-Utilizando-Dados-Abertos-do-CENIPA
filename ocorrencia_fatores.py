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
df_fator = pd.read_csv(path + "fator_contribuinte.csv", sep=";", encoding="latin1")

# =========================
# limpeza das colunas
# =========================
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()
df_fator.columns = df_fator.columns.str.strip()


# =========================
# função auxiliar
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


# =========================
# preparação da tabela principal de ocorrências
# =========================
ocorrencias = df_ocorrencia[
    [
        "codigo_ocorrencia"
    ]
].copy()

ocorrencias = ocorrencias.drop_duplicates(
    subset=[
        "codigo_ocorrencia"
    ]
)

print("\nTotal de ocorrências na base principal:")
print(ocorrencias["codigo_ocorrencia"].nunique())


# =========================
# preparação dos tipos de ocorrência
# =========================

# caso exista ocorrencia_tipo_categoria em alguma versão da base,
# usa ela; caso contrário, usa ocorrencia_tipo
if "ocorrencia_tipo_categoria" in df_tipo.columns:
    coluna_tipo = "ocorrencia_tipo_categoria"
else:
    coluna_tipo = "ocorrencia_tipo"

tipos = df_tipo[
    [
        "codigo_ocorrencia1",
        coluna_tipo
    ]
].copy()

tipos = tipos.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia",
        coluna_tipo: "ocorrencia_tipo"
    }
)

tipos = tipos[
    tipos["ocorrencia_tipo"].apply(valor_valido)
].copy()

tipos["ocorrencia_tipo"] = tipos["ocorrencia_tipo"].astype(str).str.strip()

# mantém apenas tipos ligados a ocorrências existentes na tabela principal
tipos = pd.merge(
    ocorrencias,
    tipos,
    on="codigo_ocorrencia",
    how="inner"
)

# remove duplicatas para não contar a mesma ocorrência duas vezes no mesmo tipo
tipos = tipos.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "ocorrencia_tipo"
    ]
)


# =========================
# top 5 tipos de ocorrência mais comuns
# =========================
contagem_tipos = (
    tipos
    .groupby("ocorrencia_tipo")["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias")
    .sort_values(by="qtd_ocorrencias", ascending=False)
)

top5_tipos = contagem_tipos.head(5).copy()

print("\nTop 5 tipos de ocorrência mais comuns:")
print(top5_tipos.to_string(index=False))

lista_top5_tipos = top5_tipos["ocorrencia_tipo"].tolist()


# =========================
# preparação dos fatores contribuintes
# =========================
fatores = df_fator[
    [
        "codigo_ocorrencia3",
        "fator_nome"
    ]
].copy()

fatores = fatores.rename(
    columns={
        "codigo_ocorrencia3": "codigo_ocorrencia"
    }
)

fatores = fatores[
    fatores["fator_nome"].apply(valor_valido)
].copy()

fatores["fator_nome"] = fatores["fator_nome"].astype(str).str.strip()

# mantém apenas fatores ligados a ocorrências existentes na tabela principal
fatores = pd.merge(
    ocorrencias,
    fatores,
    on="codigo_ocorrencia",
    how="inner"
)

# remove duplicatas para contar cada fator apenas uma vez por ocorrência
fatores = fatores.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "fator_nome"
    ]
)

print("\nOcorrências com pelo menos um fator contribuinte registrado:")
print(fatores["codigo_ocorrencia"].nunique())


# =========================
# relação entre top 5 tipos e fatores
# =========================
tipos_top5 = tipos[
    tipos["ocorrencia_tipo"].isin(lista_top5_tipos)
].copy()

relacao_tipo_fator = pd.merge(
    tipos_top5,
    fatores,
    on="codigo_ocorrencia",
    how="inner"
)

print("\nOcorrências dos top 5 tipos com fatores associados:")
print(relacao_tipo_fator["codigo_ocorrencia"].nunique())


# =========================
# contagem de fatores por tipo de ocorrência
# =========================
contagem_fatores = (
    relacao_tipo_fator
    .groupby(["ocorrencia_tipo", "fator_nome"])["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias_com_fator")
)

# total de ocorrências por tipo, considerando todas as ocorrências daquele tipo
total_por_tipo = top5_tipos.rename(
    columns={
        "qtd_ocorrencias": "total_ocorrencias_do_tipo"
    }
)

# total de ocorrências daquele tipo que possuem pelo menos um fator contribuinte registrado
total_com_fator_por_tipo = (
    relacao_tipo_fator
    .groupby("ocorrencia_tipo")["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="ocorrencias_do_tipo_com_fator_registrado")
)

contagem_fatores = pd.merge(
    contagem_fatores,
    total_por_tipo,
    on="ocorrencia_tipo",
    how="left"
)

contagem_fatores = pd.merge(
    contagem_fatores,
    total_com_fator_por_tipo,
    on="ocorrencia_tipo",
    how="left"
)

# percentual em relação ao total de ocorrências daquele tipo
contagem_fatores["perc_sobre_total_do_tipo"] = (
    contagem_fatores["qtd_ocorrencias_com_fator"]
    / contagem_fatores["total_ocorrencias_do_tipo"]
    * 100
)

# percentual considerando apenas as ocorrências daquele tipo que possuem fator registrado
contagem_fatores["perc_sobre_ocorrencias_com_fator"] = (
    contagem_fatores["qtd_ocorrencias_com_fator"]
    / contagem_fatores["ocorrencias_do_tipo_com_fator_registrado"]
    * 100
)

# arredonda as porcentagens para no máximo 3 casas decimais
contagem_fatores["perc_sobre_total_do_tipo"] = (
    contagem_fatores["perc_sobre_total_do_tipo"].round(3)
)

contagem_fatores["perc_sobre_ocorrencias_com_fator"] = (
    contagem_fatores["perc_sobre_ocorrencias_com_fator"].round(3)
)

contagem_fatores = contagem_fatores.sort_values(
    by=[
        "ocorrencia_tipo",
        "qtd_ocorrencias_com_fator"
    ],
    ascending=[
        True,
        False
    ]
)


# =========================
# top 10 fatores por tipo de ocorrência
# =========================
top_fatores_por_tipo = (
    contagem_fatores
    .groupby("ocorrencia_tipo")
    .head(10)
    .copy()
)

print("\nTop fatores contribuintes por tipo de ocorrência:")
print(top_fatores_por_tipo.to_string(index=False))


# =========================
# resumo de cobertura dos fatores
# =========================
cobertura_fatores = top5_tipos.copy()

cobertura_fatores = pd.merge(
    cobertura_fatores,
    total_com_fator_por_tipo,
    on="ocorrencia_tipo",
    how="left"
)

cobertura_fatores["ocorrencias_do_tipo_com_fator_registrado"] = (
    cobertura_fatores["ocorrencias_do_tipo_com_fator_registrado"]
    .fillna(0)
    .astype(int)
)

cobertura_fatores["perc_com_fator_registrado"] = (
    cobertura_fatores["ocorrencias_do_tipo_com_fator_registrado"]
    / cobertura_fatores["qtd_ocorrencias"]
    * 100
).round(3)

print("\nCobertura de fatores contribuintes nos top 5 tipos:")
print(cobertura_fatores.to_string(index=False))


# =========================
# exportação
# =========================
top5_tipos.to_csv(
    "top5_tipos_ocorrencia.csv",
    index=False,
    encoding="utf-8-sig"
)

top_fatores_por_tipo.to_csv(
    "top_fatores_por_tipo_ocorrencia.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)

contagem_fatores.to_csv(
    "fatores_por_tipo_ocorrencia_completo.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)

cobertura_fatores.to_csv(
    "cobertura_fatores_top5_tipos.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)

print("\nArquivos gerados:")
print("- top5_tipos_ocorrencia.csv")
print("- top_fatores_por_tipo_ocorrencia.csv")
print("- fatores_por_tipo_ocorrencia_completo.csv")
print("- cobertura_fatores_top5_tipos.csv")


# =========================
# gráfico: top 5 tipos de ocorrência
# =========================
plt.figure(figsize=(10, 6))

sns.barplot(
    data=top5_tipos,
    x="qtd_ocorrencias",
    y="ocorrencia_tipo"
)

plt.title("Top 5 tipos de ocorrência mais comuns")
plt.xlabel("Quantidade de ocorrências")
plt.ylabel("Tipo de ocorrência")

plt.tight_layout()
plt.savefig("Imagens/grafico_top5_tipos_ocorrencia.png", dpi=300)
plt.show()


# =========================
# gráfico: cobertura de fatores nos top 5 tipos
# =========================
plt.figure(figsize=(10, 6))

sns.barplot(
    data=cobertura_fatores,
    x="perc_com_fator_registrado",
    y="ocorrencia_tipo"
)

plt.title("Percentual de ocorrências com fator registrado nos top 5 tipos")
plt.xlabel("% com fator contribuinte registrado")
plt.ylabel("Tipo de ocorrência")

plt.tight_layout()
plt.savefig("Imagens/grafico_cobertura_fatores_top5_tipos.png", dpi=300)
plt.show()