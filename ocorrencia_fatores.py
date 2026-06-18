import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CAMINHOS DAS PASTAS
# =========================
# aqui eu deixo os caminhos separados para não precisar escrever o nome das pastas
# várias vezes ao longo do código. a pasta Recursos contém os arquivos originais
# da base, enquanto as pastas CSVs e Imagens recebem os resultados gerados.
path = "Recursos/"
pasta_csvs = "CSVs/"
pasta_imagens = "Imagens/"

# criei essas pastas automaticamente caso elas ainda não existam.
# isso evita erro na hora de salvar os arquivos de saída.
os.makedirs(pasta_csvs, exist_ok=True)
os.makedirs(pasta_imagens, exist_ok=True)


# =========================
# LEITURA DOS DADOS
# =========================
# nesta seção eu carrego apenas as tabelas necessárias para esta análise.
# a tabela ocorrencia.csv é usada como base principal, a ocorrencia_tipo.csv
# informa os tipos associados a cada ocorrência, e a fator_contribuinte.csv
# traz os fatores registrados nas ocorrências que possuem esse tipo de informação.
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")
df_fator = pd.read_csv(path + "fator_contribuinte.csv", sep=";", encoding="latin1")


# =========================
# PADRONIZAÇÃO DOS NOMES DAS COLUNAS
# =========================
# algumas bases podem vir com espaços extras nos nomes das colunas.
# para evitar erro ao selecionar uma coluna pelo nome, removo esses espaços
# logo depois da leitura dos arquivos.
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()
df_fator.columns = df_fator.columns.str.strip()


# =========================
# FUNÇÃO PARA FILTRAR VALORES INVÁLIDOS
# =========================
# pensei em criar uma função para concentrar a regra de limpeza em um só lugar.
# assim, sempre que eu precisar verificar se um valor é válido, uso a mesma função.
# isso evita repetir várias condições ao longo do código.
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
# PREPARAÇÃO DA TABELA PRINCIPAL DE OCORRÊNCIAS
# =========================
# aqui eu mantenho apenas o código da ocorrência, pois ele será usado como chave
# para integrar as demais tabelas. uso essa tabela como referência principal para
# garantir que só sejam analisados registros que realmente existem na base de ocorrências.
ocorrencias = df_ocorrencia[
    [
        "codigo_ocorrencia"
    ]
].copy()

# removo duplicatas para garantir que cada ocorrência exista apenas uma vez
# nesta tabela de referência.
ocorrencias = ocorrencias.drop_duplicates(
    subset=[
        "codigo_ocorrencia"
    ]
)


# =========================
# PREPARAÇÃO DOS TIPOS DE OCORRÊNCIA
# =========================
# nesta parte eu preparo a tabela que liga cada ocorrência ao seu tipo.
# deixei esse teste porque algumas versões da base podem ter a coluna
# ocorrencia_tipo_categoria, enquanto outras usam apenas ocorrencia_tipo.
# a ideia é deixar o código mais flexível para versões diferentes da base.
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

# renomeio as colunas para facilitar o merge com a tabela principal.
# assim todas as tabelas passam a usar o mesmo nome de chave: codigo_ocorrencia.
tipos = tipos.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia",
        coluna_tipo: "ocorrencia_tipo"
    }
)

# removo tipos vazios, desconhecidos ou não informados.
# fiz isso porque esses valores não ajudam na análise e poderiam poluir as contagens.
tipos = tipos[
    tipos["ocorrencia_tipo"].apply(valor_valido)
].copy()

tipos["ocorrencia_tipo"] = tipos["ocorrencia_tipo"].astype(str).str.strip()

# faço o merge com a tabela principal para manter apenas tipos ligados
# a ocorrências existentes em ocorrencia.csv.
tipos = pd.merge(
    ocorrencias,
    tipos,
    on="codigo_ocorrencia",
    how="inner"
)

# uma mesma ocorrência pode ter mais de um tipo, o que é esperado.
# por isso, eu não removo todas as duplicatas por codigo_ocorrencia.
# removo apenas a repetição exata da mesma combinação ocorrência + tipo,
# evitando contar duas vezes o mesmo tipo para a mesma ocorrência.
tipos = tipos.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "ocorrencia_tipo"
    ]
)


# =========================
# IDENTIFICAÇÃO DOS 5 TIPOS DE OCORRÊNCIA MAIS FREQUENTES
# =========================
# aqui eu conto quantas ocorrências únicas existem para cada tipo.
# usei nunique porque a unidade de análise é a ocorrência, e não a linha da tabela.
# isso é importante porque uma ocorrência pode aparecer em mais de uma linha.
contagem_tipos = (
    tipos
    .groupby("ocorrencia_tipo")["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias")
    .sort_values(by="qtd_ocorrencias", ascending=False)
)

top5_tipos = contagem_tipos.head(5).copy()

# guardo a lista dos cinco tipos mais comuns para usar depois na análise dos fatores.
# pensei em fazer essa seleção porque analisar todos os tipos deixaria a saída muito grande
# e com muitos casos pouco representativos.
lista_top5_tipos = top5_tipos["ocorrencia_tipo"].tolist()


# =========================
# PREPARAÇÃO DOS FATORES CONTRIBUINTES
# =========================
# nesta seção eu preparo a tabela de fatores contribuintes.
# como o objetivo é relacionar tipos de ocorrência com fatores, mantenho apenas
# o código da ocorrência e o nome do fator.
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

# removo fatores vazios ou não informativos pelo mesmo motivo usado nos tipos:
# eles não contribuem para a interpretação dos resultados.
fatores = fatores[
    fatores["fator_nome"].apply(valor_valido)
].copy()

fatores["fator_nome"] = fatores["fator_nome"].astype(str).str.strip()

# mantenho apenas fatores ligados a ocorrências existentes na tabela principal.
# isso evita analisar registros soltos ou inconsistentes entre os arquivos.
fatores = pd.merge(
    ocorrencias,
    fatores,
    on="codigo_ocorrencia",
    how="inner"
)

# uma ocorrência pode ter vários fatores, então não posso remover duplicatas apenas
# pelo código da ocorrência. removo somente a repetição exata da mesma combinação
# ocorrência + fator, para contar cada fator uma única vez por ocorrência.
fatores = fatores.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "fator_nome"
    ]
)


# =========================
# RELAÇÃO ENTRE OS 5 TIPOS MAIS FREQUENTES E OS FATORES
# =========================
# aqui eu separo apenas as ocorrências pertencentes aos cinco tipos mais comuns.
# depois cruzo essas ocorrências com os fatores contribuintes para descobrir
# quais fatores aparecem em cada um desses tipos.
tipos_top5 = tipos[
    tipos["ocorrencia_tipo"].isin(lista_top5_tipos)
].copy()

relacao_tipo_fator = pd.merge(
    tipos_top5,
    fatores,
    on="codigo_ocorrencia",
    how="inner"
)


# =========================
# CONTAGEM DE FATORES POR TIPO DE OCORRÊNCIA
# =========================
# nesta etapa eu conto quantas ocorrências únicas apresentam cada fator dentro de cada tipo.
# novamente uso nunique porque quero contar ocorrências, não linhas da tabela.
contagem_fatores = (
    relacao_tipo_fator
    .groupby(["ocorrencia_tipo", "fator_nome"])["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="qtd_ocorrencias_com_fator")
)

# total de ocorrências por tipo, considerando todas as ocorrências daquele tipo.
# esse total será usado para calcular o percentual sobre o total do tipo.
total_por_tipo = top5_tipos.rename(
    columns={
        "qtd_ocorrencias": "total_ocorrencias_do_tipo"
    }
)

# total de ocorrências daquele tipo que possuem pelo menos um fator contribuinte.
# esse valor é importante porque muitos tipos possuem poucas ocorrências com fator registrado.
total_com_fator_por_tipo = (
    relacao_tipo_fator
    .groupby("ocorrencia_tipo")["codigo_ocorrencia"]
    .nunique()
    .reset_index(name="ocorrencias_do_tipo_com_fator_registrado")
)

# junto as contagens de fatores com os totais necessários para calcular os percentuais.
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


# =========================
# CÁLCULO DOS PERCENTUAIS
# =========================
# calculei dois percentuais porque eles contam histórias diferentes.
# o primeiro mostra o peso do fator em relação a todas as ocorrências daquele tipo.
# o segundo considera apenas o subconjunto que realmente possui fator registrado.
contagem_fatores["perc_sobre_total_do_tipo"] = (
    contagem_fatores["qtd_ocorrencias_com_fator"]
    / contagem_fatores["total_ocorrencias_do_tipo"]
    * 100
)

contagem_fatores["perc_sobre_ocorrencias_com_fator"] = (
    contagem_fatores["qtd_ocorrencias_com_fator"]
    / contagem_fatores["ocorrencias_do_tipo_com_fator_registrado"]
    * 100
)

# arredondo as porcentagens para deixar os arquivos gerados mais legíveis.
contagem_fatores["perc_sobre_total_do_tipo"] = (
    contagem_fatores["perc_sobre_total_do_tipo"].round(3)
)

contagem_fatores["perc_sobre_ocorrencias_com_fator"] = (
    contagem_fatores["perc_sobre_ocorrencias_com_fator"].round(3)
)

# ordeno por tipo e, dentro de cada tipo, pelos fatores mais frequentes.
# isso facilita tanto a leitura do csv quanto a escolha dos resultados para a monografia.
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
# SELEÇÃO DOS 10 PRINCIPAIS FATORES POR TIPO
# =========================
# depois de calcular todos os fatores, separo apenas os dez primeiros de cada tipo.
# pensei em fazer isso porque a tabela completa é útil como saída, mas muito grande
# para ser interpretada diretamente no texto do trabalho.
top_fatores_por_tipo = (
    contagem_fatores
    .groupby("ocorrencia_tipo")
    .head(10)
    .copy()
)


# =========================
# RESUMO DA COBERTURA DE FATORES
# =========================
# esta parte calcula quantas ocorrências dos top 5 tipos possuem fator registrado.
# isso foi importante no trabalho porque alguns tipos aparecem muito na base,
# mas quase não possuem fatores contribuintes associados.
cobertura_fatores = top5_tipos.copy()

cobertura_fatores = pd.merge(
    cobertura_fatores,
    total_com_fator_por_tipo,
    on="ocorrencia_tipo",
    how="left"
)

# quando um tipo não possui fator registrado, o merge gera valor vazio.
# nesse caso, substituo por zero para manter a interpretação correta.
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


# =========================
# EXPORTAÇÃO DOS RESULTADOS EM CSV
# =========================
# salvo os resultados principais em arquivos csv separados.
# isso deixa o projeto mais organizado e facilita consultar os dados depois,
# sem precisar rodar o script novamente toda vez.
top5_tipos.to_csv(
    pasta_csvs + "top5_tipos_ocorrencia.csv",
    index=False,
    encoding="utf-8-sig"
)

top_fatores_por_tipo.to_csv(
    pasta_csvs + "top_fatores_por_tipo_ocorrencia.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)

contagem_fatores.to_csv(
    pasta_csvs + "fatores_por_tipo_ocorrencia_completo.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)

cobertura_fatores.to_csv(
    pasta_csvs + "cobertura_fatores_top5_tipos.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.3f"
)


# =========================
# GRÁFICO DOS 5 TIPOS DE OCORRÊNCIA MAIS FREQUENTES
# =========================
# este gráfico ajuda a visualizar rapidamente quais tipos dominam a base.
# preferi um gráfico de barras horizontal porque os nomes dos tipos são longos.
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
plt.savefig(pasta_imagens + "grafico_top5_tipos_ocorrencia.png", dpi=300)
plt.close()


# =========================
# GRÁFICO DE COBERTURA DE FATORES NOS 5 TIPOS MAIS FREQUENTES
# =========================
# aqui eu mostro o percentual de ocorrências com fator registrado em cada tipo.
# pensei nesse gráfico porque ele deixa clara uma limitação importante da base:
# nem todo tipo frequente possui muitos fatores contribuintes disponíveis.
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
plt.savefig(pasta_imagens + "grafico_cobertura_fatores_top5_tipos.png", dpi=300)
plt.close()