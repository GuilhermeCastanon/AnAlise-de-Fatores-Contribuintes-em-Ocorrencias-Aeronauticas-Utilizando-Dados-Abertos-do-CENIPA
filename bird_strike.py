# =========================
# BIBLIOTECAS
# =========================
# uso o pandas para manipular os dados em tabelas, o matplotlib para salvar os gráficos
# e o seaborn para deixar as visualizações com uma aparência mais limpa.
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CONFIGURAÇÕES GERAIS
# =========================
# aqui eu concentro os caminhos das pastas usadas no script.
# pensei em fazer assim porque facilita mudar a organização do projeto depois,
# sem precisar procurar caminho de arquivo espalhado pelo código inteiro.
path = "Recursos/"
pasta_imagens = "Imagens/"
pasta_csvs = "CSVs/"

# essas linhas garantem que as pastas de saída existam antes de salvar arquivos.
# assim o código não quebra caso eu rode o projeto em outro computador.
os.makedirs(pasta_imagens, exist_ok=True)
os.makedirs(pasta_csvs, exist_ok=True)

# configurei o tema do seaborn para deixar os gráficos mais adequados para relatório.
# o whitegrid ajuda na leitura dos valores sem deixar o gráfico visualmente pesado.
sns.set_theme(style="whitegrid", context="talk")


# =========================
# LEITURA DOS DADOS
# =========================
# nesta análise eu uso duas tabelas: a tabela de ocorrências, que possui data e hora,
# e a tabela de tipos de ocorrência, que permite identificar quais registros são
# colisões com ave.
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")


# =========================
# PADRONIZAÇÃO DOS NOMES DAS COLUNAS
# =========================
# removo espaços extras nos nomes das colunas logo no começo.
# isso evita erro quando eu for acessar uma coluna pelo nome.
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()


# =========================
# SELEÇÃO DAS COLUNAS NECESSÁRIAS
# =========================
# aqui eu reduzo a tabela de ocorrências para manter apenas o que realmente será usado:
# o código da ocorrência, a data e o horário. fiz isso para deixar as próximas etapas
# mais simples e evitar carregar informação que não entra nessa análise específica.
df_ocorrencia_sel = df_ocorrencia[
    [
        "codigo_ocorrencia",
        "ocorrencia_dia",
        "ocorrencia_hora"
    ]
].copy()

# algumas versões da base podem trazer a coluna ocorrencia_tipo_categoria,
# enquanto outras usam ocorrencia_tipo. deixei essa verificação para o código
# funcionar nas duas situações.
if "ocorrencia_tipo_categoria" in df_tipo.columns:
    col_tipo = "ocorrencia_tipo_categoria"
else:
    col_tipo = "ocorrencia_tipo"

# na tabela de tipos, mantenho o código da ocorrência, o nome do tipo e a taxonomia ICAO.
# usei também a taxonomia porque a colisão com ave pode ser identificada pelo código BIRD.
df_tipo_sel = df_tipo[
    [
        "codigo_ocorrencia1",
        col_tipo,
        "taxonomia_tipo_icao"
    ]
].copy()


# =========================
# PADRONIZAÇÃO DA CHAVE DE JUNÇÃO
# =========================
# a tabela de tipos usa o nome codigo_ocorrencia1, enquanto a tabela principal usa
# codigo_ocorrencia. renomeio a coluna para conseguir integrar as tabelas com merge.
df_tipo_sel = df_tipo_sel.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia"
    }
)


# =========================
# LIMPEZA E PADRONIZAÇÃO DOS TEXTOS
# =========================
# nesta parte eu padronizo os textos para letras maiúsculas e removo espaços extras.
# pensei em fazer assim porque o filtro de colisão com ave depende de comparação textual,
# então "COLISÃO COM AVE" e " colisão com ave " precisam ser tratados como a mesma coisa.
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
# REMOÇÃO DE DUPLICATAS NA TABELA DE TIPOS
# =========================
# uma ocorrência pode aparecer mais de uma vez na tabela de tipos, o que pode ser válido.
# por isso, removo apenas repetições exatas da combinação ocorrência + tipo + taxonomia.
# a intenção é evitar contar duplicatas artificiais sem apagar relações reais.
df_tipo_sel = df_tipo_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        col_tipo,
        "taxonomia_tipo_icao"
    ]
)


# =========================
# INTEGRAÇÃO ENTRE OCORRÊNCIAS E TIPOS
# =========================
# aqui eu junto a tabela com data e hora com a tabela que informa o tipo da ocorrência.
# uso inner join porque, nesta análise, só interessam registros que possuem informação
# de tipo associada.
df = pd.merge(
    df_ocorrencia_sel,
    df_tipo_sel,
    on="codigo_ocorrencia",
    how="inner"
)


# =========================
# FILTRO DAS OCORRÊNCIAS DE COLISÃO COM AVE
# =========================
# aqui eu filtro apenas as ocorrências relacionadas a colisão com ave.
# pensei em usar tanto o nome do tipo quanto a taxonomia ICAO porque, dependendo
# da forma como o registro aparece na base, a ocorrência pode ser identificada
# como "COLISÃO COM AVE" ou pela taxonomia "BIRD".
df_bird = df[
    (df[col_tipo] == "COLISÃO COM AVE") |
    (df["taxonomia_tipo_icao"] == "BIRD")
].copy()

# depois do filtro, removo duplicatas por código de ocorrência.
# isso é importante porque uma mesma ocorrência pode ter mais de uma linha na tabela
# de tipos, mas para esta análise cada ocorrência deve ser contada apenas uma vez.
df_bird = df_bird.drop_duplicates(
    subset=[
        "codigo_ocorrencia"
    ]
)


# =========================
# TRATAMENTO DAS DATAS
# =========================
# guardo uma cópia da data original antes da conversão.
# fiz isso porque, se alguma conversão falhar, ainda consigo rastrear o valor original.
df_bird["ocorrencia_dia_original"] = df_bird["ocorrencia_dia"].copy()

# primeiro tento converter usando o formato brasileiro, que é o formato esperado
# na base utilizada no trabalho.
df_bird["ocorrencia_dia"] = pd.to_datetime(
    df_bird["ocorrencia_dia_original"],
    format="%d/%m/%Y",
    errors="coerce"
)

# se alguma data não for convertida na primeira tentativa, faço uma segunda conversão
# mais flexível, mantendo dayfirst=True para respeitar a ordem dia/mês/ano.
mask_data_invalida = df_bird["ocorrencia_dia"].isna()

df_bird.loc[mask_data_invalida, "ocorrencia_dia"] = pd.to_datetime(
    df_bird.loc[mask_data_invalida, "ocorrencia_dia_original"],
    errors="coerce",
    dayfirst=True
)

# extraio ano e mês porque eles serão usados nas análises temporais.
df_bird["ano"] = df_bird["ocorrencia_dia"].dt.year
df_bird["mes"] = df_bird["ocorrencia_dia"].dt.month


# =========================
# TRATAMENTO DOS HORÁRIOS
# =========================
# aqui converto o horário para datetime e depois extraio apenas a hora.
# para esta análise, o minuto e o segundo não são necessários, porque o objetivo
# é observar a distribuição das ocorrências ao longo das 24 horas do dia.
df_bird["hora"] = pd.to_datetime(
    df_bird["ocorrencia_hora"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour


# =========================
# CONFIGURAÇÃO DOS MESES
# =========================
# criei esse dicionário para transformar o número do mês em nome abreviado.
# isso deixa os gráficos mais legíveis do que usar apenas os números de 1 a 12.
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
# ANÁLISE POR MÊS - BASE COMPLETA
# =========================
# nesta etapa conto quantas ocorrências únicas de colisão com ave existem em cada mês.
# uso nunique porque a unidade de análise é a ocorrência, e não a linha do dataframe.
bird_por_mes = (
    df_bird
    .dropna(subset=["mes"])
    .groupby("mes")["codigo_ocorrencia"]
    .nunique()
    .reindex(ordem_meses_num, fill_value=0)
)

# transformo a série em dataframe para salvar o resultado em CSV.
# isso substitui os prints de conferência e permite consultar os valores depois.
bird_por_mes_df = bird_por_mes.reset_index()
bird_por_mes_df.columns = ["mes", "qtd_ocorrencias"]
bird_por_mes_df["mes_nome"] = bird_por_mes_df["mes"].map(nomes_meses)

bird_por_mes_df = bird_por_mes_df[
    [
        "mes",
        "mes_nome",
        "qtd_ocorrencias"
    ]
]

bird_por_mes_df.to_csv(
    pasta_csvs + "bird_strike_por_mes.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# GRÁFICO DE COLISÕES COM AVE POR MÊS
# =========================
# este gráfico mostra a distribuição mensal das colisões com ave considerando
# toda a base filtrada. pensei em usar linha porque os meses possuem uma ordem natural.
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
plt.savefig(pasta_imagens + "ocorrencias_bird_por_mes.png", dpi=300)
plt.close()


# =========================
# ANÁLISE POR MÊS - ANOS 2023, 2024 E 2025
# =========================
# aqui faço um recorte específico dos anos mais recentes usados na comparação.
# pensei em separar esses anos porque eles permitem observar melhor a variação mensal
# recente, sem misturar todo o histórico da base em uma única curva.
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

# reorganizo a tabela para que cada linha seja um ano e cada coluna seja um mês.
# esse formato facilita tanto a geração do gráfico quanto a exportação dos resultados.
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

# salvo a tabela com os nomes dos meses nas colunas para facilitar a leitura.
bird_por_mes_ano_csv = bird_por_mes_ano.copy()
bird_por_mes_ano_csv.columns = ordem_meses_nome

bird_por_mes_ano_csv.to_csv(
    pasta_csvs + "bird_strike_por_mes_2023_2025.csv",
    index=True,
    index_label="ano",
    encoding="utf-8-sig"
)


# =========================
# GRÁFICO DE COLISÕES COM AVE POR MÊS ENTRE 2023 E 2025
# =========================
# neste gráfico, comparo os três anos no mesmo eixo mensal.
# essa visualização ajuda a perceber se algum ano teve comportamento muito diferente
# dos demais em determinados meses.
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
plt.savefig(pasta_imagens + "ocorrencias_bird_por_mes_2023_2025.png", dpi=300)
plt.close()


# =========================
# ANÁLISE POR HORA DO DIA
# =========================
# nesta etapa conto as ocorrências de colisão com ave por hora.
# pensei em fazer essa análise porque o horário do dia pode revelar concentrações
# em períodos específicos, como manhã, tarde ou noite.
horas = list(range(24))

bird_por_hora = (
    df_bird
    .dropna(subset=["hora"])
    .groupby("hora")["codigo_ocorrencia"]
    .nunique()
    .reindex(horas, fill_value=0)
)

bird_por_hora_df = bird_por_hora.reset_index()
bird_por_hora_df.columns = ["hora", "qtd_ocorrencias"]
bird_por_hora_df["hora_formatada"] = bird_por_hora_df["hora"].astype(int).astype(str).str.zfill(2) + "h"

bird_por_hora_df = bird_por_hora_df[
    [
        "hora",
        "hora_formatada",
        "qtd_ocorrencias"
    ]
]

bird_por_hora_df.to_csv(
    pasta_csvs + "bird_strike_por_hora.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# GRÁFICO DE COLISÕES COM AVE POR HORA
# =========================
# uso um gráfico de linha porque as horas também formam uma sequência ordenada.
# esse gráfico ajuda a visualizar em quais períodos do dia os registros se concentram.
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
plt.savefig(pasta_imagens + "ocorrencias_bird_por_hora.png", dpi=300)
plt.close()


# =========================
# PREPARAÇÃO DOS DADOS PARA O HEATMAP MÊS x HORA
# =========================
# para o heatmap, eu preciso manter apenas registros que tenham mês e hora válidos.
# pensei nessa visualização porque ela combina as duas dimensões temporais ao mesmo tempo,
# mostrando em quais meses e horários as ocorrências aparecem com maior frequência.
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

heatmap_data.to_csv(
    pasta_csvs + "bird_strike_heatmap_mes_hora.csv",
    index=True,
    index_label="mes",
    encoding="utf-8-sig"
)


# =========================
# GRÁFICO HEATMAP MÊS x HORA
# =========================
# neste gráfico, cada célula representa a quantidade de ocorrências em uma combinação
# específica de mês e hora. quanto mais intensa a cor, maior a quantidade de registros.
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
plt.savefig(pasta_imagens + "heatmap_bird_mes_hora.png", dpi=300)
plt.close()


# =========================
# ANÁLISE POR ANO
# =========================
# por fim, calculo a quantidade de ocorrências de colisão com ave por ano.
# essa tabela não entra necessariamente como gráfico principal, mas ajuda a entender
# a distribuição temporal geral da base e pode ser usada como apoio na apresentação.
bird_por_ano = (
    df_bird
    .dropna(subset=["ano"])
    .groupby("ano")["codigo_ocorrencia"]
    .nunique()
    .sort_index()
)

bird_por_ano_df = bird_por_ano.reset_index()
bird_por_ano_df.columns = ["ano", "qtd_ocorrencias"]

bird_por_ano_df.to_csv(
    pasta_csvs + "bird_strike_por_ano.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# RESUMO DE CONFERÊNCIA DA ANÁLISE
# =========================
# como eu removi os prints do terminal, deixei as principais conferências em um CSV.
# esse arquivo serve para verificar rapidamente quantas ocorrências entraram em cada etapa.
resumo_bird = pd.DataFrame(
    [
        {
            "metrica": "ocorrencias_bird_unicas",
            "valor": df_bird["codigo_ocorrencia"].nunique()
        },
        {
            "metrica": "ocorrencias_com_data_valida",
            "valor": df_bird.dropna(subset=["mes"])["codigo_ocorrencia"].nunique()
        },
        {
            "metrica": "ocorrencias_com_hora_valida",
            "valor": df_bird.dropna(subset=["hora"])["codigo_ocorrencia"].nunique()
        },
        {
            "metrica": "ocorrencias_com_data_e_hora_validas",
            "valor": df_heat["codigo_ocorrencia"].nunique()
        },
        {
            "metrica": "soma_tabela_mensal",
            "valor": int(bird_por_mes.sum())
        },
        {
            "metrica": "soma_tabela_horaria",
            "valor": int(bird_por_hora.sum())
        },
        {
            "metrica": "soma_heatmap_mes_hora",
            "valor": int(heatmap_data.values.sum())
        }
    ]
)

resumo_bird.to_csv(
    pasta_csvs + "resumo_bird_strike.csv",
    index=False,
    encoding="utf-8-sig"
)