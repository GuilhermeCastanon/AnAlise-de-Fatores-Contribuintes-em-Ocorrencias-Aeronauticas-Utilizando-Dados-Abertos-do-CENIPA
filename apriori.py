# =========================
# BIBLIOTECAS
# =========================
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# =========================
# CONFIGURAÇÕES
# =========================
path = "Recursos/"

SUPORTES_GERAIS = [0.05, 0.03, 0.02, 0.01]
SUPORTES_FATORES = [0.03, 0.02, 0.01, 0.005, 0.002]

MIN_CONF_GERAL = 0.50
MIN_LIFT_GERAL = 1.10

MIN_CONF_FATOR = 0.20
MIN_LIFT_FATOR = 1.00

MAX_LEN_ITEMSET = 4


# =========================
# FUNÇÕES AUXILIARES
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


def criar_item(prefixo, valor):
    if not valor_valido(valor):
        return None

    valor = str(valor).strip().upper()
    return f"{prefixo}:{valor}"


def tem_prefixo(conjunto, prefixo):
    return any(str(item).startswith(prefixo) for item in conjunto)


def prefixos_itemset(itemset):
    return {
        str(item).split(":", 1)[0]
        for item in itemset
    }


def regra_com_mesma_variavel(linha):
    prefixos_antecedente = prefixos_itemset(linha["antecedents"])
    prefixos_consequente = prefixos_itemset(linha["consequents"])

    return len(prefixos_antecedente & prefixos_consequente) > 0


def formatar_itemset(itemset):
    return " + ".join(sorted(list(itemset)))


def faixa_horaria(hora):
    if pd.isna(hora):
        return None

    hora = int(hora)

    if 0 <= hora <= 5:
        return "MADRUGADA"
    elif 6 <= hora <= 11:
        return "MANHÃ"
    elif 12 <= hora <= 17:
        return "TARDE"
    elif 18 <= hora <= 23:
        return "NOITE"

    return None


def gerar_regras(transacoes, lista_suportes, min_conf, min_lift, nome_analise):
    te = TransactionEncoder()
    te_array = te.fit(transacoes).transform(transacoes)

    df_transacoes = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    print("\n==============================")
    print(nome_analise)
    print("==============================")
    print("Total de transações:", len(transacoes))
    print("Total de itens diferentes:", len(df_transacoes.columns))

    for min_support in lista_suportes:
        print("\nTestando min_support =", min_support)

        itens_frequentes = apriori(
            df_transacoes,
            min_support=min_support,
            use_colnames=True,
            max_len=MAX_LEN_ITEMSET
        )

        if itens_frequentes.empty:
            print("Nenhum itemset frequente encontrado.")
            continue

        regras = association_rules(
            itens_frequentes,
            metric="confidence",
            min_threshold=min_conf
        )

        if regras.empty:
            print("Nenhuma regra gerada.")
            continue

        regras = regras[
            (regras["confidence"] >= min_conf) &
            (regras["lift"] >= min_lift)
        ].copy()

        if regras.empty:
            print("Nenhuma regra passou nos filtros de confiança e lift.")
            continue

        # mantém apenas regras com um único consequente
        regras = regras[
            regras["consequents"].apply(lambda x: len(x) == 1)
        ].copy()

        # remove regras em que antecedente e consequente pertencem à mesma variável
        # exemplo: Tipo:X => Tipo:Y
        regras = regras[
            ~regras.apply(regra_com_mesma_variavel, axis=1)
        ].copy()

        if regras.empty:
            print("Nenhuma regra permaneceu após remover redundâncias.")
            continue

        regras = regras.sort_values(
            by=["lift", "confidence", "support"],
            ascending=False
        )

        print("Itemsets frequentes encontrados:", len(itens_frequentes))
        print("Regras encontradas:", len(regras))
        print("Suporte utilizado:", min_support)

        return itens_frequentes, regras, min_support, df_transacoes

    print("\nNenhuma regra encontrada para:", nome_analise)

    return (
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        df_transacoes
    )


def preparar_exportacao(regras):
    colunas = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]

    if regras.empty:
        return pd.DataFrame(columns=colunas)

    regras_export = regras[colunas].copy()

    regras_export["antecedents"] = regras_export["antecedents"].apply(formatar_itemset)
    regras_export["consequents"] = regras_export["consequents"].apply(formatar_itemset)

    regras_export["support"] = regras_export["support"].round(4)
    regras_export["confidence"] = regras_export["confidence"].round(4)
    regras_export["lift"] = regras_export["lift"].round(4)

    return regras_export


def preparar_itemsets_exportacao(itemsets):
    if itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    itemsets_export = itemsets.copy()
    itemsets_export["itemsets"] = itemsets_export["itemsets"].apply(formatar_itemset)
    itemsets_export["support"] = itemsets_export["support"].round(4)

    return itemsets_export


# =========================
# LEITURA DOS DADOS
# =========================
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")
df_aeronave = pd.read_csv(path + "aeronave.csv", sep=";", encoding="latin1")
df_fator = pd.read_csv(path + "fator_contribuinte.csv", sep=";", encoding="latin1")


# =========================
# LIMPEZA DOS NOMES DAS COLUNAS
# =========================
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()
df_aeronave.columns = df_aeronave.columns.str.strip()
df_fator.columns = df_fator.columns.str.strip()


# =========================
# PREPARAÇÃO DA TABELA OCORRÊNCIA
# =========================
df_ocorrencia_sel = df_ocorrencia[
    [
        "codigo_ocorrencia",
        #"ocorrencia_classificacao",
        "ocorrencia_uf",
        "ocorrencia_dia",
        "ocorrencia_hora"
    ]
].copy()

df_ocorrencia_sel = df_ocorrencia_sel.drop_duplicates(
    subset=["codigo_ocorrencia"]
)

df_ocorrencia_sel["ocorrencia_dia"] = pd.to_datetime(
    df_ocorrencia_sel["ocorrencia_dia"],
    errors="coerce",
    dayfirst=True
)

df_ocorrencia_sel["mes_num"] = df_ocorrencia_sel["ocorrencia_dia"].dt.month

nomes_meses = {
    1: "JANEIRO",
    2: "FEVEREIRO",
    3: "MARÇO",
    4: "ABRIL",
    5: "MAIO",
    6: "JUNHO",
    7: "JULHO",
    8: "AGOSTO",
    9: "SETEMBRO",
    10: "OUTUBRO",
    11: "NOVEMBRO",
    12: "DEZEMBRO"
}

df_ocorrencia_sel["mes"] = df_ocorrencia_sel["mes_num"].map(nomes_meses)

df_ocorrencia_sel["hora"] = pd.to_datetime(
    df_ocorrencia_sel["ocorrencia_hora"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

df_ocorrencia_sel["faixa_horaria"] = df_ocorrencia_sel["hora"].apply(faixa_horaria)


# =========================
# PREPARAÇÃO DA TABELA TIPO
# =========================
if "ocorrencia_tipo_categoria" in df_tipo.columns:
    coluna_tipo = "ocorrencia_tipo_categoria"
else:
    coluna_tipo = "ocorrencia_tipo"

df_tipo_sel = df_tipo[
    [
        "codigo_ocorrencia1",
        coluna_tipo
    ]
].copy()

df_tipo_sel = df_tipo_sel.rename(
    columns={
        "codigo_ocorrencia1": "codigo_ocorrencia",
        coluna_tipo: "ocorrencia_tipo"
    }
)

df_tipo_sel = df_tipo_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "ocorrencia_tipo"
    ]
)


# =========================
# PREPARAÇÃO DA TABELA AERONAVE
# =========================
if "aeronave_tipo_equipamento" in df_aeronave.columns:
    coluna_tipo_veiculo = "aeronave_tipo_equipamento"
elif "aeronave_tipo_veiculo" in df_aeronave.columns:
    coluna_tipo_veiculo = "aeronave_tipo_veiculo"
else:
    raise KeyError(
        "Não encontrei a coluna de tipo de equipamento/veículo em aeronave.csv.\n"
        "Colunas disponíveis:\n"
        f"{list(df_aeronave.columns)}"
    )

df_aeronave_sel = df_aeronave[
    [
        "codigo_ocorrencia2",
        coluna_tipo_veiculo,
        "aeronave_fase_operacao",
        "aeronave_tipo_operacao",
        "aeronave_nivel_dano"
    ]
].copy()

df_aeronave_sel = df_aeronave_sel.rename(
    columns={
        "codigo_ocorrencia2": "codigo_ocorrencia",
        coluna_tipo_veiculo: "aeronave_tipo_veiculo"
    }
)

df_aeronave_sel = df_aeronave_sel.drop_duplicates()


# =========================
# PREPARAÇÃO DA TABELA FATOR CONTRIBUINTE
# =========================
df_fator_sel = df_fator[
    [
        "codigo_ocorrencia3",
        "fator_nome"
    ]
].copy()

df_fator_sel = df_fator_sel.rename(
    columns={
        "codigo_ocorrencia3": "codigo_ocorrencia"
    }
)

df_fator_sel = df_fator_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "fator_nome"
    ]
)


# =========================
# CRIAÇÃO DAS TRANSAÇÕES
# =========================
codigos_ocorrencia = set(df_ocorrencia_sel["codigo_ocorrencia"].unique())

transacoes_dict = {
    codigo: set()
    for codigo in codigos_ocorrencia
}


# =========================
# ITENS DA TABELA OCORRÊNCIA
# =========================
for _, linha in df_ocorrencia_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    itens = {
        #"Classificacao": linha["ocorrencia_classificacao"],
        "UF": linha["ocorrencia_uf"],
        "Mes": linha["mes"],
        "Faixa_horaria": linha["faixa_horaria"]
    }

    for prefixo, valor in itens.items():
        item = criar_item(prefixo, valor)

        if item is not None:
            transacoes_dict[codigo].add(item)


# =========================
# ITENS DA TABELA TIPO
# =========================
for _, linha in df_tipo_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    item = criar_item("Tipo", linha["ocorrencia_tipo"])

    if item is not None:
        transacoes_dict[codigo].add(item)


# =========================
# ITENS DA TABELA AERONAVE
# =========================
for _, linha in df_aeronave_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    itens = {
        #"Tipo_veiculo": linha["aeronave_tipo_veiculo"],
        "Fase_operacao": linha["aeronave_fase_operacao"],
        #"Tipo_operacao": linha["aeronave_tipo_operacao"],
        "Nivel_dano": linha["aeronave_nivel_dano"]
    }

    for prefixo, valor in itens.items():
        item = criar_item(prefixo, valor)

        if item is not None:
            transacoes_dict[codigo].add(item)


# =========================
# ITENS DA TABELA FATOR CONTRIBUINTE
# =========================
for _, linha in df_fator_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    item = criar_item("Fator", linha["fator_nome"])

    if item is not None:
        transacoes_dict[codigo].add(item)


# =========================
# LISTA FINAL DE TRANSAÇÕES
# =========================

# regras gerais:
# usa apenas características da ocorrência, sem fatores contribuintes
transacoes_gerais = []

for itens in transacoes_dict.values():
    itens_sem_fator = [
        item for item in itens
        if not item.startswith("Fator:")
    ]

    if len(itens_sem_fator) > 1:
        transacoes_gerais.append(itens_sem_fator)


# regras com fator:
# usa apenas ocorrências que possuem fator contribuinte registrado
transacoes_com_fator = [
    list(itens)
    for itens in transacoes_dict.values()
    if len(itens) > 1 and any(item.startswith("Fator:") for item in itens)
]


print("\n==============================")
print("RESUMO DAS TRANSAÇÕES")
print("==============================")
print("Total de ocorrências:", len(transacoes_dict))
print("Transações gerais:", len(transacoes_gerais))
print("Transações com pelo menos um fator:", len(transacoes_com_fator))


# =========================
# REGRAS GERAIS
# =========================
itens_freq_gerais, regras_gerais, suporte_usado_geral, df_transacoes_gerais = gerar_regras(
    transacoes=transacoes_gerais,
    lista_suportes=SUPORTES_GERAIS,
    min_conf=MIN_CONF_GERAL,
    min_lift=MIN_LIFT_GERAL,
    nome_analise="REGRAS GERAIS"
)


# =========================
# REGRAS COM FATOR COMO CONSEQUENTE
# =========================
itens_freq_fatores, regras_fatores_todas, suporte_usado_fator, df_transacoes_fatores = gerar_regras(
    transacoes=transacoes_com_fator,
    lista_suportes=SUPORTES_FATORES,
    min_conf=MIN_CONF_FATOR,
    min_lift=MIN_LIFT_FATOR,
    nome_analise="REGRAS EM TRANSAÇÕES COM FATOR"
)

if not regras_fatores_todas.empty:
    regras_fatores = regras_fatores_todas[
        regras_fatores_todas["consequents"].apply(
            lambda x: len(x) == 1 and tem_prefixo(x, "Fator:")
        )
        &
        ~regras_fatores_todas["antecedents"].apply(
            lambda x: tem_prefixo(x, "Fator:")
        )
    ].copy()

    regras_fatores = regras_fatores.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    )
else:
    regras_fatores = pd.DataFrame()


# =========================
# EXPORTAÇÃO DOS RESULTADOS
# =========================
regras_gerais_export = preparar_exportacao(regras_gerais)
regras_fatores_export = preparar_exportacao(regras_fatores)

itens_freq_gerais_export = preparar_itemsets_exportacao(itens_freq_gerais)
itens_freq_fatores_export = preparar_itemsets_exportacao(itens_freq_fatores)

regras_gerais_export.to_csv(
    "regras_associacao_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_fatores_export.to_csv(
    "regras_associacao_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_gerais_export.head(20).to_csv(
    "top20_regras_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_fatores_export.head(20).to_csv(
    "top20_regras_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

itens_freq_gerais_export.to_csv(
    "itemsets_frequentes_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

itens_freq_fatores_export.to_csv(
    "itemsets_frequentes_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)


# =========================
# EXIBIÇÃO NO TERMINAL
# =========================
print("\n==============================")
print("TOP 20 REGRAS GERAIS")
print("==============================")

if regras_gerais_export.empty:
    print("Nenhuma regra geral encontrada.")
else:
    print(regras_gerais_export.head(20).to_string(index=False))


print("\n==============================")
print("TOP 20 REGRAS COM FATOR COMO CONSEQUENTE")
print("==============================")

if regras_fatores_export.empty:
    print("Nenhuma regra com fator como consequente encontrada.")
else:
    print(regras_fatores_export.head(20).to_string(index=False))


print("\n==============================")
print("ARQUIVOS GERADOS")
print("==============================")
print("- regras_associacao_gerais.csv")
print("- regras_associacao_fatores.csv")
print("- top20_regras_gerais.csv")
print("- top20_regras_fatores.csv")
print("- itemsets_frequentes_gerais.csv")
print("- itemsets_frequentes_fatores.csv")

print("\nSuporte usado nas regras gerais:", suporte_usado_geral)
print("Suporte usado nas regras com fatores:", suporte_usado_fator)