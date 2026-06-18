# =========================
# BIBLIOTECAS
# =========================
# uso o pandas para preparar as tabelas e montar as transações.
# o TransactionEncoder transforma as transações em uma matriz booleana,
# que é o formato esperado pelo algoritmo Apriori da biblioteca mlxtend.
import os

import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# =========================
# CONFIGURAÇÕES GERAIS
# =========================
# aqui eu deixo centralizados os caminhos e os parâmetros principais da mineração.
# pensei em separar isso no começo do código porque facilita ajustar os suportes,
# a confiança mínima e o lift mínimo sem precisar procurar esses valores no meio do script.
path = "Recursos/"
pasta_csvs = "CSVs/"

os.makedirs(pasta_csvs, exist_ok=True)

# suportes testados para regras gerais, isto é, regras sem considerar fatores contribuintes.
# esses valores são testados em ordem até que alguma regra útil seja encontrada.
SUPORTES_GERAIS = [0.05, 0.03, 0.02, 0.01]

# suportes testados para regras que envolvem fatores contribuintes.
# usei uma lista com valores menores porque a quantidade de ocorrências com fator registrado
# é bem menor do que a quantidade total de ocorrências da base.
SUPORTES_FATORES = [0.03, 0.02, 0.01, 0.005, 0.002]

# parâmetros das regras gerais.
MIN_CONF_GERAL = 0.50
MIN_LIFT_GERAL = 1.10

# parâmetros das regras envolvendo fatores contribuintes.
# deixei a confiança mínima menor porque os fatores aparecem em um subconjunto limitado
# da base, então filtros muito rígidos poderiam eliminar associações relevantes.
MIN_CONF_FATOR = 0.20
MIN_LIFT_FATOR = 1.00

# limite máximo de itens em cada itemset.
# isso evita combinações muito grandes, difíceis de interpretar na apresentação.
MAX_LEN_ITEMSET = 4


# =========================
# FUNÇÃO PARA IDENTIFICAR VALORES VÁLIDOS
# =========================
# esta função concentra a regra de limpeza de valores inválidos.
# pensei em fazer assim porque várias tabelas possuem campos vazios, desconhecidos
# ou não informados, e todos eles devem ser tratados da mesma forma.
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
# FUNÇÃO PARA CRIAR ITENS DAS TRANSAÇÕES
# =========================
# cada característica da ocorrência é transformada em um item no formato Prefixo:Valor.
# por exemplo: Tipo:FALHA DO MOTOR EM VOO ou Fator:MANUTENÇÃO DA AERONAVE.
# fiz isso porque o Apriori trabalha com itens, então preciso transformar as colunas
# da base em uma lista de características associadas a cada ocorrência.
def criar_item(prefixo, valor):
    if not valor_valido(valor):
        return None

    valor = str(valor).strip().upper()
    return f"{prefixo}:{valor}"


# =========================
# FUNÇÃO PARA VERIFICAR PREFIXOS
# =========================
# esta função verifica se um itemset possui algum item com determinado prefixo.
# uso isso principalmente para identificar regras cujo consequente é um fator contribuinte.
def tem_prefixo(conjunto, prefixo):
    return any(str(item).startswith(prefixo) for item in conjunto)


# =========================
# FUNÇÃO PARA EXTRAIR OS PREFIXOS DE UM ITEMSET
# =========================
# aqui eu separo apenas a parte antes dos dois-pontos.
# por exemplo, em Tipo:FALHA DO MOTOR EM VOO, o prefixo é Tipo.
# isso ajuda a identificar regras redundantes entre itens da mesma variável.
def prefixos_itemset(itemset):
    return {
        str(item).split(":", 1)[0]
        for item in itemset
    }


# =========================
# FUNÇÃO PARA REMOVER REGRAS DA MESMA VARIÁVEL
# =========================
# esta função verifica se o antecedente e o consequente pertencem à mesma variável.
# pensei em remover esse tipo de regra porque ela pode gerar associações pouco úteis,
# como Tipo:X => Tipo:Y, que normalmente não acrescentam interpretação relevante
# para o objetivo do trabalho.
def regra_com_mesma_variavel(linha):
    prefixos_antecedente = prefixos_itemset(linha["antecedents"])
    prefixos_consequente = prefixos_itemset(linha["consequents"])

    return len(prefixos_antecedente & prefixos_consequente) > 0


# =========================
# FUNÇÃO PARA FORMATAR ITEMSETS
# =========================
# os itemsets gerados pelo mlxtend aparecem como frozenset.
# esta função transforma esses conjuntos em texto legível para salvar nos arquivos CSV.
def formatar_itemset(itemset):
    return " + ".join(sorted(list(itemset)))


# =========================
# FUNÇÃO PARA CRIAR FAIXA HORÁRIA
# =========================
# aqui eu agrupo as horas do dia em quatro períodos.
# pensei em usar faixas horárias em vez da hora exata porque isso gera itens mais gerais
# e mais interpretáveis nas regras de associação.
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


# =========================
# FUNÇÃO PARA GERAR REGRAS DE ASSOCIAÇÃO
# =========================
# esta é a função central da mineração de padrões.
# ela recebe uma lista de transações, testa diferentes valores de suporte mínimo
# e retorna as primeiras regras que passam pelos filtros definidos.
#
# pensei em testar suportes em sequência porque nem sempre um suporte alto encontra regras.
# então o código começa com critérios mais rígidos e vai flexibilizando até encontrar
# regras interpretáveis.
def gerar_regras(transacoes, lista_suportes, min_conf, min_lift, nome_analise):
    historico = []

    if len(transacoes) == 0:
        historico.append(
            {
                "analise": nome_analise,
                "min_support": None,
                "total_transacoes": 0,
                "total_itens_diferentes": 0,
                "itemsets_frequentes": 0,
                "regras_apos_filtros": 0,
                "status": "sem_transacoes"
            }
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            pd.DataFrame(),
            pd.DataFrame(historico)
        )

    # o TransactionEncoder transforma a lista de transações em uma matriz de True/False.
    # cada coluna representa um item possível, e cada linha representa uma ocorrência.
    te = TransactionEncoder()
    te_array = te.fit(transacoes).transform(transacoes)

    df_transacoes = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    for min_support in lista_suportes:
        itens_frequentes = apriori(
            df_transacoes,
            min_support=min_support,
            use_colnames=True,
            max_len=MAX_LEN_ITEMSET
        )

        if itens_frequentes.empty:
            historico.append(
                {
                    "analise": nome_analise,
                    "min_support": min_support,
                    "total_transacoes": len(transacoes),
                    "total_itens_diferentes": len(df_transacoes.columns),
                    "itemsets_frequentes": 0,
                    "regras_apos_filtros": 0,
                    "status": "sem_itemsets_frequentes"
                }
            )

            continue

        regras = association_rules(
            itens_frequentes,
            metric="confidence",
            min_threshold=min_conf
        )

        if regras.empty:
            historico.append(
                {
                    "analise": nome_analise,
                    "min_support": min_support,
                    "total_transacoes": len(transacoes),
                    "total_itens_diferentes": len(df_transacoes.columns),
                    "itemsets_frequentes": len(itens_frequentes),
                    "regras_apos_filtros": 0,
                    "status": "sem_regras"
                }
            )

            continue

        # aplico os filtros mínimos de confiança e lift.
        # a confiança mede a frequência com que o consequente aparece quando o antecedente aparece.
        # o lift ajuda a verificar se a associação é mais forte do que seria esperado ao acaso.
        regras = regras[
            (regras["confidence"] >= min_conf) &
            (regras["lift"] >= min_lift)
        ].copy()

        if regras.empty:
            historico.append(
                {
                    "analise": nome_analise,
                    "min_support": min_support,
                    "total_transacoes": len(transacoes),
                    "total_itens_diferentes": len(df_transacoes.columns),
                    "itemsets_frequentes": len(itens_frequentes),
                    "regras_apos_filtros": 0,
                    "status": "sem_regras_apos_confidence_lift"
                }
            )

            continue

        # mantenho apenas regras com um único consequente.
        # pensei em fazer assim porque regras com vários consequentes são mais difíceis
        # de explicar e menos adequadas para a discussão do TCC.
        regras = regras[
            regras["consequents"].apply(lambda x: len(x) == 1)
        ].copy()

        # removo regras em que antecedente e consequente pertencem à mesma variável.
        # isso evita relações redundantes e pouco interpretáveis.
        regras = regras[
            ~regras.apply(regra_com_mesma_variavel, axis=1)
        ].copy()

        if regras.empty:
            historico.append(
                {
                    "analise": nome_analise,
                    "min_support": min_support,
                    "total_transacoes": len(transacoes),
                    "total_itens_diferentes": len(df_transacoes.columns),
                    "itemsets_frequentes": len(itens_frequentes),
                    "regras_apos_filtros": 0,
                    "status": "sem_regras_apos_remover_redundancias"
                }
            )

            continue

        # ordeno as regras priorizando lift, confiança e suporte.
        # isso faz com que as primeiras regras sejam as mais interessantes
        # segundo os critérios adotados no trabalho.
        regras = regras.sort_values(
            by=["lift", "confidence", "support"],
            ascending=False
        )

        historico.append(
            {
                "analise": nome_analise,
                "min_support": min_support,
                "total_transacoes": len(transacoes),
                "total_itens_diferentes": len(df_transacoes.columns),
                "itemsets_frequentes": len(itens_frequentes),
                "regras_apos_filtros": len(regras),
                "status": "regras_encontradas"
            }
        )

        return itens_frequentes, regras, min_support, df_transacoes, pd.DataFrame(historico)

    return (
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        df_transacoes,
        pd.DataFrame(historico)
    )


# =========================
# FUNÇÃO PARA PREPARAR A EXPORTAÇÃO DAS REGRAS
# =========================
# esta função deixa as regras em formato mais legível antes de salvar em CSV.
# sem isso, os antecedentes e consequentes ficariam como objetos do tipo frozenset.
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


# =========================
# FUNÇÃO PARA PREPARAR A EXPORTAÇÃO DOS ITEMSETS
# =========================
# além das regras, salvo também os itemsets frequentes.
# isso ajuda a verificar quais combinações de itens foram consideradas frequentes
# antes da geração das regras de associação.
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
# nesta etapa carrego as quatro tabelas usadas para montar as transações.
# cada ocorrência será tratada como uma transação, e as características associadas
# a ela serão transformadas em itens.
df_ocorrencia = pd.read_csv(path + "ocorrencia.csv", sep=";", encoding="latin1")
df_tipo = pd.read_csv(path + "ocorrencia_tipo.csv", sep=";", encoding="latin1")
df_aeronave = pd.read_csv(path + "aeronave.csv", sep=";", encoding="latin1")
df_fator = pd.read_csv(path + "fator_contribuinte.csv", sep=";", encoding="latin1")


# =========================
# PADRONIZAÇÃO DOS NOMES DAS COLUNAS
# =========================
# removo espaços extras dos nomes das colunas para evitar erro de seleção depois.
df_ocorrencia.columns = df_ocorrencia.columns.str.strip()
df_tipo.columns = df_tipo.columns.str.strip()
df_aeronave.columns = df_aeronave.columns.str.strip()
df_fator.columns = df_fator.columns.str.strip()


# =========================
# PREPARAÇÃO DA TABELA DE OCORRÊNCIAS
# =========================
# aqui seleciono as informações gerais da ocorrência.
# deixei a classificação fora das transações finais porque ela gerava associações
# muito óbvias com outras variáveis e pouco úteis para a interpretação dos fatores.
df_ocorrencia_sel = df_ocorrencia[
    [
        "codigo_ocorrencia",
        "ocorrencia_uf",
        "ocorrencia_dia",
        "ocorrencia_hora"
    ]
].copy()

df_ocorrencia_sel = df_ocorrencia_sel.drop_duplicates(
    subset=["codigo_ocorrencia"]
)

# converto a data para extrair o mês da ocorrência.
# o mês entra como item porque pode existir algum padrão temporal associado às ocorrências.
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

# converto o horário para extrair apenas a hora e depois agrupar em faixa horária.
df_ocorrencia_sel["hora"] = pd.to_datetime(
    df_ocorrencia_sel["ocorrencia_hora"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

df_ocorrencia_sel["faixa_horaria"] = df_ocorrencia_sel["hora"].apply(faixa_horaria)


# =========================
# PREPARAÇÃO DA TABELA DE TIPOS DE OCORRÊNCIA
# =========================
# nesta tabela eu busco o tipo de ocorrência associado a cada código.
# novamente deixo a verificação de coluna porque a base pode mudar um pouco
# dependendo da versão baixada.
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

# removo apenas duplicatas da mesma combinação ocorrência + tipo.
# uma ocorrência pode ter mais de um tipo, e isso deve ser preservado.
df_tipo_sel = df_tipo_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "ocorrencia_tipo"
    ]
)


# =========================
# PREPARAÇÃO DA TABELA DE AERONAVES
# =========================
# a coluna que representa o tipo de equipamento pode aparecer com nomes diferentes.
# por isso faço essa verificação antes de selecionar as colunas.
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

# nesta análise, mantive principalmente fase de operação e nível de dano.
# pensei em usar essas variáveis porque elas ajudam a contextualizar a ocorrência
# sem deixar as transações excessivamente específicas.
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
# PREPARAÇÃO DA TABELA DE FATORES CONTRIBUINTES
# =========================
# aqui eu preparo os fatores contribuintes, que são os principais consequentes
# de interesse nas regras finais.
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

# removo duplicatas da mesma combinação ocorrência + fator.
# isso evita contar duas vezes o mesmo fator dentro da mesma ocorrência.
df_fator_sel = df_fator_sel.drop_duplicates(
    subset=[
        "codigo_ocorrencia",
        "fator_nome"
    ]
)


# =========================
# CRIAÇÃO DA ESTRUTURA DAS TRANSAÇÕES
# =========================
# cada ocorrência será uma transação.
# começo criando um dicionário em que a chave é o código da ocorrência
# e o valor é um conjunto de itens associados a ela.
codigos_ocorrencia = set(df_ocorrencia_sel["codigo_ocorrencia"].unique())

transacoes_dict = {
    codigo: set()
    for codigo in codigos_ocorrencia
}


# =========================
# ADIÇÃO DOS ITENS DA TABELA DE OCORRÊNCIAS
# =========================
# aqui adiciono itens gerais da ocorrência, como UF, mês e faixa horária.
# esses itens entram como possíveis antecedentes nas regras.
for _, linha in df_ocorrencia_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    itens = {
        "UF": linha["ocorrencia_uf"],
        "Mes": linha["mes"],
        "Faixa_horaria": linha["faixa_horaria"]
    }

    for prefixo, valor in itens.items():
        item = criar_item(prefixo, valor)

        if item is not None:
            transacoes_dict[codigo].add(item)


# =========================
# ADIÇÃO DOS ITENS DA TABELA DE TIPOS
# =========================
# nesta parte adiciono o tipo de ocorrência à transação.
# uma mesma ocorrência pode ter mais de um tipo, então todos os tipos válidos
# são adicionados ao conjunto de itens daquela ocorrência.
for _, linha in df_tipo_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    item = criar_item("Tipo", linha["ocorrencia_tipo"])

    if item is not None:
        transacoes_dict[codigo].add(item)


# =========================
# ADIÇÃO DOS ITENS DA TABELA DE AERONAVES
# =========================
# aqui adiciono informações da aeronave associada à ocorrência.
# mantive fase de operação e nível de dano porque são atributos interpretáveis
# e úteis para relacionar o contexto da ocorrência com possíveis fatores contribuintes.
for _, linha in df_aeronave_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    itens = {
        "Fase_operacao": linha["aeronave_fase_operacao"],
        "Nivel_dano": linha["aeronave_nivel_dano"]
    }

    for prefixo, valor in itens.items():
        item = criar_item(prefixo, valor)

        if item is not None:
            transacoes_dict[codigo].add(item)


# =========================
# ADIÇÃO DOS ITENS DA TABELA DE FATORES CONTRIBUINTES
# =========================
# nesta seção adiciono os fatores contribuintes às transações.
# esses itens são fundamentais para a análise final, porque o objetivo principal
# é encontrar características que apareçam associadas a determinados fatores.
for _, linha in df_fator_sel.iterrows():
    codigo = linha["codigo_ocorrencia"]

    if codigo not in transacoes_dict:
        continue

    item = criar_item("Fator", linha["fator_nome"])

    if item is not None:
        transacoes_dict[codigo].add(item)


# =========================
# CRIAÇÃO DA LISTA FINAL DE TRANSAÇÕES GERAIS
# =========================
# as transações gerais usam apenas características da ocorrência, sem fatores.
# pensei em separar essa etapa porque regras gerais podem mostrar relações internas
# da base, mas não são o foco principal da interpretação final do TCC.
transacoes_gerais = []

for itens in transacoes_dict.values():
    itens_sem_fator = [
        item for item in itens
        if not item.startswith("Fator:")
    ]

    if len(itens_sem_fator) > 1:
        transacoes_gerais.append(itens_sem_fator)


# =========================
# CRIAÇÃO DA LISTA FINAL DE TRANSAÇÕES COM FATORES
# =========================
# aqui mantenho apenas ocorrências que possuem pelo menos um fator contribuinte registrado.
# isso é necessário porque, para gerar regras com fator como consequente, a transação
# precisa conter algum item do tipo Fator.
transacoes_com_fator = [
    list(itens)
    for itens in transacoes_dict.values()
    if len(itens) > 1 and any(item.startswith("Fator:") for item in itens)
]


# =========================
# RESUMO DAS TRANSAÇÕES
# =========================
# como removi os prints do terminal, salvo um resumo em CSV.
# esse arquivo ajuda a conferir quantas ocorrências viraram transações em cada análise.
resumo_transacoes = pd.DataFrame(
    [
        {
            "metrica": "total_ocorrencias_base",
            "valor": len(transacoes_dict)
        },
        {
            "metrica": "transacoes_gerais",
            "valor": len(transacoes_gerais)
        },
        {
            "metrica": "transacoes_com_pelo_menos_um_fator",
            "valor": len(transacoes_com_fator)
        }
    ]
)

resumo_transacoes.to_csv(
    pasta_csvs + "resumo_transacoes_apriori.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# GERAÇÃO DAS REGRAS GERAIS
# =========================
# aqui gero regras sem fatores contribuintes.
# elas são mantidas como uma análise complementar, mas a interpretação principal
# do trabalho fica nas regras em que o consequente é um fator contribuinte.
itens_freq_gerais, regras_gerais, suporte_usado_geral, df_transacoes_gerais, historico_geral = gerar_regras(
    transacoes=transacoes_gerais,
    lista_suportes=SUPORTES_GERAIS,
    min_conf=MIN_CONF_GERAL,
    min_lift=MIN_LIFT_GERAL,
    nome_analise="REGRAS GERAIS"
)


# =========================
# GERAÇÃO DAS REGRAS EM TRANSAÇÕES COM FATOR
# =========================
# nesta etapa gero regras usando apenas transações que possuem fatores.
# ainda assim, depois faço um filtro adicional para garantir que o consequente
# da regra seja um fator contribuinte.
itens_freq_fatores, regras_fatores_todas, suporte_usado_fator, df_transacoes_fatores, historico_fator = gerar_regras(
    transacoes=transacoes_com_fator,
    lista_suportes=SUPORTES_FATORES,
    min_conf=MIN_CONF_FATOR,
    min_lift=MIN_LIFT_FATOR,
    nome_analise="REGRAS EM TRANSAÇÕES COM FATOR"
)


# =========================
# FILTRO FINAL DAS REGRAS COM FATOR COMO CONSEQUENTE
# =========================
# aqui eu mantenho apenas regras do tipo:
# característica da ocorrência => fator contribuinte.
#
# pensei em fazer esse filtro porque uma regra com fator no antecedente seria menos útil
# para o objetivo do trabalho. o interesse é observar quais características aparecem
# associadas a determinados fatores contribuintes, e não o contrário.
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
# PREPARAÇÃO DOS RESULTADOS PARA EXPORTAÇÃO
# =========================
# antes de salvar os resultados, converto os itemsets e as regras para uma forma
# mais legível. isso facilita abrir os arquivos no Excel, no VS Code ou usar os dados
# diretamente nas tabelas da monografia.
regras_gerais_export = preparar_exportacao(regras_gerais)
regras_fatores_export = preparar_exportacao(regras_fatores)

itens_freq_gerais_export = preparar_itemsets_exportacao(itens_freq_gerais)
itens_freq_fatores_export = preparar_itemsets_exportacao(itens_freq_fatores)


# =========================
# EXPORTAÇÃO DAS REGRAS
# =========================
# salvo as regras em arquivos separados.
# as regras com fatores são as mais importantes para a discussão final do trabalho.
regras_gerais_export.to_csv(
    pasta_csvs + "regras_associacao_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_fatores_export.to_csv(
    pasta_csvs + "regras_associacao_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_gerais_export.head(20).to_csv(
    pasta_csvs + "top20_regras_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

regras_fatores_export.head(20).to_csv(
    pasta_csvs + "top20_regras_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)


# =========================
# EXPORTAÇÃO DOS ITEMSETS FREQUENTES
# =========================
# além das regras, salvo os itemsets frequentes que serviram de base para gerá-las.
# isso é útil para conferir quais combinações passaram pelo critério de suporte.
itens_freq_gerais_export.to_csv(
    pasta_csvs + "itemsets_frequentes_gerais.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

itens_freq_fatores_export.to_csv(
    pasta_csvs + "itemsets_frequentes_fatores.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)


# =========================
# EXPORTAÇÃO DO HISTÓRICO DE EXECUÇÃO
# =========================
# como não há mais prints no terminal, salvo também um histórico das tentativas
# de suporte mínimo. esse arquivo mostra qual suporte foi testado e se gerou regras.
historico_execucao = pd.concat(
    [
        historico_geral,
        historico_fator
    ],
    ignore_index=True
)

historico_execucao.to_csv(
    pasta_csvs + "resumo_execucao_apriori.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# RESUMO DOS SUPORTES UTILIZADOS
# =========================
# este arquivo guarda diretamente o suporte final usado em cada análise.
# isso ajuda a justificar os parâmetros utilizados na apresentação e na monografia.
resumo_suportes = pd.DataFrame(
    [
        {
            "analise": "regras_gerais",
            "suporte_usado": suporte_usado_geral,
            "min_conf": MIN_CONF_GERAL,
            "min_lift": MIN_LIFT_GERAL
        },
        {
            "analise": "regras_com_fatores",
            "suporte_usado": suporte_usado_fator,
            "min_conf": MIN_CONF_FATOR,
            "min_lift": MIN_LIFT_FATOR
        }
    ]
)

resumo_suportes.to_csv(
    pasta_csvs + "resumo_suportes_apriori.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)