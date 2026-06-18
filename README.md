# Análise de Ocorrências Aeronáuticas com Dados Abertos do CENIPA

Este repositório contém os códigos desenvolvidos para o Trabalho de Conclusão de Curso em Engenharia de Computação, cujo objetivo é realizar uma análise exploratória de dados públicos de ocorrências aeronáuticas da aviação civil brasileira, disponibilizados pelo CENIPA por meio da base OpenData AIG Brazil.

O projeto busca identificar padrões relacionados aos tipos de ocorrência, fatores contribuintes e aspectos temporais dos eventos registrados. Além disso, aplica regras de associação utilizando o algoritmo Apriori, com foco em encontrar combinações de características associadas a fatores contribuintes.

## Objetivos

- Realizar a leitura, limpeza e integração dos arquivos CSV da base do CENIPA;
- Identificar os tipos de ocorrência mais frequentes;
- Analisar fatores contribuintes associados a determinados tipos de ocorrência;
- Investigar a distribuição temporal de ocorrências de colisão com ave;
- Aplicar regras de associação com o algoritmo Apriori;
- Avaliar limitações da base e dos padrões encontrados.

## Base de Dados

A base utilizada é a OpenData AIG Brazil, disponibilizada pelo Centro de Investigação e Prevenção de Acidentes Aeronáuticos (CENIPA). Os principais arquivos utilizados foram:

- `ocorrencia.csv`
- `ocorrencia_tipo.csv`
- `aeronave.csv`
- `fator_contribuinte.csv`

Esses arquivos contêm informações sobre ocorrências aeronáuticas, aeronaves envolvidas, tipos de ocorrência e fatores contribuintes registrados em investigações finalizadas.

## Estrutura do Projeto

```text
.
├── Recursos/
│   ├── ocorrencia.csv
│   ├── ocorrencia_tipo.csv
│   ├── aeronave.csv
│   └── fator_contribuinte.csv
├── main.py
├── bird_strike.py
├── apriori.py
├── README.md
└── ...
