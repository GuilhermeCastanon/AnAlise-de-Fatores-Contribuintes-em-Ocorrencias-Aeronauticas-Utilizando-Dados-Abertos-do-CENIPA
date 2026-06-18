## Estrutura do Projeto

A organização do repositório foi definida de forma a separar os códigos-fonte, os arquivos de entrada, os resultados em CSV e as imagens geradas durante as análises.

```text
.
├── main.py
├── apriori.py
├── bird_strike.py
├── ocorrencia_fatores.py
├── README.md
│
├── Recursos/
│   ├── ocorrencia.csv
│   ├── ocorrencia_tipo.csv
│   ├── aeronave.csv
│   └── fator_contribuinte.csv
│
├── Imagens/
│   ├── grafico_top5_tipos_ocorrencia.png
│   ├── ocorrencias_bird_por_mes.png
│   ├── ocorrencias_bird_por_hora.png
│   ├── ocorrencias_bird_por_mes_2023_2025.png
│   └── heatmap_bird_mes_hora.png
│
└── CSVs/
    ├── regras_associacao_fatores.csv
    ├── top20_regras_fatores.csv
    ├── itemsets_frequentes_fatores.csv
    └── demais resultados gerados pelas análises
```

### Descrição das pastas

* `Recursos/`: contém os arquivos CSV originais utilizados como entrada nas análises.
* `Imagens/`: armazena os gráficos e visualizações gerados pelos scripts.
* `CSVs/`: contém os arquivos CSV produzidos como resultado das análises.
* Arquivos `.py`: contêm os códigos responsáveis pelo processamento, análise exploratória, estudo de colisões com ave e aplicação do algoritmo Apriori.

### Scripts principais

* `ocorrencia_fatores.py`: analisa os fatores contribuintes associados aos principais tipos de ocorrência.
* `bird_strike.py`: realiza a análise específica das ocorrências de colisão com ave, incluindo gráficos por mês, por hora e mapa de calor.
* `apriori.py`: aplica o algoritmo Apriori para geração de regras de associação envolvendo fatores contribuintes.

