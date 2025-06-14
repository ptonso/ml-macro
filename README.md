Download data with

```bash
python3 -m src.fetch.pipeline
```

Para escolher a variável resposta basta mudar "TARGET_VARIABLE" no notebook. As implementações das funções estão no arquivo lasso.py. As funções precisam de uma base de dados em .csv para rodarem, como estamos trabalhando com um modelo por país, precisamos de uma base para cada país. O seguinte exemplo mostra o comando que transforma a nossa base geral para as específicas:

```bash
nome_do_pais = "brazil" 
dados_do_pais = dataset.get_country_data(nome_do_pais)
```
