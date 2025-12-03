<details open>

<summary><i>Visão geral do projeto</i></summary>

<h2><b>
    Visão geral do projeto
</b></h2>

A classificação correta da qualidade dos vinhos é extremamente importante para produtores e distribuidores, já que a venda de vinhos classificados incorretamente pode resultar em <b>reembolso à empresa</b>. Por esse motivo, desenvolvi um algoritmo de aprendizado de máquina personalizável para diferentes casos de negócios. A precisão geral do algoritmo foi de quase 74%, assumindo que todas as qualidades de vinhos têm a mesma importância, o que nem sempre é o caso, pois se um comerciante vende mais vinhos de uma qualidade específica, é importante que ele tenha um melhor desempenho de classificação nessa categoria, em vez das outras, reduzindo o custo do reembolso. A personalização do algoritmo é explicada na seção Teste e ajuste fino do modelo. 

Usei o Banco de Dados de Qualidade do Vinho fornecido pelo artigo “[Modelagem de preferências de vinho por mineração de dados a partir de propriedades físico-químicas](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub)”. Os pesquisadores utilizaram o algoritmo <b>Support Vector Machine (SVM)</b> para prever a qualidade dos vinhos, mas não conseguiram bons resultados, com precisão de aproximadamente 63%. Isso mostra que a previsão da qualidade do vinho não é uma tarefa fácil, e talvez seja devido à distribuição desequilibrada dos dados, como discuto na seção “Manipulação de dados”.

Minha proposta é combinar algumas qualidade de vinho em uma única categoria para equilibrar a distribuição dos dados. Além disso, foi realizado um processo de engenharia e seleção de features, que eliminou a multicolinearidade entre elas, e sobre a parte de treinamento, foi aplicado o procedimento de cross-validation, que proporcionou uma boa generalização do melhor algoritmo, sugerida por seu desempenho no conjunto de dados de teste. No final, o conhecimento de negócios foi aplicado para melhorar ainda mais as previsões. 

Clique em “<i>Parâmetros do vinho</i>” para obter mais informações sobre as características do banco de dados.

<details>

<summary><h3><b><i>Parâmetros do vinho</i></b></h3></summary>

<details>
<summary><b>Fixed Acidity (g/L)</b></summary>

<i>Os ácidos são responsáveis pela "corpo", quanto maior a concentração, mais o vinho fica encorpado, azedo. Os [ácidos fixos](https://waterhouse.ucdavis.edu/whats-in-wine/fixed-acidity) encontrados nos vinhos são o tartárico, o málico, o cítrico e o succínico.</i>
</details>

<details>
<summary><b>Volatile Acidity (g/L)</b></summary>

<i>O [ácido volátil](https://www.awri.com.au/wp-content/uploads/2018/03/s1982.pdf) mais comum é o ácido acético, responsável pelo cheiro do vinho devido à evaporação. Além disso, o ácido sulfúrico também é um ácido volátil.</i>
</details>

<details>
<summary><b>Citric Acid (g/L)</b></summary>

<i>É um [ácido fixo](https://waterhouse.ucdavis.edu/whats-in-wine/fixed-acidity) encontrado na faixa de 0 a 0,5 g/L.</i>
</details>

<details>
<summary><b>Residual Sugar (g/L)</b></summary>

<i>É o [açúcar natural](https://winefolly.com/deep-dive/what-is-residual-sugar-in-wine/) das uvas que permanece no vinho após a interrupção do processo de fermentação. Sua quantidade determina a doçura do vinho.</i>
</details>

<details>
<summary><b>Chlorides (g/L)</b></summary>

<i>Os cloretos influenciam a “suavidade” e a persistência do sabor. As uvas cultivadas em regiões próximas ao mar tendem a produzir um suco com maior [teor de cloreto](https://www.awri.com.au/wp-content/uploads/2018/08/s1530.pdf).</i>
</details>

<details>
<summary><b>Sulphates (g/L)</b></summary>

<i>Os sulfatos são responsáveis pela [atividade antioxidante e antimicrobiana](https://www.lasommeliere.com/en/blog/sulfites-in-wine-what-are-they-and-what-do-they-do--n520), atuando como conservantes para os vinhos.</i>
</details>

<details>
<summary><b>Free Sulfur Dioxide (mg/L)</b></summary>

<i>Tal como os sulfatos, o [dióxido de enxofre livre](https://extension.okstate.edu/fact-sheets/understanding-free-sulfur-dioxide-fso2-in-wine.html) atua como conservante no vinho. Tende a ligar-se a outras moléculas, perdendo a sua ação conservante.</i>
</details>

<details>
<summary><b>Total Sulfur Dioxide (mg/L)</b></summary>

<i>Basicamente, é a [soma do dióxido de enxofre livre e do dióxido de enxofre ligado](https://www.oiv.int/public/medias/7840/oiv-collective-expertise-document-so2-and-wine-a-review.pdf).</i>
</details>

<details>
<summary><b>Density (g/mL)</b></summary>

<i>É um parâmetro importante para monitorar o processo de fermentação. Uma vez estabilizada, pode estar relacionada com a suavidade do vinho.</i>
</details>

<details>
<summary><b>pH</b></summary>

<i>É uma [medida da acidez do vinho](https://www.awri.com.au/industry_support/winemaking_resources/frequently_asked_questions/acidity_and_ph/). Um pH elevado significa que há mais íons de hidrogênio livres disponíveis para se ligarem ao dióxido de enxofre livre. Portanto, esses dois parâmetros devem se combinar para proporcionar a sensação perfeita da acidez desejada e evitar a deterioração do vinho.</i>
</details>

<details>
<summary><b>Alcohol (%)</b></summary>

<i>Atua como conservante, mas também é responsável pela [sensação de queimação](https://vinaliawine.com/blogs/our-journal/alcohol-and-its-role-in-wine?srsltid=AfmBOoooYR_PZUzfbiqLh8isStKaKnK6DNTravGMLjqb9kQZBiRmL9m6) do vinho</i>
</details>

</details>

</details>

As informações sobre o desenvolvimento do modelo são fornecidas nas seguintes guias. Basta <b>clicar nelas</b> para abrir a explicação.

<details>

<summary><i>Criando um banco de dados SQL e conectando o Python ao banco de dados SQL Server</i></summary>

<h2><b>
    Criando um banco de dados SQL e conectando o Python ao banco de dados SQL Server
</b></h2>

Para criar o banco de dados SQL com as features dos vinhos, podemos executar o [código](https://github.com/L-Loreti/Wine-Quality-Classifier/blob/main/src/CREATE_WINE_DATABASE.sql).

Com o banco de dados dos vinhos montado no SQL Server, podemos usar a biblioteca <i>mysql</i> para conectar o Python a ele e obter a tabela de features químicos:
```python
connection = connector.connect(
  host = '127.0.0.1',
  user = 'Leonardo-Loreti',
  password = '########',
  database = 'WineQT')

query = 'SELECT * FROM WineData'

wine = pd.read_sql(query, con = connection)
```

</details>
