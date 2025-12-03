<details open>

<summary><i>Visão geral do projeto</i></summary>

<h2><b>
    Visão geral do projeto
</b></h2>

A classificação correta da qualidade dos vinhos é extremamente importante para produtores e distribuidores, já que a venda de vinhos classificados incorretamente pode resultar em <b>reembolso à empresa</b>. Por esse motivo, desenvolvi um algoritmo de aprendizado de máquina personalizável para diferentes casos de negócios. A precisão geral do algoritmo foi de quase 74%, assumindo que todas as qualidades de vinhos têm a mesma importância, o que nem sempre é o caso, pois se um comerciante vende mais vinhos de uma qualidade específica, é importante que ele tenha um melhor desempenho de classificação nessa categoria, em vez das outras, reduzindo o custo do reembolso. A personalização do algoritmo é explicada na seção Teste e ajuste fino do modelo. 

Usei o Banco de Dados de Qualidade do Vinho fornecido pelo artigo “[Modelagem de preferências de vinho por mineração de dados a partir de propriedades físico-químicas](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377?via%3Dihub)”. Os pesquisadores utilizaram o algoritmo <b>Support Vector Machine (SVM)</b> para prever a qualidade dos vinhos, mas não conseguiram bons resultados, com precisão de aproximadamente 63%. Isso mostra que a previsão da qualidade do vinho não é uma tarefa fácil, e talvez seja devido à distribuição desequilibrada dos dados, como discuto na seção “Manipulação de dados”.

Minha proposta é combinar algumas qualidade de vinho em uma única categoria para equilibrar a distribuição dos dados. Além disso, foi realizado um processo de engenharia e seleção de características, que eliminou a multicolinearidade entre as características, e para a parte de treinamento, foi aplicado o procedimento de validação cruzada, que proporcionou uma boa generalização do melhor algoritmo, sugerida por seu desempenho no conjunto de dados de teste. No final, o conhecimento de negócios foi aplicado para melhorar ainda mais as previsões. 

Clique em “<i>Parâmetros do vinho</i>” para obter mais informações sobre as características do banco de dados.

</details>
