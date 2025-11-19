[15/11/2025] Primeio commit do projeto.

[16/11/2025] Início da análise exploratória.
[17/11/2025] 	A quantidade de dados para as diferentes classes
		difere bastante, então é possível juntar algumas
		classes de modo a aumentar a quantidade de dados.

		Há algumas features correlacionadas (coef > 0.3),
		então podemos tentar transformar tais variáveis 
		de modo a descorrelacioná-las, e diminuir o VIF.
		
		Após a transformação das variáveis, todos os VIFs
		ficaram próximos a um, i.e., a multicolinearidade
		foi suprimida. Porém, mesmo antes da transformação
		de variáveis, nenhuma feature apresentava grande
		correlação com o target, somente a porcentagem de
		densidade de álcool tem uma correlação maior, o 
		que pode dificultar o aprendizado dos algoritmo.
[18/11/2025] Término da análise exploratória.
	     Seleção das melhores features com o forward feature
	     selector.
	     Início do treinamento dos modelos (leitura das melhores
	     features em um arquivo .txt)
[19/11/2025] Continuação do treinamento dos modelos
		Inicialmente a ideia era tentar predizar três
		qualidades de vinho, porém, os algoritmos empregados
		não possuem tal capacidade, pelo menos, com o proce-
		dimento que realizei até o momento. Então, diminui 
		uma classe de qualidade para o vinho, deixando 
		somente duas. Assim, ao menos um algoritmo fica 
		com score acima de 70%, o que pode ser aceitável.
		Para tanto, desenvolvi o processo de treinamento
		dos modelos no arquivo "model_training.py", porém,
		tive que alterar o arquivo Exploratory_data_analysis.py
		também. Ainda vou decidir se deixarei assim, ou 
		tentarei algum outro procedimento.
