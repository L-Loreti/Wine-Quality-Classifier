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

