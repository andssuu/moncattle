# moncattle
Repositório dedicado para o desenvolvimento dos experimentos relacionados à dissertação de mestrado em Informática Aplicada da UFRPE.

## Dados dos experimentos

* 2 períodos: primeiro de 25/03/15 a 30/03/15 e de 06/04/15 a 09/04/15
* 4 animais (3 distintos)
* Foram utilizadas 4 coleiras (A, B, C e D). Entretanto, houve algumas falhas durante o experimento e, no final, ficaram as seguintes bases:
  - A2 e A3
  - B2 e B3
  - C3 e C4
  - D1, D2, D3 e D4
* Sensores: acelerômetro, giroscópio, magnetômetro e GPS
* As coletas eram feitas a cada 1 segundo para todos os sensores
* Carregamento offline dos dados por um cartão SD
* Classes de labels: pastar, em pé, deitado e andando
    * **Pastando/Procurando**: caracterizado pelo animal sobre as quatros patas, com a cabeça baixa procurando ou mastigando o capim. O animal pode ou não estar em movimento, já que ele pode estar se deslocando à procura de capim;
    * **Andando**: o animal também está sobre as quatros patas, porém com o pescoço reto (apontando o fucinho para frente) e se deslocando pela área de pasto;
    * **Em Pé**: o animal está sobre as quatro patas, com a cabeça erguida e não há deslocamento;
    * **Deitado**: o animal está com as patas abaixadas e com a barriga tocando o solo.

### Base de Dados

O arquivo dataset.csv contém os dados de todas as coleiras. No total, há 13088 amostras. A tabela a seguir mostra a distribuição das amostras por coleira

A2 | A3 | B2 | B3 | C3 | C4 | D1 | D2 | D3 | D4 | Total
--- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- | 
1112 | 2033 | 1131 | 1735 | 1852 | 406 | 1126 | 1690 | 1598 | 405 | 13088

## Pré-Processamento

## Aprendizagem Supervisionada

## Links

[Dissertação Luiz Lomba](https://repositorio.ufms.br/jspui/bitstream/123456789/2627/1/LUIZ%20FERNANDO%20DELBONI%20LOMBA.pdf)
