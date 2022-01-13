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
* Houve uma divisão dos comportamentos em primários e secundários. A seguir a definição dos comportamnetos primários:
    * **Pastando/Procurando**: caracterizado pelo animal sobre as quatros patas, com a cabeça baixa procurando ou mastigando o capim. O animal pode ou não estar em movimento, já que ele pode estar se deslocando à procura de capim;
    * **Andando**: o animal também está sobre as quatros patas, porém com o pescoço reto (apontando o fucinho para frente) e se deslocando pela área de pasto;
    * **Em Pé**: o animal está sobre as quatro patas, com a cabeça erguida e não há deslocamento;
    * **Deitado**: o animal está com as patas abaixadas e com a barriga tocando o solo.
* Já os comportamentos secundários, consistem em uma ramificação dos comportamentos “Em Pé” e “Deitado” adicionando o comportamento ruminar, o qual o animal está com a mandíbula em movimento. Logo, surgem duas novas classes: “Ruminando Em Pé” e “Ruminando Deitado”.

## Base de Dados

O arquivo dataset.csv contém os dados de todas as coleiras. No total, há 13088 amostras. A seguir, a tabela mostra a distribuição das amostras por coleira

A2 | A3 | B2 | B3 | C3 | C4 | D1 | D2 | D3 | D4 | Total
--- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- | 
1112 | 2033 | 1131 | 1735 | 1852 | 406 | 1126 | 1690 | 1598 | 405 | 13088

## Notebooks Sensores

* [Acelerômetro](https://colab.research.google.com/drive/1Zx0bVCSNSRDoqhBQ6uySONDKvaZkiZw1?usp=sharing)

* [Giroscópio](https://colab.research.google.com/drive/1trXq9sLZd5u5y0RtJhvtcBTyMN64Y3l1?usp=sharing)

* [Magnetômetro](https://colab.research.google.com/drive/1DHpElWUB1YtBNloKtzsXeQKEDzahV15R?usp=sharing)

## Trabalhos Relacionados

### Leandro de Jesus

*   [Identificação do Comportamento Bovino por meio do Monitoramento Animal (Dissertação)](https://repositorio.ufms.br/bitstream/123456789/2075/1/Leandro%20de%20Jesus.pdf)

*   [UTILIZAÇÃO DE TÉCNICAS DE RECONHECIMENTO DE PADRÕES PARA ESTIMAR O COMPORTAMENTO DE BOVINOS EM FUNÇÃO DE VARIÁVEIS AMBIENTAIS (Tese Doutorado)](https://repositorio.pgsskroton.com/bitstream/123456789/22927/1/LEANDRO%20DE%20JESUS.pdf)

*   [IDENTIFICAÇÃO DO COMPORTAMENTO BOVINO POR MEIO DO MONITORAMENTO
ANIMAL (Resumo ao I Simpósio Brasileiro de Pecuária de Precisão Aplicada à Bovinocultura de Corte)](https://ainfo.cnptia.embrapa.br/digital/bitstream/item/119723/1/identificacao-do-comportamento-bovino-por-meio-do-monitoramento-animal.pdf)

### Luiz Fernando Delboni Lomba

*   [Identificacao do Comportamento Bovino a partir dos Dados de Movimentacao e do Posicionamento do Animal (Dissertação)](https://repositorio.ufms.br/bitstream/123456789/2627/1/LUIZ%20FERNANDO%20DELBONI%20LOMBA.pdf)

### Outros

*   [Sistema para monitoramento da movimentação bovina e aferição dos comportamentos (SBIAgro)](https://ainfo.cnptia.embrapa.br/digital/bitstream/item/169799/1/Sistema-para-monitoramento-da-movimentacao-bovina.pdf)

*   [PREDIÇÃO DO COMPORTAMENTO BOVINO COM SENSORES DE POSIÇÃO E MOVIMENTAÇÃO 1](http://reunioessbpc.org.br/campogrande/inscritos/resumos/4888_1693116b9f38336f4c0bb9860d3dd9ab0.pdf)

*   [PREDIÇÃO DO COMPORTAMENTO BOVINO COM SENSORES DE POSIÇÃO E MOVIMENTAÇÃO 2](https://www.brazilianjournals.com/index.php/BRJD/article/view/22203/17723)

*   [O uso de inteligência artificial na identificação do comportamento bovino](http://www.eventos.uepg.br/sbiagro/2015/anais/SBIAgro2015/pdf_resumos/16/16_luiz_fernando_delboni_lomba_85.pdf)

*   [Aplicação de técnicas de reconhecimento de padrões para estimar o comportamento debovinos em função de dados de posicionamento GPS - JESUS (2018)](https://www.geopantanal.cnptia.embrapa.br/Anais-Geopantanal/pdfs/p3.pdf)
