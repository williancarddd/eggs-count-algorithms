<h1 align="center">Welcome to Sistema de Contagem de Ovos Automático em Paleta de Aucatex 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://twitter.com/williancarddd" target="_blank">
    <img alt="Twitter: williancarddd" src="https://img.shields.io/twitter/follow/williancarddd.svg?style=social" />
  </a>
</p>

> Essa é a implementação do backend para treino e teste dos modelos de inteligência artificial para plataforma de contagem de ovos automática em paletas de aucatex, projeto em parceiria com a Fiocruz e Fiotec. O diferencial está na divisão de imagens de alta resolução para imagens de pequenas resolução com processamento local.

<p>
  <img alt="Version" src="readme.png" />
</p>

## Usage

```sh
As pastas drafts são diferentes versões de desenvolvimento do modelo de contagem. Actual são versões estáveis que realmente estão contando ovos. Bases são as imagens divididas em bases/lotes de 10 imagens.
```

## Instruções
- O projeto está dividido em 12 bases de imagens, nas quais as 3 primeiras bases são ignoradas, porque não são da etapa de scanner.

- Após receber uma nova leva de imagens, é rodado o script rename.py para que seja identificado o padrão delas e renomeadas para o usado no projeto.

- Após as imagens estarem renomeadas é rodado o script tests.ipynb que deve percorrer todas as pastas e executar os modelos para comparação,
o resultado dele é um arquivo txt e uma pasta "processed" com algumas imagens para verificar visualmente a contagem.

- O script generated-squares é utilizado para gerar quadrados para para determinada base de dados, ele pega todas imagens nessa base e para cada
uma gera vários quadrados de 254x254, foi utilizado para etapa de anotação manual do ovos.

- As pastas que possuem draft, em ordem, são as tentativas e algoritmos que passamos até chegar em draft-actual-5, o mais atual e funcional, no qual usa visão computacional com redes neurais, alguns modelos neles são best-v2-train1.pt , primeiro modelo treinado, best-v1.pt, segundo modelo treinado e best-train2.pt terceiro modelo treinado, esses treinamentos estão em eggs-scanner-image.v2i.yolov11.

- yolo-train contém as tentativas e resultados dos treinos de visão computacional eggs-scanner-image.v2i.yolov11 é o utilizado,
nas execuções, por ordem, temos os modelos utilizados para gerar o resultados esperados.

- resume_dataset.py exibe um resumo dos dados utilizados para treinamento

- rebalance balanceia os dados para que fique na perspectiva de 70, 20 e 10 para treino, teste e validação respectivamente, ele considera a quantidade de ovos contado para balancear  e não a quantidade de imagens.


> Foi utilizado a base de dados "base-5" para treinamento

## Author

👤 **William Cardoso Barbosa**

* Website: www.ecotechamazonia.com.br
* Twitter: [@williancarddd](https://twitter.com/williancarddd)
* Github: [@williancarddd](https://github.com/williancarddd)
* LinkedIn: [@https:\/\/www.linkedin.com\/in\/william-cardoso-9363a015a\/](https://linkedin.com/in/https:\/\/www.linkedin.com\/in\/william-cardoso-9363a015a\/)

## Show your support

Give a ⭐️ if this project helped you!

***
