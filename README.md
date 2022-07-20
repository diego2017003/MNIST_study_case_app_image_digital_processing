# MNIST_study_case_app_image_digital_processing
app with a case study of image digital processing to prepare handwritten digits to a machine learning model prediction
---

# 1. Introdução

----
A área de dados vem se desenvolvendo bastante nos últimos anos, e junto com ela, a existências de sistemas baseados em machine learning vem se tornando cada vez mais comuns para tomadas de decisões. A área de machine learning é um dos ramos da ciência da computação que vem sendo estudado desde o século passado, em contrapartida, a utilização dos modelos em software costumava ser bastante experimental e baseado em tentativa até poucos anos atrás. Para tornar softwares de machine mais adaptáveis ao mercado, vem se estudando a área de MLOPS, na qual os softwares de machine learning começam a incorporar práticas de desenvolvimento dos padrões de devOPS que já estão disseminados no atual cenário da programação. Mlops Tem por intuito, automatizar parte do processo de criação e incorporação dos modelos em aplicativos, utilizando os conceitos de entregas continuas e integração continua(CI/CD).
Dentro do processo de Mlops, podemos refletir sobre como os modelos funcionam em produção, e como os dados do mundo real devem ser tratados para que o sistema continue funcionando bem. Dentro deste contexto, esse trabalho propõe um estudo de caso para um modelo de machine learning funcionando em um aplicativo python. O aplicativo tem por intuito, permitir que o usuário interaja com as variáveis do processamento de imagem que fazem borramento, detecção de bordas e dilatação, para extrair digitos escritos a mão de uma imagem e manda-los para predição com um modelo treinado previamente.
Esse relatório será dividido em 4 sessões: Introdução(essa sessão), processamento de imagens(abordar como cada algoritmo interfere na imagem mostrando os resultados), como usar o aplicativo(Explorar o aplicativo como um guia de usuário), e por último, como instalar o aplicativo(um passo a passo para execução na máquina de quem tiver interesse em reproduzir este experimento).
Antes, de partir para os próximos tópicos, vamos nos situar em como o aplicativo se encaixa no contexto de machine learning em produção:
![prediction_image](https://user-images.githubusercontent.com/30417399/180027225-dcb0ecc9-e204-4147-84aa-bdf3e6c3eecd.png)
Depois que os dados iniciais já foram processados, o modelo ,treinado e validado, pronto para produção. O aplicativo vem como a interface do modelo e o usuário final para que o modelo entre definitivamente em utilização.
---

# 2. processamento das imagens

---
No estudo de redes neurais e aprendizado profundo(Redes neurais convolucionais, redes auto generativas ... etc), um dos conjuntos de dados mais uilizados para estudo dos modelos é o MNIST. O MNIST consiste em uma base de dados com digitos(0 a 9) escritos à mão, a base é bem balanceada e é composta por aproximadamente 10000 amostras de cada digito, onde cada digito é uma imagem em tom de cinza de tamanho 28 X 28.
### Amostra da base
![mnist_eda](https://user-images.githubusercontent.com/30417399/180029600-666fc18d-ef52-4fcc-bb4e-c3c997379bd6.png)

### balanceamento dos dados
![balance_dataset](https://user-images.githubusercontent.com/30417399/180029639-15bb592f-649d-4cbc-979a-070dc2887ce1.png)

O modelo utilizado é relativamente simples com uma camada de flatten que lineariza os dados em um vetor, uma camada escondida e uma camada de saída, e os dados de entrada do modelo são essas matrizes de dimensão 28 X 28, relacionadas a um rótulo que informa qual é o digito correspondente. Logo, as imagens que o usuário pode mandar prever deve seguir esse mesmo formato.

O processamento de novas imagens passa por algumas etapas, o blur para reduzir ruído com o borramento, a detecção de bordas com canny para delimitarmos as figuras, dilatamos as bordas para unir todas as bordas da mesma figura detectadas pelo algoritmo de canny. Para extrair as figuras aplicamos o algortimo findcontours do opencv e em cima dele o bounding boxes para desenhar caixas delimitando os digitos, por último usamos as caixas como endereços bases no vetor para saber de qual pixel até qual píxel nós temos determinados digito na imagem original. Depois de extrair as figuras aplicamos um threshold para deixar o fundo preto e o digito branco, e então redimensionamos as figuras extraídas.

![processamento_pipeline](https://user-images.githubusercontent.com/30417399/180041745-88ae0c7c-ec41-4a1e-9fed-1ba7ae02f88e.png)
