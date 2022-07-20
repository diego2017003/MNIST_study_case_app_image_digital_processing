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

Para o blur da imagem, utilizamos o blur gaussiano do opencv, com a função GaussianBlur. Dentro do código, recebemos a imagem e deixamos que o usuário regule o tamanho do kernel do gaussiano.

```python
def blur_image(image, mask_size=21):
    """receive the initial image and apply the gaussian blur according to the
    mask_size kernel passed as argument

    Args:
        image (_type_): matriz in gray_scale containing the trget image
        mask_size (_type_): size of the gaussian kernel the user wanna apply in the image

    Returns:
        _type_: blur image
    """
    if mask_size % 2 == 0:
        mask_size = mask_size + 1
    blur_img = cv.GaussianBlur(image, (mask_size, mask_size), 0)
    return blur_img
```

Padronizamos em inglês a documentação das funções para tornarmos o código mais uniforme. Para o canny foi padronizado que o threshold menor seria 50, para o threshold maior utilizamos um multiplicador inicial de 2 e permitimos que o usuário regule o multiplicador e o grau de abertura para o canny. A função recebe a imagem e esse outros parametros e devolve as bordas dos objetos na imagem.

```python
def edges_canny(image, lower_threshold=50, multiplier=2, aperture_size=3):
    """apply the canny algorithm to evidence the image edges and draw
    it's contour

    Args:
        image (_type_): numpy matrix containing the target image
        lower_threshold (_type_): lower threshold to canny algorithm
        multiplier (_type_): multiplier to the upper threshold
        aperture_size (_type_): aperture size must be 3,5 ou 7

    Returns:
        canny_img: img with the canny algorithm applyed with the parameters passed
        as argument
    """
    t_lower = lower_threshold
    t_upper = multiplier * lower_threshold
    if aperture_size in [3, 5, 7]:
        ap_size = aperture_size
    else:
        ap_size = 3

    canny_img = cv.Canny(image, t_lower, t_upper, apertureSize=ap_size)
    return canny_img
```

A partir da saída da função do canny nós começamos a dilatar as bordas encontradas para que o contorno da figura comece a se desenhar. A dilatação tem dois parâmetros, um deles diz respeito ao tamanho do kernel para dilatação, e o outro parâmetro diz quantas vezes a dilatação será aplicada na figura.

```python
def dilate_image(image, kernel_size=5, iterations=1):
    """dilate the image according to the kernel size, this function is mainly used to
    dilate the canny's edges found on the edges_canny function to evidence it better to
    contours algorithm and bounding boxes.

    Args:
        image (_type_): numpy matrix with the image the user gonna dilate
        kernel_size (int, optional): kernel's size of the boxe dilate kernel that the
        function gonna apply on the image. Defaults to 5
        iterations(int): iterations of the dilate algorithm on the image.Defaults to 1.

    Returns:
        _type_: dilated image according to the kernel_size
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv.dilate(image, kernel, iterations=iterations)
    return dilated_img

```

depois de dilatar as bordas, utilizamos a função findcontours nas bordas dilatadas. Com o resultado dos contornos encontramos as bounding boxes, e desenhamos as boxes em uma cópia da imagem original para o usuário interagir com as delimitações das figuras. 

```python
def find_contours_and_boxes(image):
    """find the contours of the image. This functions evidence the contours of the image,
    and intends to find the contours of the dilated image resulted from the dilate_image's
    function. We evidence the contours to find the ROI(Region of interest) of the target
    objects in the image and returns the bounding-boxes and the countors in two arrays.

    Args:
        image (_type_): target image in grayscale

    Returns:
        _type_: contours and the bounding boxes of the ROI's
    """
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    return contours_poly, boundRect, contours


def draw_bounding_boxes_contours(gray_image, contours_poly, boundRect, contours):
    """use the find_contours_and_boxes's function result to draw the countours
    into the target image. it intends to show how are the bounding boxes and the
    contours

    Args:
        gray_image (_type_): _description_
        contours_poly (_type_): _description_
        boundRect (_type_): _description_
        contours (_type_): _description_

    Returns:
        _type_: _description_
    """
    gray_image2 = gray_image.copy()
    for i in range(len(contours)):
        color = (
            0,
            0,
            0,
        )
        cv.drawContours(gray_image2, contours_poly, i, color)
        cv.rectangle(
            gray_image2,
            (int(boundRect[i][0]), int(boundRect[i][1])),
            (
                int(boundRect[i][0] + boundRect[i][2]),
                int(boundRect[i][1] + boundRect[i][3]),
            ),
            color,
            2,
        )
    return gray_image2

```
Depois de encontrarmos as boxes para todas as regiões de interesse da figura principal, nós usamos as coordenadas das boxes para extrair as regiões de interesse para posteriormente podermos fazer a predição das figuras. As imagens extraídas são retornadas em um array. Mas antes de inserirmos as figuras em um array, as imagens passam por um threshold que binariza as figuras em 0 e 255 assim como no dataset principal. 

```python
def extract_roi_black_white(image, boundRect, threshold=160):
    """extract roi with black and white

    Args:
        image (_type_): numpy array with the original image
        boundRect (_type_): bound rects array with bounding boxes
        threshold (_type_): lower threshold value

    Returns:
        list : array with roi's in black and white
    """
    letters_bw = []
    letters = extract_roi_by_boxes(image, boundRect)
    for letter in letters:
        (thresh, blackAndWhiteLetter) = cv.threshold(
            letter, threshold, 255, cv.THRESH_BINARY
        )
        letters_bw.append(255 - blackAndWhiteLetter)
    return letters_bw
```

Por último, utilizamos a função resize para deixarmos as figuras no formato 28X28 assim como no dataset original.
```python
def resize_rois(letters_bw):
    """resize the images for a 28X28 image as the mnist source format

    Args:
        letters_bw (_type_): roi's array

    Returns:
        list: list with resized images
    """
    resized_letters = []
    for letter_bw in letters_bw:
        resized_letter = cv.resize(letter_bw, (28, 28), interpolation=cv.INTER_AREA)
        resized_letters.append(resized_letter)
    return resized_letters
```

Ao final de todo o processamento, ainda tem uma última etapa antes do modelo na qual normalizamos os tons de cinza em valores entre 0 e 1 para facilitar a predição pelo modelo. 
O próximo setor deste trabalho será um overview sobre o app desenvolvido.

# 3. Overview do app
---

O app foi desenvolvido utilizando o streamlit em python. Uma ferramenta open-source para criação de telas com poucas linhas de código. Na ferramenta abrimos um modelo de machine learning desenvolvido com o tensorflow. A tela consiste em uma imagem com vários digitos escritos à mão(atualmente fixa), dois botões, sendo um para extrair as figuras ou trechos delimitados pelas bounding boxes e outro botão para submeter a imagem para a predição. Além disso, temos uma barra lateral contendo 5 sliders para que o usuário possa interagir com a figura selecionando o tamanho do kernel do gaussiano, o multiplicador dos threshold's do canny e o grau de abertura do canny, por último selcionar o kernel de dilatação e quantas vezes o algoritmo irá dilatar.
Para cada modificação dos parâmetros a função de bounding boxes tenta enquadrar as caixas nos objetos da figura, mas é importante salientar que os slider não modificam a imagem exibida, eles modificam uma cópia da imagem na qual todas as operações são feitas em conjunto, é fácil de ver isso quando modificamos os dois últimos slider que tratam da dilatação. É possível ver os contornos dilatando e contraindo com a mudança dos parâmetros.

![tela inicial](https://user-images.githubusercontent.com/30417399/180064018-784ce5a4-d4e9-49d4-a72c-9cd2480ce670.png)

O objetivo do estudo de caso é ir alterando os parâmetros nos sliders até enquadrar os digitos dentro das boxes.

![boxes_enquadradas](https://user-images.githubusercontent.com/30417399/180064351-6d1450d8-daea-475a-a1e3-21a013c35cc2.png)

Após enquadrar os digitos podemos apertar em extract rois para removermos as imagens selecionadas e exibirmos individualmente na tela.

![extracted](https://user-images.githubusercontent.com/30417399/180064751-624f8452-a822-4688-bb03-c9201742b53b.png)

Além de extrair as figuras podemos também manda-las para previsão apertando em predict.

![predict](https://user-images.githubusercontent.com/30417399/180064843-c5c3066b-8eb9-492f-96b4-c4115196064d.png)

Nativo do streamlit, o usuário ainda pode optar por alterar o theme da página durante a utilização do aplicativo. indo na extrema direita do site em settings.

![settings](https://user-images.githubusercontent.com/30417399/180065600-6c2a8be2-93da-4d21-aefd-ebedbaed5739.png)

Agora que já mostramos a aparência do app, vamos para a próxima etapa. Pois depois de todas as utilizações apresentadas você deve estar pensando "Estou interessado, como posso utilizar o app na minha máquina local??". E essa é a pergunta que iremos responder agora !!!

# 4. intruções de instalação e uso
---

O aplicativo foi desenvolvido usando uma ferramenta de organização e padronização de projeto python chamado poetry. Nesse repositório você vai encontrar um arquivo de extensão .toml com o nome pyproject.toml. Nesse arquivo existem os detalhes dos projeto desde as bibliotecas que possuem compatibilidade até a versão do código python e o author do projeto.

Para fazer o preparo do seu ambiente python e executar o projeto você pode simplesmente baixar o poetry python e executar o comando:
```poetry install```
no mesmo repositório do arquivo pyproject.toml, Assim o poetry irá criar o enviroment com a versão correta do python e instalar todas as bibliotecas necessárias para execução do projeto.
Entretanto, não é a única forma de se executar esse código. Caso não seja do interesse instalar o poetry você pode serguir os seugintes passos:
* intalar o virtualenv e o pyenv
* ```pyenv install 3.10.2```
* Executar o comando ```pyenv global 3.10.2``` para selecionar a versão do python
* cria o enviroment com virtual env
* ```python -m venv .venv```
* ```source .venv/bin/activate```
*  atualizar o pip ```python -m pip install --upgrade pip```
*  instalar as dependências ```python -m pip install -r requirements.txt```

Assim depois de preparado o ambiente a execução do código é feita com o comando ```streamlit run streamlit_app.py``` e se divertir utilizando o app à vontade
