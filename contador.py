# importando o modulo matemático do python
import numpy as np
# importando o open cv
import cv2


# Função de centralização
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# chamando a função VideoCapture e pedidno para que ele mostre o vídeo
cap = cv2.VideoCapture('1.mp4')
# objeto para criação de mascara
fgbd = cv2.createBackgroundSubtractorMOG2()

# Criando a linha na vertical
posL = 150
# Quantidade de pixels que serão usados para contar
offset = 30
# Posição da linha - inicio
xy1 = (20, posL)
# Posição da linha - fim
xy2 = (300, posL)
# Cache para fazer a linha de tracking
detects = []
# Contagem do total
total = 0
# Contagem do total que desceu
down = 0
# Contagem do total que subiu
up = 0
# Loop para mostrar os frames
while 1:
     # boolean: enquanto existir imagem, e frame leia o cap read.
    ret, frame = cap.read()
    # Imagem em cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mascara na aplicação em cinza
    fgmask = fgbd.apply(gray)

    # Aplicando um threshhold, deixando somente os valores preto e branco
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    # kernel para poder usar o OPENING
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # criando opening para diminuir o noise do Threshold
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    # Deixando as formas do opening mais grosso
    dillatation = cv2.dilate(opening, kernel, iterations=8)
    # fecha os espaços em preto deixado pelo opening, deixando a forma mais uniforme
    closing = cv2.morphologyEx(
        dillatation, cv2.MORPH_CLOSE, kernel, iterations=8)
    # tirando o contorno da imagem
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # criando as linhas na imagem, a linha do meio serve para saber se a pessoa está indo para cima ou para baixo
    cv2.line(frame, xy1, xy2, (255, 0, 0), 3)
    # criando uma linha trinta px abaixo
    cv2.line(frame, (xy1[0], posL-offset),
             (xy2[0], posL-offset), (255, 255, 0), 2)
    # criando uma linha trinta px acima. isso é feito para que se crie um espaço aonde a contagem será feita caso alguem passe
    cv2.line(frame, (xy1[0], posL+offset),
             (xy2[0], posL+offset), (255, 255, 0), 2)
    # contagem da ID de quem está lá dentro
    i = 0

    # Criando e definindo os contornos
    for cnt in contours:
        # Puxando as medidas da imagem
        (x, y, w, h) = cv2.boundingRect(cnt)
        # calculando a área com as medidas da imagem
        area = cv2.contourArea(cnt)
        # Para garantir que ele não vá pegar nenhum ruido na contagem, garantindo que só pessoas serão capturadas.
        if int(area) > 3000:
            # puxando a função center
            centro = center(x, y, w, h)
            # Escrevendo no vídeo a contagem
            cv2.putText(frame, str(i), (x+5, y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # criando um circulo no centro do retangulo
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
            # dando as caracteristicas do retangulo que está sendo criado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # se o len for menor que o I ele cria um append dentro da lista
            if len(detects) <= i:
                detects.append([])
            if centro[1] > posL-offset and centro[1] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            # Fazendo a contagem de quantas pessoas passaram e foram calculadas
            i += 1
    # se não tem nenhum contorno (não há ninguém), limpe a lista
    if len(contours) == 0:
        detects.clear()
    else:
        # Criando a linha de tracking, ela será necessaria pra saber se a pessoa está subindo ou descendo
        for detect in detects:
            for (c, l) in enumerate(detect):
                if detect[c-1][1] < posL and l[1] > posL:
                    detect.clear()
                    up += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                    continue

                if detect[c-1][1] > posL and l[1] < posL:
                    detect.clear()
                    down += 1
                    total += 1
                    cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                    continue

                if c > 0:
                    cv2.line(frame, detect[c-1], l, (0, 0, 255), 1)
    # Printando os valores da lista aonde vai os valores da imagem
    print(detects)
    # Escrevendo o total no tela
    cv2.putText(frame, "TOTAL: " + str(total), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    # Escrevendo quantos estão subindo na tela
    cv2.putText(frame, "SUBINDO " + str(up), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Escrevendo quantos estão descendo na tela
    cv2.putText(frame, "DESCENDO: " + str(down), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Criando a tela frame, ou seja, a imagem original só com os desenhos declarados acima
    cv2.imshow('frame', frame)
    # Criando a tela closing, ou seja, com todas as alterações feitas acima.
    cv2.imshow("closing", closing)
    # Declarando em quantos fps ela vai rodar, se você aperta "q" os processos são fechados e as janelas se fecham.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
