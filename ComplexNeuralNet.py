import numpy as np
import time

def Sigmoid(x, flag):
    s = 1/(1+np.exp(-x))
    if (flag==1):
        return s*(1-s) # Retorna o valor da derivada no ponto x
    else:
        return s # Retorna o valor da funcao no ponto x


## Implementa uma rede neuronal 
#  com 4 entradas, uma camada oculta e 4 saidas
#
# Para o treinamento:
#      L0                     L1                   L2
# +------------+  Syn0  +-------------+  Syn1  +---------+ 
# | 4 entradas |------->| 32 neuronios|------->|4 Saidas |
# |   16x4     |  4x32  |    16x32    |  32x4  |  16x4   |
# +------------+        +-------------+        +---------+
# 16 registros de treino
# com 4 entradas cada
#
# Entradas de treinamento e ...
X = np.array([[-1,-1,-1,-1],
     [-1,-1,-1, 1],
     [-1,-1, 1,-1],
     [-1,-1, 1, 1],
     [-1, 1,-1,-1],
     [-1, 1,-1, 1],
     [-1, 1, 1,-1],
     [-1, 1, 1, 1],
     [ 1,-1,-1,-1],
     [ 1,-1,-1, 1],
     [ 1,-1, 1,-1],
     [ 1,-1, 1, 1],
     [ 1, 1,-1,-1],
     [ 1, 1,-1, 1],
     [ 1, 1, 1,-1],
     [ 1, 1, 1, 1]])
# ... suas respectivas saidas:
y = np.array([[1,0,0,0],
     [1,0,0,0],
     [1,0,0,0],
     [1,0,0,0],
     [0,1,0,0],
     [0,1,0,0],
     [0,1,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,1,0],
     [0,0,1,0],
     [0,0,1,0],
     [0,0,0,1],
     [0,0,0,1],
     [0,0,0,1],
     [0,0,0,1]])

alpha = 0.15 # Tamanho do passo
HLS = 32 # Tamanho da camanda oculta
Precisao = 1E-4
# Tamanhos: 
L,C = np.shape(X) # L linhas (registros) X C colunas (entradas)
s,S = np.shape(y) # Qtde de saidas: S
#
# Inicializa os pesos com numeros aleatorios entre -1 e 1
# As sinapses 0 conectam a camada 0 (entrada) com a camada 1 (c. oculta)
Syn0 = 2*np.random.random((C,HLS))-1
# As sinapses 1 conectam a camada 1 com a camada 2 (saida)
Syn1 = 2*np.random.random((HLS,C))-1
#
# Inicializa as camadas da rede e suas variaveis auxiliares:
L0 = X             
L1 = np.zeros((L,HLS))  
L2 = np.zeros((L,S))
L1_delta = np.zeros((L,HLS))
L2_delta = np.zeros((L,S))
#
tic = time.time()
for I in range(300000): # treinamento em varias iteracoes
    ## Calcula a camada 1:
    Prod = L0.dot(Syn0) # Poduto matricial L0 X Syn0
    #Passa o resultado na Sigmoid para obter a camada 1:
    #for J in range(L):
    #    for K in range(HLS):
    #        L1[J,K] = Sigmoid(Prod[J,K],0)
    L1=Sigmoid(Prod,0)

    ## Calcula a camada 2:
    Prod = L1.dot(Syn1) # Poduto matricial L1 X Syn1
    #Passa o resultado na Sigmoid para obter a camada 2:
    #for J in range(L):
    #    for K in range(S):
    #        L2[J,K] = Sigmoid(Prod[J,K],0)
    L2=Sigmoid(Prod,0)

    # A partir daqui - backpropagation:
    ## Calcula os novos pesos para as sinapses 1:
    L2_loss = y - L2 # Funcao perda para a camada 2
    MSE = sum(sum(L2_loss*L2_loss)/L) # Mean Squared Error
    # Se o MSE ja esta aceitavel, encerra as iteracoes:
    if MSE<Precisao:
        break

    # A cada 1000 iteracoes, mostra o MSE:
    if (I%1000)==0:
        print(MSE)
    #
    #for J in range(L):
    #    for K in range(S):
    #        # Multiplica as perdas pelos gradientes da sigmoid:
    #        L2_delta[J,K] = L2_loss[J,K] * Sigmoid(L2[J,K],1)
    L2_delta=L2_loss*Sigmoid(L2,1)

    # Atualiza os pesos das sinapses 1:
    Syn1 = Syn1 + alpha*(np.transpose(L1).dot(L2_delta)) 
    
    ## Calcula os novos pesos para as sinapses 0:
    L1_loss = L2_delta.dot(np.transpose(Syn1)) # Funcao perda para a camada 1
    #for J in range(L):
    #    for K in range(HLS):
    #        # Multiplica as perdas pelos gradientes da sigmoid:
    #        L1_delta[J,K] = L1_loss[J,K] * Sigmoid(L1[J,K],1)
    L1_delta=L1_loss*Sigmoid(L1,1)

    # Atualiza os pesos das sinapses 0:
    Syn0 = Syn0 + alpha*np.transpose(L0).dot(L1_delta)

###############################################################
# Testa a rede:
L0 =  np.array([1, -1, 1, -1]) # Entrada
Prod = L0.dot(Syn0) # Produto matricial
#L1 = np.zeros((S,HLS))  
#Passa o resultado na Sigmoid para obter a camada 1:
#for K in range(HLS):
#    L1[0,K] = Sigmoid(Prod[K],0)
L1=Sigmoid(Prod,0)

Prod = L1.dot(Syn1) 
#L2 = np.zeros((1,S))
#Passa o resultado na Sigmoid para obter a saida:
#for K in range(S):
#    L2[0,K] = Sigmoid(Prod[0,K],0)
L2=Sigmoid(Prod,0)

print(np.round(L2))
toc = time.time()
print("Elapsed time is " + str(toc-tic)+ " seconds.")

