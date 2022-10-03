import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

def redeNeural(nome, momentum, lr, epocas, hiddenSize, datasetNome, entradas):
    entradas = entradas.replace("[", "")
    entradas = entradas.replace("]", "")
    entradas = entradas.split(",")
    momentum = float(momentum)
    lr = float(lr)
    epocas = int(epocas)
    hiddenSize = int(hiddenSize)
    start = time.perf_counter()
    datasetNome = datasetNome+'.csv'
    columns_data = ['HomeTeam', 'AwayTeam', 'FTR']
    for entrada in entradas:
        columns_data.append(entrada)
    data = pd.read_csv('app/data/dataset/'+datasetNome, header=0, sep=',', usecols=columns_data)
    data = data.replace(np.nan, 0)
    training = data.iloc[:-10]
    test = data.iloc[-10:]

    training = training.sample(frac=1)
    test = test.sample(frac=1)

    nomes = data[['HomeTeam', 'AwayTeam']]
    # mediaStats(data)
    data1 = data.iloc[:, 3:]
    data2 = data[['FTR']]
    data_transformed = data2.replace({"H":1,"D":0,"A":-1})
    input = data1.replace({"H":1,"D":0,"A":-1})
    output = data_transformed[['FTR']]
    nomes_training = nomes[:-10]
    nomes_test = nomes[-10:]
    #Normalização de dados
    for e in range(len(input.columns)): 
        max = input.iloc[:, e].max() #checar o valor maximo de cada coluna
        if max < 10:
            input.iloc[:, e] /= 10
            
        elif max < 100:
            input.iloc[:, e] /= 100
        else:
            print("Error in normalization! Please check!")

    training_input = input[:-10]
    training_output= output[:-10]
    test_input = input[-10:]
    test_output = output[-10:]

    # Classe do modelo da Rede Neural
    class Net(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Net, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.Tanh = torch.nn.Tanh() 
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.Tanh(output)
            return output
    # Convertendo para tensor
    training_input = torch.FloatTensor(training_input.values)
    training_output = torch.FloatTensor(training_output.values)
    test_input = torch.FloatTensor(test_input.values)
    test_output = torch.FloatTensor(test_output.values)

    # Criar a instância do modelo
    input_size = training_input.size()[1] 
    hidden_size = hiddenSize
    model = Net(input_size, hidden_size) 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum)

    # Treinamento
    model.train()
    epochs = epocas
    errors = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Fazer o forward
        y_pred = model(training_input)
        # Cálculo do erro
        loss = criterion(y_pred.squeeze(), training_output.squeeze())
        errors.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Época: {epoch} Loss: {loss.item()}')
        # Backpropagation
        loss.backward()
        optimizer.step()
    # Testar o modelo já treinado
    end = time.perf_counter()
    tempo_total = end-start
    model.eval()
    y_pred = model(test_input)
    erro_pos_treinamento = criterion(y_pred.squeeze(), test_output.squeeze())
    predicted = y_pred.detach().numpy()
    real = test_output.numpy()
    plotcharts(test_output, y_pred, errors, nomes_test)
    torch.save(model.state_dict(), "app/data/modeloTreinado.pth")
    erro_pos_treinamento = erro_pos_treinamento.item()/len(test_output)
    return tempo_total, erro_pos_treinamento, predicted, real

def plotcharts(test_output, y_pred, errors, nomes_test):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(test_output.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    xx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for x, home, away in zip(xx, nomes_test['HomeTeam'], nomes_test['AwayTeam']):
      plt.text(x, 1.3, home, rotation='vertical')
      plt.text(x, -1.3, away, rotation='vertical', verticalalignment='top')
    plt.legend(loc=0)
    plt.savefig('app/static/temp/graficoTrain.png', bbox_inches='tight')

# Tratamento String nome arquivo modelo
def tratarString(nome):
    nome = nome.lower()
    nome = nome.replace(' ', '-')

def testarRedeNeural(jogos):
    return False