import pandas as pd
from app.iaModel import redeNeural
import os
from flask import Flask, jsonify, redirect, request, url_for
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["GET"])
def rodarServidor():
    return 'API rodando'

@app.route("/treinar", methods=["POST"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def criarModelos():
    nome, momentum, lr, epocas, hiddenSize, datasetNome, entradas = None, None, None, None, None, None, None
    request.data = request.get_json()
    if request.data:
        if 'name' in request.data:
            nome = request.data['name']
        if 'momentum' in request.data:
            momentum = request.data['momentum']
        if 'learningRate' in request.data:
            lr = request.data['learningRate']
        if 'epochs' in request.data:
            epocas = request.data['epochs']   
        if 'hiddenSize' in request.data:
            hiddenSize = request.data['hiddenSize']   
        if 'datasetName' in request.data:
            datasetNome = request.data['datasetName']   
        if 'netInput' in request.data:
            if (request.data['netInput'] != ''):
                entradas = request.data['netInput']   
            else:
                return "Erro! Lista de entradas vazia!"
        tempo, erros, previsoes, reais = redeNeural(nome, momentum, lr, epocas, hiddenSize, datasetNome, entradas)
        salvarEstatisticas(nome, datasetNome, lr, momentum, hiddenSize, epocas, reais, previsoes, entradas)
        return jsonify(nome, datasetNome, lr, momentum, hiddenSize, epocas, entradas) 

@app.route("/previsao", methods=["POST"])
def previsao():
    return False

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='temp/' + filename), code=301)

def salvarEstatisticas(nome, dataset, lr, momentum, hiddenSize, epocas, real, previsao, entradas):
    for i in range(0, len(previsao)):
        if previsao[i] > 0.50:
            previsao[i] = 1
        elif previsao[i] <= 0.50 and previsao[i] >= -0.50:
            previsao[i] = 0
        elif previsao[i] < -0.50:
            previsao[i] = -1
    acertos = 0
    for i in range (0, len(real)):
        if real[i] == previsao[i]:
            acertos += 1
    tabela = pd.read_excel("app/data/Estatisticas.xlsx")
    tamanho_tabela = int(len(tabela))
    tabela.loc[tamanho_tabela, "Nome"] = nome
    tabela.loc[tamanho_tabela, "Dataset"] = dataset
    tabela.loc[tamanho_tabela, "Learning Rate"] = lr
    tabela.loc[tamanho_tabela, "Momentum"] = momentum
    tabela.loc[tamanho_tabela, "Tamanho Camada Oculta"] = hiddenSize
    tabela.loc[tamanho_tabela, "Epocas"] = epocas
    tabela.loc[tamanho_tabela, "Acertos/Total"] = f'{acertos} de {len(real)}'
    tabela.loc[tamanho_tabela, "Entradas"] = str(entradas)
    tabela.to_excel("app/data/Estatisticas.xlsx", index=False)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))