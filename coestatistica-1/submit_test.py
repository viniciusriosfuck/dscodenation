import json

def score():
    with open('desafio1_respostas.json') as json_data:
        respostas = json.load(json_data)

    with open('submission.json') as json_data:
        submission = json.load(json_data)

    total = 0
    acertos = 0
    for estado in respostas.keys():
        respostas_estado = respostas[estado]
        respostas_sub = submission[estado]
        for metrica in respostas_estado.keys():
            total += 1
            if round(respostas_estado[metrica],2) == round(respostas_sub[metrica], 2):
                acertos += 1
    return json.dumps({"score":round(acertos/total*100, 2)})

if __name__ == "__main__":
    print(score())
