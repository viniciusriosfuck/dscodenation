import pandas as pd


def main():
    df = pd.read_csv('desafio1.csv')
    df_out = (df.groupby('estado_residencia')['pontuacao_credito']
              .agg([lambda x: x.value_counts().index[0], 'median', 'mean', 'std']))
    df_out.columns = ['moda', 'mediana', 'media', 'desvio_padrao']
    df_out.sort_index(ascending=False, inplace=True)
    df_out.to_json('submission.json', orient='index')


if '__name__' == main():
    main()
