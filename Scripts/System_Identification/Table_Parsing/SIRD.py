import camelot
import tabula
import re
import pandas as pd
if __name__ == '__main__':
    file = r"C:\Users\Jonas\Downloads\SIRD_Modeling.pdf"

    table = tabula.read_pdf(file, pages="all")
    SIRD = table[0]
    SIRD = SIRD.drop([2, 4]).drop(columns=['Unnamed: 0', 'Unnamed: 1', 'Other sources'])
    p = re.compile(r'\d+\.\d+')
    Vals = [p.findall(a) for a in SIRD['Value']]
    Vals = [[float(a) for a in b] for b in Vals]
    Vals = pd.Series(Vals)
    SIRD['Values'] = Vals
    SIRD['N'] = 420294250
    pd.to_pickle(SIRD, r'../data/SIRD_Brazil.pck')

    # df = pd.read_csv(r'C:\Users\Jonas\Downloads\2020-03_Modeling_and_forecasting_Covid-19_Brazil\2020-03_Modeling_and_forecasting_Covid-19_Brazil\data\brazil_states.csv', delimiter=';')