import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots

BANNER_IMG = 'banner.png'

# --------- LEITURA DO DATASET --------------------
df = pd.read_csv('wages_cleaned.csv')
print(f'Leitura do dataset realizado com sucesso! Temos {len(df)} linhas neste dataset.')


#--------------------------------------------------------
st.image(BANNER_IMG)
st.header("DataFrame dos salários dos jogadores de futebol")
st.table(
    df.head(5),
    alt="Inspeção das 5 primeiras linhas",
)


with st.sidebar:
    divisor = st.select_slider(
        "Selecione o valor de salario divisor:",
        options=[
            "500000",
            "1500000",
            "2500000",
            "12500000",
        ],
    )
    print(f"Foi selecionado: {divisor}")
    n_divisor = int(divisor)
    st.image("https://www.cesar.school/wp-content/themes/alfama/assets/img/logo_cesar.png")


df1 = df[df['Salary'] > n_divisor] # pop star
df2 = df[df['Salary'] <= n_divisor] # jogadores comuns
#-------------------------------------------------------------------------------------------------------------
st.header("Histograma dos Salários")

fig_subplot = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Jogadores Populares", "Pop Star")
)

fig_subplot.add_traces(
    px.histogram(df2, x='Salary').data,
    rows=[1] * len(px.histogram(df2, x='Salary').data),
    cols=[1] * len(px.histogram(df2, x='Salary').data)
)

fig_subplot.add_traces(
    px.histogram(df1, x='Salary').data,
    rows=[1] * len(px.histogram(df1, x='Salary').data),
    cols=[2] * len(px.histogram(df1, x='Salary').data)
)

# Rótulos dos eixos X e Y de cada subplot
fig_subplot.update_xaxes(title_text="Salário (R$)", row=1, col=1)
fig_subplot.update_yaxes(title_text="Frequência", row=1, col=1)

fig_subplot.update_xaxes(title_text="Salário (R$)", row=1, col=2)
fig_subplot.update_yaxes(title_text="Frequência", row=1, col=2)

fig_subplot.update_layout(
    height=600,
    width=1200,
    title_text="Histograma dos Salários dos Jogadores",
    showlegend=False
)

st.plotly_chart(fig_subplot)

#--------------------------------------------------------
st.header("Matriz de Correlação do dataset")

df = df.drop(['Last_Transfer_Fee'],axis=1)
M = df.corr()
mask = np.tril(np.ones_like(M, dtype=bool))
M_masked = M.mask(mask)
M_masked = np.round(M_masked, 3)
fig2 = px.imshow(M_masked , text_auto=True, color_continuous_scale='Blues')
st.plotly_chart(fig2)

#--------------------------------------------------------
st.header("Seção 4")

fig3= px.pie(
    df,
    names='Is_top_ranked_nation',
    title='Distribuição de Nações no Top Ranking',
    color_discrete_sequence=[
        '#05132A',
        '#10408D',
        '#2166AC',
        '#4393C3',
        '#92C5DE',
        '#D1E5F0'
    ]
)

fig3.update_traces(
    textinfo='label+value+percent',
    textposition='inside'
)

st.plotly_chart(fig3)
