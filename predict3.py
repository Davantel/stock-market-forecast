import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import tkinter as tk
import customtkinter as tkc
from PIL import Image, ImageTk

def predict():
    symbol = symbol_entry.get()
    data = yf.download(symbol)

    # Filtra apenas as colunas 'Close'
    df = data[['Close']]

    # Separa os dados em treino e teste
    train = df[:-7]
    test = df[-7:]

    # Cria as variáveis X e y para o modelo de regressão linear
    X_train = np.array([d.toordinal() for d in train.index]).reshape(-1, 1)
    y_train = train['Close'].values

    # Treina o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faz a previsão para os próximos dias
    if var.get() ==0 and var2.get() ==0:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=5, freq='D')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 5 dias')
        valor_label.place(x=120, y=280)
    elif var.get() ==0 and var2.get() ==1:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=10, freq='D')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 10 dias')
        valor_label.place(x=120, y=280)
    elif var.get() ==0 and var2.get() ==2:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=15, freq='D')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 15 dias')
        valor_label.place(x=120, y=280)

    # Faz a previsão para os próximos meses
    elif var.get() ==1 and var2.get() ==0:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=5, freq='M')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 5 meses')
        valor_label.place(x=120, y=280)
    elif var.get() ==1 and var2.get() ==1:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=10, freq='M')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 10 meses')
        valor_label.place(x=120, y=280)
    elif var.get() ==1 and var2.get() ==2:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=15, freq='M')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 15 meses')
        valor_label.place(x=120, y=280)

    # Faz a previsão para os próximos anos
    elif var.get() ==2 and var2.get() ==0:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=5, freq='Y')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 5 anos')
        valor_label.place(x=120, y=280)
    elif var.get() ==2 and var2.get() ==1:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=10, freq='Y')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 10 anos')
        valor_label.place(x=120, y=280)
    else:
        last_date = df.index.max()
        X_test = np.array([d.toordinal() for d in pd.date_range(start=last_date, periods=15, freq='Y')]).reshape(-1, 1)
        y_pred = model.predict(X_test)
        valor_label.configure(text='Valor para os proximos 15 anos')
        valor_label.place(x=120, y=280)

    # Converte as datas de volta para o formato Timestamp
    dates = [datetime.fromordinal(d) for d in X_test.flatten()]

    # Cria um DataFrame com as previsões
    predictions = pd.DataFrame({
        'Data': dates,
        'Valor': y_pred
    })

    # Faz a previsão para os dados de teste
    X_test = np.array([d.toordinal() for d in test.index]).reshape(-1, 1)
    y_pred = model.predict(X_test)

    # Calcula o MSE e o RMSE
    mse = mean_squared_error(test['Close'], y_pred)
    rmse = np.sqrt(mse)

    # Exibe as previsões e os resultados
    linhas = predictions[predictions.columns[0]].count()
    if linhas == 5:
        predictions_label.configure(text=predictions.to_string(index=False), font=('Arial', 18))
    elif linhas == 10:
        predictions_label.configure(text=predictions.to_string(index=False), font=('Arial', 17))
    else:
        predictions_label.configure(text=predictions.to_string(index=False), font=('Arial', 13))

    mse_label.configure(text=f'MSE: {mse:.2f}', font=('Arial', 14))
    rmse_label.configure(text=f'RMSE: {rmse:.2f}', font=('Arial', 14))


tkc.set_appearance_mode("System")
tkc.set_default_color_theme("blue")
app = tkc.CTk()
app.geometry('500x800')
app.title('Cotação')

#logo
imagem = tkc.CTkImage(dark_image=Image.open('bolsa-de-dinheiro.png'), size=(55, 55))
logo = tkc.CTkLabel(app, text="", height=15, image=imagem)
logo.place(x=100,y=7)

#texto intro
intro = tkc.CTkLabel(app, text="Prever valor", font=('Arial', 28, 'bold'),  height=10)
intro.place(x=180,y=18)

linha = tkc.CTkLabel(app, text="", width=750, height=1, font=('Arial', 1), fg_color=('#bc9ce4'))
linha.place(x=0,y=70)

#frame de checkbutton
frame = tkc.CTkFrame(app)
frame.place(x=23, y=150)

#variavel dos checkbutton de periodo
var = tkc.IntVar()
var.set(0)

#frame de checkbutton
frame2 = tkc.CTkFrame(app)
frame2.place(x=23, y=180)

#variavel dos checkbutton de periodo
var2 = tkc.IntVar()
var2.set(0)

symbol_label = tkc.CTkLabel(app, text="Symbol", font=('Arial', 16, 'bold'))
symbol_label.place(x=22, y=100)

symbol_entry = tkc.CTkEntry(app, width=155, corner_radius= 15, border_color='#bc9ce4')
symbol_entry.place(x=100, y=100)

#checkbutton de periodo
check_dolar = tkc.CTkRadioButton(frame, text="Dias", font=("Arial", 13, 'bold'), variable=var, value=0, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=70, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_dolar.pack(side=tkc.LEFT)

check_moedas = tkc.CTkRadioButton(frame, text="Meses", font=("Arial", 13, 'bold'), variable=var, value=1, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=90, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_moedas.pack(side=tkc.LEFT)

check_acoes = tkc.CTkRadioButton(frame, text="Anos", font=("Arial", 13, 'bold'), variable=var, value=2, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=70, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_acoes.pack(side=tkc.LEFT)

#checkbutton de periodo
check_dolar = tkc.CTkRadioButton(frame2, text="5", font=("Arial", 13, 'bold'), variable=var2, value=0, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=70, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_dolar.pack(side=tkc.LEFT)

check_moedas = tkc.CTkRadioButton(frame2, text="10", font=("Arial", 13, 'bold'), variable=var2, value=1, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=90, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_moedas.pack(side=tkc.LEFT)

check_acoes = tkc.CTkRadioButton(frame2, text="15", font=("Arial", 13, 'bold'), variable=var2, value=2, border_width_checked=2, radiobutton_width=15, radiobutton_height=15, width=70, hover_color=('#bc9ce4'), fg_color=('#bc9ce4'))
check_acoes.pack(side=tkc.LEFT)

exemplos_label = tkc.CTkLabel(app, text="Exemplos de Simbolos", font=('Arial', 16, 'bold'))
exemplos_label.place(x=290, y=100)

exemplos_sim_label = tkc.CTkLabel(app, text="VALE\nPETR4.SA\nBBSE3.SA\nTAEE11.SA\nITSA4.SA\nBTC-USD\nETH-USD", font=('Arial', 13, 'bold'), pady=3, width=180, height=60, fg_color=('black', '#323232'), corner_radius=8)
exemplos_sim_label.place(x=290, y=136)

predict_button = tkc.CTkButton(app, text="PREVER", width=152, font=("Arial", 13, 'bold'), fg_color=('#bc9ce4'), corner_radius=15, command=predict)
predict_button.place(x=100,y=225)

linha = tkc.CTkLabel(app, text="", width=750, height=1, font=('Arial', 1), fg_color=('#bc9ce4'))
linha.place(x=0,y=265)

valor_label = tkc.CTkLabel(app, text="Valor previsto", font=("Arial", 17, 'bold'))
valor_label.place(x=190, y=280)

predictions_label = tkc.CTkLabel(app, text="", font=("Arial", 14), pady=15, width=350, height=290, fg_color=('black', '#323232'), corner_radius=8)
predictions_label.place(x=70, y=310)

mse_val_label = tkc.CTkLabel(app, text="Valor para o MSE", font=("Arial", 17, 'bold'))
mse_val_label.place(x=170, y=610)

mse_label = tkc.CTkLabel(app, text="", font=("Arial", 14), pady=15, width=350, height=30, fg_color=('black', '#323232'), corner_radius=8)
mse_label.place(x=70, y=640)

rmse_val_label = tkc.CTkLabel(app, text="Valor para o RMSE", font=("Arial", 17, 'bold'))
rmse_val_label.place(x=170, y=710)

rmse_label = tkc.CTkLabel(app, text="", font=("Arial", 14), pady=15, width=350, height=30, fg_color=('black', '#323232'), corner_radius=8)
rmse_label.place(x=70, y=740)

app.mainloop()
