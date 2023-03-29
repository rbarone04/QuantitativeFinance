# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:44:02 2023

@author: Riccardo
"""

import numpy as np
import pandas as pd
import random
import datetime as dt
import math

#import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

po.init_notebook_mode(connected=True)

class ModuleI():
    def __init__(self, frequency='weekly'):
        
        self.freq = frequency
        # Carico tutti i file contenenti i dati storici
        yield_90_21 = pd.read_csv('data/yield-curve-rates-1990-2021.csv',sep=',')
        yield_22 = pd.read_csv('data/yield-curve-rates-2022.csv',sep=',')
        yield_23 = pd.read_csv('data/yield-curve-rates-2023.csv',sep=',')
        
        # Carico un set di date pre-impostate: tutti i fine mese dal 31/1/1990 al 31/12/2019, poi tutti i giorni sino al 8/3/2023
        date = pd.read_csv('data/date_daily.csv',sep=',')
        date = date.astype({'Date':str})
        self.date = date['Date'].tolist()
        
        # Carico un set di date pre-impostate: tutti i fine mese dal 31/1/1990 al 31/12/2019, poi tutte le settimane sino al 7/3/2023
        date_w = pd.read_csv('data/date_weekly.csv',sep=',')
        date_w = date_w.astype({'Date':str})
        self.date_w = date_w['Date'].tolist()
        
        # Creao un unico dataframe con tutti i valori storici dal 1990 al 2023
        complete_yield = pd.concat([yield_90_21,yield_22,yield_23], ignore_index=True)
        
        # Elimino i dati relativi a 2M e 4M
        complete_yield = complete_yield.drop('0.17',axis=1)
        complete_yield = complete_yield.drop('0.33',axis=1)
        
        self.freq = 'weekly'
        
        # Sistemo il formato delle date per poterle elaborare correttamente
        complete_yield['Date'] = pd.to_datetime(complete_yield['Date'])
        complete_yield['Date'] = complete_yield['Date'].apply(lambda x: x.strftime('%Y%m%d'))
                
        # Tengo solo i dati relativi al set di date scelto (giornaliere o settimanali)
        partial_yield = None
        
        if self.freq == 'daily':
            partial_yield = complete_yield[complete_yield['Date'].isin(self.date)]
        elif self.freq == 'weekly':
            partial_yield = complete_yield[complete_yield['Date'].isin(self.date_w)]
    
        # Faccio il melt del dataframe e riordino i dati
        self.melted_yield = partial_yield.melt(id_vars='Date',var_name='Maturity',value_name='Rate')
        self.melted_yield = self.melted_yield.astype({'Maturity':float})
        self.melted_yield = self.melted_yield.sort_values(['Date','Maturity'],ascending=True)
        self.melted_yield = self.melted_yield.reset_index(drop=True)
        
        # Definisco le label da mostrare nei grafici
        self.label = ['1M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y']
        
        # Manipolo i dati per una visualizzazione più ordinata
        self.melted_yield['Date'] = self.melted_yield['Date'].apply(self.date_conversion)
        self.melted_yield['Rate'] = self.melted_yield['Rate'].apply(lambda x : x /100)
        
        self.sp500 = pd.read_csv('data/S&P_historical.csv',sep=';')
        self.sp500['Date'] = pd.to_datetime(self.sp500['Date'], format='%d/%m/%Y')
        
        
    def date_conversion(self,data):
        d = dt.datetime.strptime(data,'%Y%m%d')
        return d.strftime('%d/%m/%Y')
    
    def why_yield(self,height=800, width=1600):
        btp = pd.read_csv('data/btp_italia2.csv', sep=';')
        btp['Maturity'] = pd.to_datetime(btp['Maturity'], format='%d/%m/%Y')
        
        # Definisco il range per le maturity
        min_maty = min(btp['Maturity'])
        min_xRange = dt.datetime(min_maty.year-1, min_maty.month, min_maty.day)
        max_maty = max(btp['Maturity'])
        max_xRange = dt.datetime(max_maty.year+1, max_maty.month, max_maty.day)
        btp_price = go.Scatter(x=btp['Maturity'],y=btp['Price'],
                               hovertemplate='<br><b>Price</b>: %{y:.2f}<br>'+
                               '<b>%{text}</b><extra></extra>',
                               text=btp['Description'],
                               name='Bonds price', mode='markers', marker=dict(color='Red', size=5),
                               line=dict(color='Red', width=1, dash='dot'),
                               visible=True)
        
        btp_yield = go.Scatter(x=btp['Maturity'],y=btp['Yield'],
                               hovertemplate='<br><b>Yield</b>: %{y:.4p}<br>'+
                               '<b>%{text}</b><extra></extra>',
                               text=btp['Description'],
                               name='Bonds YtM', mode='markers', marker=dict(color='Green', size=5),
                               line=dict(color='Green', width=1, dash='dot'),
                               visible=False)
        
        
        f = go.FigureWidget()
        f.add_trace(btp_yield)
        f.add_trace(btp_price)
        f['layout']['xaxis'] = {'range':[min_xRange,max_xRange],
                                     'title': 'Maturity', 'titlefont':{'size': 15}}
        f['layout']['yaxis'] = {'range':[90,160],'fixedrange': False,
                                'title': 'Price','titlefont':{'size': 15},
                                'hoverformat': '.2f'}
        f['layout']['title'] = {'text':'Why yields (instead of prices)?','font':{'size': 25},'x':0.5,'xanchor':'center'}
        f['layout']['updatemenus']= list([
            dict(type="buttons",
                 active=-1,
                 buttons=list([
                    dict(label = 'Prices',
                         method = 'update',
                         args = [{'visible':[False,True]},
                                 {'yaxis':{'range':[90,160],'fixedrange': False,
                                           'title': 'Price','titlefont':{'size': 15},
                                           'hoverformat': '.2f'}}
                                ]
                        ),
                    dict(label = 'Yields',
                         method = 'update',
                         args = [{'visible':[True,False]},
                                 {'yaxis':{'range':[-0.005,0.0225],'fixedrange': False,
                                           'title': 'Yield-to-Maturity','tickformat': '.2p',
                                           'titlefont':{'size': 15},'hoverformat': '.4p'}}
                                ]
                        )
                 ]),
                direction='right',
                pad = {'r': 10, 't': 87},
                showactive = True,
                x = 0.5,
                xanchor = 'center',
                y = 0.1,
                yanchor = 'top'
            ),
            dict(type="buttons",
                 active=-1,
                 buttons=list([dict(label = 'Marker',method = 'update',args = [{'mode':['markers','markers']}]),
                               dict(label = 'Marker+Lines',method = 'update',args = [{'mode':['markers+lines','markers+lines']}]),
                               dict(label = 'Line',method = 'update',args = [{'mode':['lines','lines']}])
                              ]),
                direction='right',
                pad = {'r': 10, 't': 87},
                showactive = True,
                x = 0.5,
                xanchor = 'center',
                y = 0.0,
                yanchor = 'top'
            )
        ])
        f.update_layout(showlegend=False, template ='plotly_dark',height=height, width=width)
        
        return f
    
    def get_fancy_animation(self,height=800, width=1600, duration = 100):
        # Imposto una palette di colori 
        colors = px.colors.qualitative.Alphabet
        
        dates = list(set(self.melted_yield['Date'].tolist()))
        
        # Creo la scala di colori da applicare ad ogni yield
        rgb = px.colors.convert_colors_to_same_type(colors)[0]
        colorscale = []
        n_steps = math.ceil(len(dates)/(len(rgb)-1)) 
        for i in range(len(rgb)-1):
            for step in np.linspace(0, 1, n_steps):
                colorscale.append(px.colors.find_intermediate_color(rgb[i], rgb[i + 1], step, colortype='rgb'))
                
        # Seleziono il tasso massimo e quello minimo
        rates = self.melted_yield['Rate'].tolist()
        rates = [i for i in rates if not math.isnan(i)]
        rateMax = np.max(rates)
        rateMin = np.min(rates)
        
        # Creo la figura animata
        fig = px.line(self.melted_yield,x='Maturity',y='Rate',color='Date', animation_frame='Date',
                      color_discrete_sequence=colorscale, template="plotly_dark")
        
        # Aggiorno l'asse x in modo da mostrare le label delle maturity
        fig.update_xaxes(ticktext=self.label,tickvals=self.melted_yield['Maturity'].tolist(),tickangle=0,showticklabels=True)
        
        # Aggiorno l'asse y in modo da manterene il range fisso e mostrare i dati in formato percentuale
        fig.update_yaxes(range=[rateMin,rateMax],tickformat='.1%')
        
        # Sistemo le impostazioni dell'animazione
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = duration
        
        # Imposto il layout del titolo ed elimino la visualizzazione della legenda
        fig.update_layout(showlegend=False, title = {'text': 'U.S. Treasury Yield','font':{'size': 25},'x':0.5,'xanchor':'center'},
                          height=height, width=width)
        
        fig.layout.sliders[0].currentvalue = {'font':{'size':20}, 'xanchor':'center', 'prefix' : 'Date: '}    
        return fig
    
    def get_complete_animation(self, height = 800, width=1600, duration = 100):
        single_date = '31/01/2000'
        testing_fig = make_subplots(rows = 2, cols = 1,subplot_titles=('U.S. Treasury Yield','S&P 500 Historical data'))
        tmp_df = self.melted_yield[self.melted_yield['Date']==single_date]
        
        tmp_sp500 = self.sp500[self.sp500['Date']==dt.datetime.strptime(single_date,'%d/%m/%Y')]
        trace1 = go.Scatter(x = tmp_df['Maturity'], y = tmp_df['Rate'],mode="lines")
        trace2 = go.Scatter(x=tmp_sp500['Date'],y=tmp_sp500['S&P'],mode="markers",marker_symbol='arrow-down',marker_size=15)
        trace3 = go.Scatter(x=self.sp500['Date'],y=self.sp500['S&P'],mode='lines')
        
        
        testing_fig.add_trace(trace1,row=1,col=1)
        testing_fig.add_trace(trace2,row=2,col=1)
        testing_fig.add_trace(trace3,row=2,col=1)
        
        # Seleziono il tasso massimo e quello minimo
        rates = self.melted_yield['Rate'].tolist()
        rates = [i for i in rates if not math.isnan(i)]
        rateMax = np.max(rates)
        rateMin = np.min(rates)
        
        testing_fig.update_xaxes(range=[0,30],ticktext=self.label,tickvals=tmp_df['Maturity'].tolist(),tickangle=0,showticklabels=True,row=1,col=1)
        testing_fig.update_yaxes(range=[rateMin,rateMax],tickformat='.1%',row=1,col=1)
        testing_fig.update_layout(showlegend=False, template ='plotly_dark',height=height, width=width)
        
        d = [dt.datetime.strptime(i,'%Y%m%d').strftime('%d/%m/%Y') for i in self.date_w]
        frames=[go.Frame(data=[go.Scatter(x=self.melted_yield[self.melted_yield['Date']==k]['Maturity'],
                                          y=self.melted_yield[self.melted_yield['Date']==k]['Rate'],
                                          mode="lines"),
                               go.Scatter(x=self.sp500[self.sp500['Date']==dt.datetime.strptime(k,'%d/%m/%Y')]['Date'],
                                          y=self.sp500[self.sp500['Date']==dt.datetime.strptime(k,'%d/%m/%Y')]['S&P'],
                                          mode="markers",
                                          marker_symbol='arrow-down',
                                          marker_size=15)]
                        ) for k in d]
        testing_fig.frames = frames
        testing_fig.layout.updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": duration, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "&#9654;",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "&#9724;",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.5,
                "xanchor": "center",
                "y": 0.58,
                "yanchor": "middle"
            }
        ]
        
        return testing_fig
    
    def exercise(self, height = 800, width=1600):
        dates_exercise = ['28/02/2007','30/06/2010','30/06/2017','29/11/2019','28/02/2020','31/03/2020','22/10/2021','29/03/2022','18/10/2022','31/01/2023']
        exercise_fig = make_subplots(rows = 2, cols = 1,subplot_titles=('U.S. Treasury Yield','S&P 500 Historical data'))
        
        trace = go.Scatter(x=self.sp500['Date'],y=self.sp500['S&P'],mode='lines')
        exercise_fig.add_trace(trace,row=2,col=1)
        
        date_ex = random.choice(dates_exercise)
        tmp_df = self.melted_yield[self.melted_yield['Date']==date_ex]
        tmp_sp500 = self.sp500[self.sp500['Date']==dt.datetime.strptime(date_ex,'%d/%m/%Y')]
        
        trace1 = go.Scatter(x = tmp_df['Maturity'], y = tmp_df['Rate'],mode="lines",visible=True)
        trace2 = go.Scatter(x=tmp_sp500['Date'],y=tmp_sp500['S&P'],mode="markers",marker_symbol='arrow-down',marker_size=15,visible=False)
        exercise_fig.add_trace(trace1,row=1,col=1)
        exercise_fig.add_trace(trace2,row=2,col=1)
        
        # Seleziono il tasso massimo e quello minimo
        rates = self.melted_yield['Rate'].tolist()
        rates = [i for i in rates if not math.isnan(i)]
        rateMax = np.max(rates)
        rateMin = np.min(rates)
        
        exercise_fig.update_xaxes(range=[0,30],ticktext=self.label,tickvals=tmp_df['Maturity'].tolist(),tickangle=0,showticklabels=True,row=1,col=1)
        exercise_fig.update_yaxes(range=[rateMin,rateMax],tickformat='.1%',row=1,col=1)
        exercise_fig.update_annotations(font_size=20)
        exercise_fig.update_layout(showlegend=False, template ='plotly_dark',height=height, width=width)
        # Add dropdown
        exercise_fig.update_layout(
            updatemenus=[
                dict(
                    type = "buttons",
                    font=dict(color='black'),
                    #direction = "left",
                    buttons=list([
                        dict(
                            args=[{'visible':[True,True,True]},
                                  {'annotations':[{'font': {'size': 20},
                                                   'showarrow': False,
                                                   'text': 'U.S. Treasury Yield',
                                                   'x': 0.5,
                                                   'xanchor': 'center',
                                                   'xref': 'paper',
                                                   'y': 1.0,
                                                   'yanchor': 'bottom',
                                                   'yref': 'paper'},
                                                  {'font': {'size': 20},
                                                   'showarrow': False,
                                                   'text': 'S&P 500 Historical data',
                                                   'x': 0.5,
                                                   'xanchor': 'center',
                                                   'xref': 'paper',
                                                   'y': 0.375,
                                                   'yanchor': 'bottom',
                                                   'yref': 'paper'},
                                                  {'font': {'size': 20},
                                                   'showarrow': False,
                                                   'text': date_ex,
                                                   'x': 0.5,
                                                   'xanchor': 'center',
                                                   'xref': 'paper',
                                                   'y': 0.525,
                                                   'yanchor': 'bottom',
                                                   'yref': 'paper'}
                                                 ]
                                  }
                                 ],
                            label="Show date",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=0.51,
                    yanchor="top"
                ),
            ]
        )
        
        
        return exercise_fig
