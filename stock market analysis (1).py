#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np


# In[2]:


import yfinance as yf


# In[3]:


import plotly.express as px
import cufflinks
import plotly.io as pio 
import yfinance as yf
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
pio.renderers.default = "notebook" # should change by looking into pio.renderers

pd.options.display.max_columns = None



# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from pandas_datareader import data,wb






# In[ ]:





# In[6]:


from datetime import datetime


# In[7]:


from __future__ import division


# In[ ]:





# In[8]:


end=datetime.now()


# In[9]:


start=datetime(end.year-1,end.month,end.day)


# In[10]:


tech_list=['AAPL','GOOG','MSFT','AMZN']


# In[11]:


for stock in tech_list:
    globals()[stock] = data.DataReader(stock, 'stooq', start, end)


# In[12]:


AMZN


# In[13]:


AAPL=data.DataReader("AAPL", 'stooq', start, end)
AAPL



# In[14]:


data.DataReader("GBP/USD",'av-forex',start,end,api_key='YourAPIKEY')


# In[15]:


AAPL.describe()


# In[16]:


AAPL.info()


# In[17]:


AAPL['Close'].plot(legend=True,figsize=(10,4))


# In[18]:


AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[19]:


symbols = ["AAPL"]

df = yf.download(tickers=symbols)
df.head()


# In[ ]:





# In[20]:


df.columns = [c.lower() for c in df.columns]


ndf = pd.DataFrame()
for c in df.columns:
    nc = df[c].isna().sum()
    tr = len(df[c])
    rate = nc/tr
    ndf = ndf.append({"col_name":c,"total_rows": tr, 
                "null_rows": nc,
                "rate": rate},ignore_index=True)
ndf



# In[21]:


fig = df.iplot(kind="hist",subplots=True, title="Distribution of All Variables", asFigure=True)

fig.show()


# In[22]:


fig = df.iplot(kind="box",subplots=True, title="Box of All Variables", asFigure=True)

fig.show()


# In[23]:


df.describe()


# In[24]:


fig=df.iplot(kind="line",subplots=True, title="Trend of All Variables", asFigure=True)

fig.show()


# In[25]:


tdf = df.copy()
smadf = tdf.rolling(window=5).mean()
smadf


# In[26]:


for c in smadf.columns:
    tdf[f"sma_{c}"] = smadf[c]
tdf


# In[27]:


smac = [c for c in tdf.columns if "sma" in c]
col = [c for c in tdf.columns if "sma" not in c]

for s,c in zip(smac,col):
    fig = tdf[[c, s]].iplot(kind="line", title=f"{s} vs {c}", xTitle="Date", asFigure=True)
    
    fig.show()


# In[28]:


emadf=df.ewm(span=5, min_periods=5, adjust=True).mean()
emadf


# In[29]:


for c in emadf.columns:
    tdf[f"ema_{c}"] = emadf[c]
tdf


# In[ ]:





# In[30]:


smac = [c for c in tdf.columns if "sma" in c]
wmac = [c for c in tdf.columns if "wma" in c]
emac = [c for c in tdf.columns if "ema" in c]
col = [c for c in tdf.columns if "sma" not in c and "wma" not in c and "ema" not in c]

for s,c,w,e in zip(smac,col, wmac, emac):
    fig=tdf[-100:][[c, s, w, e]].iplot(kind="line", title=f"{s} vs {c} vs {w} vs {e}", xTitle="Date", asFigure=True)
    
    fig.show()


# In[31]:


import plotly.graph_objects as go

fig=go.Figure()

fig.add_trace(go.Candlestick(x=tdf[-1000:].index,
                open=tdf[-1000:]['open'],
                high=tdf[-1000:]['high'],
                low=tdf[-1000:]['low'],
                close=tdf[-1000:]['close'], 
                 name = 'Stock Market Data'))
fig.add_trace(go.Candlestick(x=tdf[-1000:].index,
                open=tdf[-1000:]['ema_open'],
                high=tdf[-1000:]['ema_high'],
                low=tdf[-1000:]['ema_low'],
                close=tdf[-1000:]['ema_close'], 
                 name = 'EMA Stock Market Data'))

fig.update_layout(
    title= "AAPL Stock Data",
    yaxis_title="Stock's Price in USD",
    xaxis_title="Date")               

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=150, label="150D", step="day", stepmode="backward"),
            dict(count=4, label="4m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)

color_hi_fill = 'black'
color_hi_line = 'blue'

color_lo_fill = 'yellow'
color_lo_line = 'purple'

fig.data[0].increasing.fillcolor = color_hi_fill
fig.data[0].increasing.line.color = color_hi_line
fig.data[0].decreasing.fillcolor = 'rgba(0,0,0,0)'
fig.data[0].decreasing.line.color = 'rgba(0,0,0,0)'

fig.data[1].increasing.fillcolor = 'rgba(0,0,0,0)'
fig.data[1].increasing.line.color = 'rgba(0,0,0,0)'
fig.data[1].decreasing.fillcolor = color_lo_fill
fig.data[1].decreasing.line.color = color_lo_line



fig.show()


# In[32]:


tdf = df.copy()
smmdf = tdf.rolling(window=5).median()

for c in smmdf.columns:
    tdf[f"smm_{c}"] = smmdf[c]

emadf=df.ewm(span=5, min_periods=5, adjust=True).mean()

for c in emadf.columns:
    tdf[f"ema_{c}"] = emadf[c]

smmc = [c for c in tdf.columns if "smm" in c]
emac = [c for c in tdf.columns if "ema" in c]
col = [c for c in tdf.columns if "smm" not in c and "ema" not in c]

for s,c,e in zip(smmc,col,emac):
    fig=tdf[-100:][[c, s, e]].iplot(kind="line", title=f"{s} vs {c} vs {e}", xTitle="Date", asFigure=True)
    
    fig.show()



# In[33]:


tdf = df.copy()
smmdf = tdf.rolling(window=5).var()

for c in smmdf.columns:
    tdf[f"smv_{c}"] = smmdf[c]

emadf=df.ewm(span=5, min_periods=5, adjust=True).var()

for c in emadf.columns:
    tdf[f"emv_{c}"] = emadf[c]

smmc = [c for c in tdf.columns if "smv" in c]
emac = [c for c in tdf.columns if "emv" in c]
col = [c for c in tdf.columns if "smv" not in c and "emv" not in c]

for s,c,e in zip(smmc,col,emac):
    fig=tdf[-100:][[c, s, e]].iplot(kind="line", y = [s,e], secondary_y=c, title=f"{s} vs vs {e}", xTitle="Date", asFigure=True)
    
    fig.show()



# In[ ]:





# In[34]:


AAPL['Daily Return'] = AAPL['Close'].pct_change()


# In[35]:


AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[36]:


sns.displot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[37]:


AAPL['Daily Return'].hist(bins=100)


# In[38]:


closing_df=data.DataReader(tech_list, 'stooq', start, end)['Close']


# In[39]:


closing_df


# In[40]:


tech_rets=closing_df.pct_change()


# In[41]:


tech_rets


# In[42]:


GOOG=data.DataReader("GOOG", 'stooq', start, end)


# In[43]:


GOOG


# In[44]:


sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen')


# In[45]:


from IPython.display import SVG


# In[46]:


tech_rets.head()


# In[47]:


sns.pairplot(tech_rets.dropna())

returns_fig=sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
# In[48]:


returns_fig=sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[49]:


returns_fig=sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[50]:


import sklearn
sns.heatmap(tech_rets.dropna(),annot=True)


# In[51]:


dataplot=sns.heatmap(tech_rets.dropna().corr(),annot=True)
  
# displaying heatmap
plt.show()


# In[52]:


dataplot=sns.heatmap(closing_df.corr(),annot=True)
  
# displaying heatmap
plt.show()


# In[53]:


#risk anaysis
rets=tech_rets.dropna()
area=np.pi*20
plt.scatter(rets.mean(),rets.std(),alpha=0.5,s=area)
plt.ylim([0.01,0.05])
plt.xlim([-0.003,0.005])
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
     label,
     xy=(x,y),xytext=(50,50),
     textcoords='offset points',ha='right',va='bottom',
     arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=-0.3'))


# In[54]:


sns.displot(AAPL['Daily Return'].dropna(),bins=100)


# In[55]:


rets.head()


# In[56]:


rets['AAPL'].quantile(0.05)


# In[57]:


days=365
dt=1/days
mu=rets.mean()['GOOG']
sigma=rets.std()['GOOG']


# In[58]:


def stock_monte_carlo(start_price,days,mu,sigma):
    price=np.zeros(days)
    price[0]=start_price
    shock=np.zeros(days)
    drift=np.zeros(days)
    for x in range(1,days):
        shock[x]=np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x]=mu*dt
        price[x]=price[x-1]+(price[x-1]*(drift[x]+shock[x]))
    return price    


# In[59]:


GOOG.head()


# In[60]:


start_price=95.95
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte carlo analysis for google')


# In[63]:


runs=10000
simulation=np.zeros(runs)
for run in range(runs):
    simulation[run]=stock_monte_carlo(start_price,days,mu,sigma)[days-1]


# In[66]:


q=np.percentile(simulation,1)

plt.hist(simulation,bins=200)

plt.figtext(0.6,0.8,s="Start price : $%.2f"%start_price)
plt.figtext(0.6,0.7,"mean final price : $%.2f"%simulation.mean())
plt.figtext(0.6,0.6,"VaR(0.99) : $%.2f"%(start_price-q,))
plt.figtext(0.15,0.6,"q(0.99) : $%.2f"%q)
plt.axvline(x=q,linewidth=4,color='r')
plt.title(u"final price distribution for google stock after %s days" %days,weight='bold');



# In[ ]:




