from pytrends.request import TrendReq
import plotly.express as px

pytrends = TrendReq(hl='en-US', tz=360) 

# build payload

kw_list = ["Sooners"] # list of keywords to get data 
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-D') 

#1 Interest over Time
data = pytrends.interest_over_time() 
data = data.reset_index() 

import plotly.express as px
fig = px.line(data, x="date", y=['Sooners'], title='Keyword Web Search Interest Over Time')
fig.show() 