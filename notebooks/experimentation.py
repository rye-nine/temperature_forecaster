#!/usr/bin/env python
# coding: utf-8

# In[16]:


import meteostat as ms


# In[17]:


import sys

sys.path.append("../src")


# In[18]:


from datetime import date
import altair as alt
import pandas as pd
import pickle
import numpy as np
from temperature_forecaster.probability_model import get_final_residuals


# In[19]:


df_list = get_final_residuals()


# In[20]:


miami = df_list[3].copy()


# In[21]:


miami = miami.reset_index()


# In[22]:


miami


# In[23]:


miami["final_residuals"].describe()


# In[31]:


minimum = miami["tmax"].min()
maximum = miami["tmax"].max()


# In[74]:


c1 = alt.Chart(miami).mark_line().encode(
    x = "time",
    #y = alt.Y("tmax", scale = alt.Scale(domain=[minimum,maximum]))
    y = "tmax"
).properties(
    width = 1200
)


# In[75]:


c1


# In[76]:


c2 = alt.Chart(miami).mark_line(color="red").encode(
    x = "time",
    #y = alt.Y("final_prediction:Q", 
              #scale = alt.Scale(domain=[minimum,maximum]))
    y = "final_prediction:Q"
).properties(
    width = 1200
)


# In[77]:


c2


# In[78]:


c1 + c2


# In[79]:


c3 = alt.Chart(miami).mark_line(color = "green").encode(
    x = "time",
    y = "final_residuals:Q"
).properties(
    width = 1200
)


# In[80]:


c3


# In[82]:


c1+c2+c3


# In[83]:


histogram = alt.Chart(miami).mark_bar().encode(
    alt.X("final_residuals:Q", bin=alt.Bin(maxbins=50), title="Residual (°F)"),
    alt.Y("count()", title="Frequency")
)


# In[84]:


histogram


# In[89]:


from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(miami["final_residuals"], dist="norm", plot=plt)
plt.show()


# In[ ]:




