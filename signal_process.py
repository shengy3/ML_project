#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
raw_data = genfromtxt('signal.csv', delimiter=',')


# In[79]:


df = pd.DataFrame(raw_data)


# In[80]:


plt.plot(df.iloc[0,:])
plt.xlabel("channel")
plt.ylabel("adc value")
plt.xlim([0,300])


# In[ ]:




