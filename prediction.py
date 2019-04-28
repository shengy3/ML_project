# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:59:03 2018

@author: evanw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# In[ ]:
# Read Files
matches = pd.read_csv('matches.csv')
players = pd.read_csv('players.csv',encoding='latin1',index_col=0)
q_matches = pd.read_csv('qualifying_matches.csv')
ranks = pd.read_csv('rankings.csv')

## cleaning up data
#for i in range(9,len(matches.iloc[2])-1):
#   matches.iat(11906,i] = matches.iloc[11906][i+1])

matches.iat[11906,9] = float('NaN')
matches.iat[11906,15] = '6-2 6-3'
matches.iat[11906,16] = 'Hard'
matches.iat[11906,17] = 20030421
matches.iat[11906,18] = '2003-D069'
matches.iat[11906,19] = 'D'
matches.iat[11906,20] = float('NaN')
matches.iat[11906,21] = 16.42984
matches.iat[11906,22] = float('NaN')
matches.iat[11906,23] ='R'
matches.iat[11906,24] =153
matches.iat[11906,25] = 201420
matches.iat[11906,26] = 'IND'
matches.iat[11906,27] = 'Sania Mirza'
matches.iat[11906,28] = 425
matches.iat[11906,29] =36
matches.iat[11906,30] =float('NaN')
matches.iat[11906,31] =2003

#mymap = {'O':float('NaN')}
#matches.applymap(lambda s: mymap.get(s) if s in mymap else s)
matches.iat[11912,28] =float('NaN')
matches.iat[11914,28] =float('NaN')
'''
Splitting matches based on surface
'''
surface_list  = matches['surface'].unique().tolist()
m_clay = matches[matches['surface'] =='Clay']
m_grass = matches[matches['surface'] =='Grass']
m_Hard  = matches[matches['surface'] =='Hard']
m_Carpet = matches[matches['surface'] =='Carpet']


# In[ ]:
"""
Winner's age histogram at different surface
"""
bins = np.linspace(10, 45,(45-9))
age_diff = round(matches['winner_age'])
age_diff = age_diff.dropna()
age_diff = age_diff.astype(int)

age_diff.plot.hist(bins,title="Winner's Age")
plt.show()

age_diff_g = round(m_grass['winner_age'])
age_diff_g = age_diff_g.dropna()
age_diff_g = age_diff_g.astype(int)
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
plt.figure(figsize=(14,7))
plt.subplot(221)
age_diff_g.plot.hist(bins,title="Winner's Age on grass")

age_diff_h = round(m_Hard['winner_age'])
age_diff_h = age_diff_h.dropna()
age_diff_h = age_diff_h.astype(int)
plt.subplot(222)
age_diff_h.plot.hist(bins,title="Winner's Age on hard")

age_diff_cp = round(m_Carpet['winner_age'])
age_diff_cp = age_diff_cp.dropna()
age_diff_cp = age_diff_cp.astype(int)
plt.subplot(223)
age_diff_cp.plot.hist(bins,title="Winner's Age on carpet")

age_diff_c = round(m_clay['winner_age'])
age_diff_c = age_diff_c.dropna()
age_diff_c = age_diff_c.astype(int)
plt.subplot(224)
age_diff_c.plot.hist(bins,title="Winner's Age on clay")
plt.show()


# In[ ]:
'''
Checking upset rate
Define upset as ranking difference > 10
'''

## Overall upset rate


matches['loser_rank'] = pd.to_numeric(matches['loser_rank'])
matches['winner_rank'] = pd.to_numeric(matches['winner_rank'])
matches['loser_rank'] = matches['loser_rank'].fillna(1483)
matches['winner_rank'] = matches['winner_rank'].fillna(1483)
matches['rk_diff'] = matches['loser_rank'] - matches['winner_rank']

upset_rate = sum(x<0 for x in matches['rk_diff'])/len(matches)
print('Upset rate = ',upset_rate)

'''
Upsets at different rankings, how often does it happens in the 
Top1-20, 20-40, 40-60, 60-80,80-100, 100-Unrank
'''
ups_t20_counter_1 = 0
ups_t20_counter_2 = 0
ups_t40_counter_1 = 0
ups_t40_counter_2 = 0
ups_t60_counter_1 = 0
ups_t60_counter_2 = 0
ups_t80_counter_1 = 0
ups_t80_counter_2 = 0
ups_t100_counter_1 = 0
ups_t100_counter_2 = 0
ups_tR_counter_1 = 0
ups_tR_counter_2 = 0
ups_t200_counter_1 = 0
ups_t200_counter_2 = 0
ups_t400_counter_1 = 0
ups_t400_counter_2 = 0
ups_t800_counter_1 = 0
ups_t800_counter_2 = 0




for x in range(0,len(matches)):
        if (matches['loser_rank'][x] <= 20) & (matches['winner_rank'][x]<=20):
            ups_t20_counter_1 = ups_t20_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t20_counter_2 = ups_t20_counter_2+1
                
for x in range(0,len(matches)):                
        if (matches['loser_rank'][x] >= 21) & (matches['loser_rank'][x]<=40) &  (matches['winner_rank'][x]>=21) & (matches['winner_rank'][x]<=40):
            ups_t40_counter_1 = ups_t40_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t40_counter_2 = ups_t40_counter_2+1        
        
for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 41) & (matches['loser_rank'][x]<=60) &  (matches['winner_rank'][x]>=41) & (matches['winner_rank'][x]<=60):
            ups_t60_counter_1 = ups_t60_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t60_counter_2 = ups_t60_counter_2+1     
        
for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 61) & (matches['loser_rank'][x]<=80) &  (matches['winner_rank'][x]>=61) & (matches['winner_rank'][x]<=80):
            ups_t80_counter_1 = ups_t80_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t80_counter_2 = ups_t80_counter_2+1             
        
for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 81) & (matches['loser_rank'][x]<=100) &  (matches['winner_rank'][x]>=81) & (matches['winner_rank'][x]<=100):
            ups_t100_counter_1 = ups_t100_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t100_counter_2 = ups_t100_counter_2+1                  

for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 101) & (matches['loser_rank'][x]<=200) &  (matches['winner_rank'][x]>=101) & (matches['winner_rank'][x]<=200):
            ups_t200_counter_1 = ups_t200_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t200_counter_2 = ups_t200_counter_2+1   
                
for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 201) & (matches['loser_rank'][x]<=400) &  (matches['winner_rank'][x]>=201) & (matches['winner_rank'][x]<=400):
            ups_t400_counter_1 = ups_t400_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t400_counter_2 = ups_t400_counter_2+1   

for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 401) & (matches['loser_rank'][x]<=800) &  (matches['winner_rank'][x]>=401) & (matches['winner_rank'][x]<=800):
            ups_t800_counter_1 = ups_t800_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_t800_counter_2 = ups_t800_counter_2+1   
 
for x in range(0,len(matches)):        
        if (matches['loser_rank'][x] >= 801)  &  (matches['winner_rank'][x]>=801) :
            ups_tR_counter_1 = ups_tR_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                ups_tR_counter_2 = ups_tR_counter_2+1          
                
                
t20_upset_rate = ups_t20_counter_2/ups_t20_counter_1  
t40_upset_rate = ups_t40_counter_2/ups_t40_counter_1
t60_upset_rate = ups_t60_counter_2/ups_t60_counter_1
t80_upset_rate = ups_t80_counter_2/ups_t80_counter_1
t100_upset_rate = ups_t100_counter_2/ups_t100_counter_1
t200_upset_rate = ups_t200_counter_2/ups_t200_counter_1
t400_upset_rate = ups_t400_counter_2/ups_t400_counter_1
t800_upset_rate = ups_t800_counter_2/ups_t800_counter_1
ab800_upset_rate = ups_tR_counter_2/ups_tR_counter_1
   
print('Top 20 upset rate = ',t20_upset_rate, 'in', ups_t20_counter_1, 'match')
print('Top 40 upset rate = ',t40_upset_rate, 'in', ups_t40_counter_1, 'match')
print('Top 60 upset rate = ',t60_upset_rate, 'in', ups_t60_counter_1, 'match')
print('Top 80 upset rate = ',t80_upset_rate, 'in', ups_t80_counter_1, 'match')
print('Top 100 upset rate = ',t100_upset_rate, 'in', ups_t100_counter_1, 'match')
print('Top 200 upset rate = ',t200_upset_rate, 'in', ups_t200_counter_1, 'match')
print('Top 400 upset rate = ',t400_upset_rate, 'in', ups_t400_counter_1, 'match')
print('Top 800 upset rate = ',t800_upset_rate, 'in', ups_t800_counter_1, 'match')
print('Above 800 upset rate = ',ab800_upset_rate, 'in', ups_tR_counter_1, 'match')



'''
Upset rate at different range, 5, 10, 20, 50, 100
'''
rk_5_dif_counter_1 = 0
rk_5_dif_counter_2 = 0
rk_10_dif_counter_1 = 0
rk_10_dif_counter_2 = 0
rk_20_dif_counter_1 = 0
rk_20_dif_counter_2 = 0
rk_50_dif_counter_1 = 0
rk_50_dif_counter_2 = 0
rk_100_dif_counter_1 = 0
rk_100_dif_counter_2 = 0
rk_200_dif_counter_1 = 0
rk_200_dif_counter_2 = 0
rk_500_dif_counter_1 = 0
rk_500_dif_counter_2 = 0
rk_500abv_dif_counter_1 = 0
rk_500abv_dif_counter_2 = 0
for x in range(0,len(matches)):
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=5:
            rk_5_dif_counter_1 = rk_5_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_5_dif_counter_2 = rk_5_dif_counter_2+1
                
for x in range(0,len(matches)):                
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=10:
            rk_10_dif_counter_1 = rk_10_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_10_dif_counter_2 = rk_10_dif_counter_2+1
                
for x in range(0,len(matches)):
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=20:
            rk_20_dif_counter_1 = rk_20_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_20_dif_counter_2 = rk_20_dif_counter_2+1
                
for x in range(0,len(matches)):
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=50:
            rk_50_dif_counter_1 = rk_50_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_50_dif_counter_2 = rk_50_dif_counter_2+1
                
for x in range(0,len(matches)):                
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=100:
            rk_100_dif_counter_1 = rk_100_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_100_dif_counter_2 = rk_100_dif_counter_2+1
                
for x in range(0,len(matches)):                
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=200:
            rk_200_dif_counter_1 = rk_200_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_200_dif_counter_2 = rk_200_dif_counter_2+1
                
for x in range(0,len(matches)):                
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])<=500:
            rk_500_dif_counter_1 = rk_500_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_500_dif_counter_2 = rk_500_dif_counter_2+1
                
for x in range(0,len(matches)):
        if abs(matches['loser_rank'][x] - matches['winner_rank'][x])>500:
            rk_500abv_dif_counter_1 = rk_500abv_dif_counter_1+1
            if matches['loser_rank'][x]<matches['winner_rank'][x]:
                rk_500abv_dif_counter_2 = rk_500abv_dif_counter_2+1

rk_5_dif_up_rate = rk_5_dif_counter_2/rk_5_dif_counter_1
rk_10_dif_up_rate = rk_10_dif_counter_2/rk_10_dif_counter_1
rk_20_dif_up_rate = rk_20_dif_counter_2/rk_20_dif_counter_1
rk_50_dif_up_rate = rk_50_dif_counter_2/rk_50_dif_counter_1
rk_100_dif_up_rate = rk_100_dif_counter_2/rk_100_dif_counter_1
rk_200_dif_up_rate = rk_200_dif_counter_2/rk_200_dif_counter_1
rk_500_dif_up_rate = rk_500_dif_counter_2/rk_500_dif_counter_1
rk_500abv_dif_up_rate = rk_500abv_dif_counter_2/rk_500abv_dif_counter_1

print('Rank 5 difference upset rate = ',rk_5_dif_up_rate, 'in', rk_5_dif_counter_1, 'match')
print('Rank 10 difference upset rate = ',rk_10_dif_up_rate, 'in', rk_10_dif_counter_1, 'match')
print('Rank 20 difference upset rate = ',rk_20_dif_up_rate, 'in', rk_20_dif_counter_1, 'match')
print('Rank 50 difference upset rate = ',rk_50_dif_up_rate, 'in', rk_50_dif_counter_1, 'match')
print('Rank 100 difference upset rate = ',rk_100_dif_up_rate, 'in', rk_100_dif_counter_1, 'match')
print('Rank 200 difference upset rate = ',rk_200_dif_up_rate, 'in', rk_200_dif_counter_1, 'match')
print('Rank 500 difference upset rate = ',rk_5_dif_up_rate, 'in', rk_500_dif_counter_1, 'match')
print('Rank 500 above difference upset rate = ',rk_500abv_dif_up_rate, 'in', rk_500abv_dif_counter_1, 'match')

rk_dif_data = np.array([rk_5_dif_up_rate,rk_10_dif_up_rate,rk_20_dif_up_rate,rk_50_dif_up_rate,rk_100_dif_up_rate,rk_200_dif_up_rate,rk_500_dif_up_rate,rk_500abv_dif_up_rate])
rk_dif_x    = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
rk_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0])
rk_xticks = ['5', '10', '20', '50', '100', '200', '500', 'above 500']
plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
plt.xticks(x, rk_xticks)
plt.plot(rk_x,rk_dif_data,'bo')
plt.xlabel('Rank Difference')
plt.ylabel('Upset Rate')
plt.title('Rank Difference vs. Upset Rate')


x = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
y = np.array([t20_upset_rate,t40_upset_rate,t60_upset_rate,t80_upset_rate,t100_upset_rate,t200_upset_rate,t400_upset_rate,t800_upset_rate,ab800_upset_rate])
my_xticks = ['1-20','20-40','40-60','60-80','80-100','100-200','200-400','400-800','800-Unrank']

plt.subplot(1,2,2)
plt.xticks(x, my_xticks)
plt.plot(x, y,'bo')
plt.xlabel('Rank Interval')
plt.ylabel('Upset Rate')
plt.title('Rank Interval vs. Upset Rate')
plt.show()