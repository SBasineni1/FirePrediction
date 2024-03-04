
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:52:51 2022
This program reads the weather data file which has Header and detail record. 
The header record which is identified as it starts with # . For my research I only need the 
header data. The following program reads the IGRA data derived file and parses the file to 
extract header files and writes into a new file to import for further analysis
@author: Suchit Basineni
"""
# Importing Libraries

import scipy
from scipy import stats
from sklearn import datasets
import pandas as pd
import seaborn as sb
# import numpy as np
#import os

#Create a variable for the file name
filename = "YNWC1-2021-2023-inputfile.csv"
filenamew = "Fire_data.txt"
#initialize an empty list
word_list= []
#w tells python we are opening the file to write into it
outfile = open(filenamew, 'w')
#Open the file
infile = open(filename, 'r') 
lines = infile.readlines() 
for line in lines: #lines is a list with each item representing a line of the file
	if ',' in line:
         word_list.append(line)	
outfile.writelines(word_list) 
outfile.close()
infile.close() #close the file when you're done!

# Read the fixed format file into a Pandas Data Frame

cols = [']
df = pd.read_fwf('/Users/basin/.vscode/Weather/Fire_data', 
                 header=None,widths=[12,5,3,3,3,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
                 names=cols)
df.shape
print(df)
df.describe(include='all')

'''df_2007 = df[df.Year >= 2007 ]

df_2007 = df_2007[(df_2007.PW != -99999)]
df_2007 = df_2007[(df_2007.CAPE != -99999)]  
# No of Rows and columns                
df.shape

# Variable types 
df.dtypes

#Used to import the actual tornado reports
df_tor = pd.read_csv('1950-2021_actual_tornadoes.csv')
df_tor.describe(include='all').transpose()
df_tor_f = df_tor[['om','yr','mo','dy','st','mag']]
#removing duplicate tornado records
df_tor_u =  df_tor_f.drop_duplicates()

# using group by to find the maximum F-Scale Tornado happened on that day
df_max = df_tor_u.groupby(['yr','mo','dy','st'])['mag'].max().reset_index(name='max_mag_f_scale')
# Using group by to add the no of recorded tornadoes happened by date and state
df_cnt = df_tor_u.groupby(['yr','mo','dy','st']).size().reset_index(name='no_of_tornadoes')
#Merge the data to prepare a final data frame to refer for the days tornadoes happened.
df_grp = pd.merge(df_max,df_cnt, how='inner')
#Filtering data by year and state
df_tor_2007 = df_grp[df_grp.yr >= 2007]
df_tor_2007_OK = df_tor_2007[df_tor_2007.st == "OK"]

#frames = [df_tor_2013_OK, df]
frames = pd.merge(df_2007, df_tor_2007_OK, how='left',left_on=['Year','Month','Day'] , right_on = ['yr','mo','dy'])
final_frames = frames[['Year','Month','Day','Hour','RELTIME','PW','st','max_mag_f_scale','no_of_tornadoes',
                       'LCLPRESS','LNBHGT','LI','SI','KI','TTI','CAPE','CIN']] 
# Fill the NaN values 
df_final =final_frames.fillna(value={'st':'OK','no_of_tornadoes':0,'max_mag_f_scale':-9})
df_final.rename(columns = {'max_mag_f_scale':'f_scale'}, inplace = True)

df_corr=  df_final[['st','f_scale','PW','LCLPRESS','LI','SI','KI','TTI','LNBHGT','CAPE','CIN']] 
pearsoncorr = round(df_corr.corr(method='pearson'),3)
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
# df_corr1=  final_frames[final_frames.PW != -99999]
df_corr1=  df_final[['st','f_scale','no_of_tornadoes','PW','LCLPRESS','LI','SI','KI','TTI']] 

#df_corr2=  final_frames[final_frames.CAPE != -99999]
df_corr2=  df_final[['st','f_scale','no_of_tornadoes','LNBHGT','CAPE','CIN']] 

pearsoncorr2 = df_corr2.corr(method='pearson')
pearsoncorr1 = df_corr1.corr(method='pearson')
pearsoncorr1
sb.heatmap(pearsoncorr2, 
            xticklabels=pearsoncorr2.columns,
            yticklabels=pearsoncorr2.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
sb.heatmap(pearsoncorr1, 
            xticklabels=pearsoncorr1.columns,
            yticklabels=pearsoncorr1.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

#[frames.Year,frames.Month,frames.Day]
#result = pd.concat(frames)
 

#Descriptive Statistics.
df_dsc_Stat= df_corr.describe().transpose()




#Getting the Corelation coefficient and the P Value 
coff_df = pd.DataFrame(columns=['r','p'])
 
for col in df_corr:
    print(col)
    if pd.api.types.is_numeric_dtype(df_corr[col]):
        r,p = stats.pearsonr(df_corr.f_scale,df_corr[col])
        coff_df.loc[col] = [round(r,4),round(p,5)]
    
coff_df

coff_df.rename(columns = {'r':'corr-coeff','p':'p-value'}, inplace = True)


# Graphical analysis 
df_corr['f_scale']=df_corr.f_scale.astype(str)
sb.histplot(data=df_corr[df_corr.f_scale != '-9.0'], x= 'f_scale' )
#import seaborn as sns
sb.scatterplot(x="CAPE", y="CIN", hue="f_scale" ,data=df_corr);
sb.scatterplot(x="CAPE", y="CIN", hue="f_scale" ,data=df_corr[df_corr.f_scale != '-9.0']);
sb.boxplot(x=df_corr['CAPE'], y=df_corr['f_scale'], showmeans=True)
sb.displot(x='CAPE', col='f_scale', data=df_corr[df_corr.f_scale != '-9.0'], linewidth=3, kde=True);
sb.displot(x='CAPE',  data=df_corr[df_corr.f_scale != '-9.0'], linewidth=1, kde=True);'''
