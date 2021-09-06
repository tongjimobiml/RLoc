#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import numpy as np
import pandas as pd
import math as Math
import matplotlib
matplotlib.use("Pdf")
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import random
import grid


# In[2]:


import time
import datetime

def compute_time_interval(start, end):
   
    start = datetime.datetime.fromtimestamp(start / 1000.0)
    end = datetime.datetime.fromtimestamp(end / 1000.0)
    
    # 相减得到秒数
    seconds = (end- start).seconds
    
    return seconds


# In[3]:


def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p


# In[4]:


from scipy.spatial.distance import pdist
def distribution(data,low=0,up=125,r=5,bins=25):
    p_list=np.zeros((bins))
    i=0
    total=len(data)
    #print total
    while low + r <= up:
        for t in data:
            if t >= low and t < low + r:
                p_list[i] += 1
        low += r
        i += 1
    p_u_list=list()
    #print p_list
    if len(data)>0:
        for t in p_list:
            #print int(t)
            p_u_list.append(float(t)/float(total))
    else:
        p_u_list=p_list.tolist()
    return p_u_list
    
def p_norm_distance(x,y):
    #print x,y
    X = np.vstack([x,y])
    d2 = pdist(X,'minkowski',p=3)
    return d2


# In[5]:


rc = 6378137
rj = 6356725
from math import atan, cos, asin, sqrt, pow, pi, sin
def rad(d):
    return d * math.pi / 180.0

def azimuth(pt_a, pt_b):
    lon_a, lat_a = pt_a
    lon_b, lat_b = pt_b
    rlon_a, rlat_a = rad(lon_a), rad(lat_a)
    rlon_b, rlat_b = rad(lon_b), rad(lat_b)
    ec=rj+(rc-rj)*(90.-lat_a)/90.
    ed=ec*cos(rlat_a)

    dx = (rlon_b - rlon_a) * ec
    dy = (rlat_b - rlat_a) * ed
    if dy == 0:
        angle = 90. 
    else:
        angle = atan(abs(dx / dy)) * 180.0 / pi
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    if dlon > 0 and dlat <= 0:
        angle = (90. - angle) + 90
    elif dlon <= 0 and dlat < 0:
        angle = angle + 180 
    elif dlon < 0 and dlat >= 0:
        angle = (90. - angle) + 270 
    return angle

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
    Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s

def sq(x):
    return x*x


# In[6]:


col_name_new = [
    #'Num_connected',
    'TrajID',
    'RNCID_1',
    'CellID_1',
    'AsuLevel_1',
    'Dbm_1',
    'SignalLevel_1',
    'RNCID_2',
    'CellID_2',
    'AsuLevel_2',
    'Dbm_2',
    'SignalLevel_2',
    'RNCID_3',
    'CellID_3',
    'AsuLevel_3',
    'Dbm_3',
    'SignalLevel_3',
    'RNCID_4',
    'CellID_4',
    'AsuLevel_4',
    'Dbm_4',
    'SignalLevel_4',
    'RNCID_5',
    'CellID_5',
    'AsuLevel_5',
    'Dbm_5',
    'SignalLevel_5',
    'RNCID_6',
    'CellID_6',
    'AsuLevel_6',
    'Dbm_6',
    'SignalLevel_6',
    #'RSSI_6',
]


# In[7]:


col_name_rf = [
    'RNCID_1',
    'CellID_1',
    'AsuLevel_1',
    'Dbm_1',
    'SignalLevel_1',
    'RNCID_2',
    'CellID_2',
    'AsuLevel_2',
    'Dbm_2',
    'SignalLevel_2',
    'RNCID_3',
    'CellID_3',
    'AsuLevel_3',
    'Dbm_3',
    'SignalLevel_3',
    'RNCID_4',
    'CellID_4',
    'AsuLevel_4',
    'Dbm_4',
    'SignalLevel_4',
    'RNCID_5',
    'CellID_5',
    'AsuLevel_5',
    'Dbm_5',
    'SignalLevel_5',
    'RNCID_6',
    'CellID_6',
    'AsuLevel_6',
    'Dbm_6',
    'SignalLevel_6',
    'Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4','Lon5','Lat5','Lon6','Lat6'
]


# In[8]:


def merge_2g_engpara():
    #eng_para = pd.read_csv('DLoc/siping_2g_new_gongcan.csv', encoding='gbk')
    eng_para = pd.read_csv('2g/siping_2g_new_gongcan.csv', encoding='gbk')
    eng_para = eng_para[['RNCID', 'CellID', 'BSID','Lon','Lat']]
    #eng_para = eng_para[eng_para.LAC.notnull() & eng_para[u'经度'].notnull()]
    eng_para = eng_para.drop_duplicates()
    #eng_para.rename(columns={u'经度': 'lon', u'纬度': 'lat'}, inplace=True)
    return eng_para

def make_rf_dataset(data, eng_para):
    for i in range(1, 7):
        data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['RNCID','CellID'], how='left', suffixes=('', '%d' % i))
        temp=data['CellID_%d'% i].tolist()
        new=list()
        for item in temp:
            if math.isnan(item):
                new.append(0)
            elif int(item)<=0:
                new.append(0)
            else:
                new.append(item)
        data['CellID_%d' % i]=new
    data = data.fillna(-999.)
    #print data.columns
    
    feature = data[col_name_new+['mode','MRTime','BSID','BSID2','BSID3','BSID4','BSID5','BSID6','Longitude', 'Latitude',
                                'Lon','Lat','Lon2','Lat2','Lon3','Lat3','Lon4','Lat4','Lon5','Lat5','Lon6','Lat6']]
   
    
    subset=[u'Longitude', u'Latitude', 
       u'RNCID_1', u'CellID_1',u'Dbm_1',
       u'RNCID_2', u'CellID_2',u'Dbm_2',
       u'RNCID_3', u'CellID_3',u'Dbm_3',
       u'RNCID_4', u'CellID_4',u'Dbm_4',
       u'RNCID_5', u'CellID_5',u'Dbm_5',
       u'RNCID_6', u'CellID_6',u'Dbm_6',
       ]
    #feature=feature.drop_duplicates(subset=subset) 
    label = feature[['Longitude', 'Latitude']]
    feature= feature.drop(['Longitude', 'Latitude'],axis=1)
    
    return feature, label

#eng_para = merge_2g_engpara()
eng_para =merge_2g_engpara()


# In[9]:


def conf_model_label(error):
    conf_l=list()
    
    for t in error:
        if t<=50:
            conf_l.append(1)
        else:
            conf_l.append(0)
    
    return conf_l


# In[10]:


data_2g=pd.read_csv("2g/siping_2g_mode.csv")


# In[ ]:


# 1、邻接基站实验
# bs_num = []
# for idx,row in data_2g.iterrows():
#     for i in range(1,8):
#         if row['RNCID_%d'%i]==0 or math.isnan(row['RNCID_%d'%i]) or row['CellID_%d'%i]==-1 or math.isnan(row['CellID_%d'%i]):
#             bs_num.append(i-1)
#             break
#         if i == 7:
#             bs_num.append(i)        
# data_2g['bs_num'] = bs_num
# retain_num = int(sys.argv[1])
# # retain_num = 1
# for idx,row in data_2g.iterrows():
#     if row['bs_num']-1 <= retain_num:
#         continue
#     for i in range(retain_num+2, row['bs_num']+1):
#         data_2g.loc[idx, 'RNCID_%d'%i] = np.nan
#         data_2g.loc[idx, 'CellID_%d'%i] = np.nan
#         data_2g.loc[idx, 'AsuLevel_%d'%i] = np.nan
#         data_2g.loc[idx, 'SignalLevel_%d'%i] = np.nan
#         data_2g.loc[idx, 'Dbm_%d'%i] = np.nan


# In[ ]:


# 2、基站密度实验
bs = []
for i in range(1, 8):
    bs += data_2g[['RNCID_%d'% i, 'CellID_%d'% i]].values.tolist()
bs = [tuple(t) for t in bs]
temp = []
[temp.append(i) for i in bs if not i in temp]
bs = temp
ratio = float(sys.argv[1])
drop_bs = random.sample(bs, int(len(bs) * ratio))
for idx, row in data_2g.iterrows():
    for i in range(1, 7):
        if (row['RNCID_%d'% i], row['CellID_%d'% i]) in drop_bs:
            data_2g.loc[idx, 'RNCID_%d'% i] = -999
            data_2g.loc[idx, 'CellID_%d'% i] = -999
            data_2g.loc[idx, 'Dbm_%d'% i] = -999
data_2g = data_2g.drop(data_2g[data_2g['RNCID_1']==-999].index)    


# In[ ]:


# 3、运动模式实验
# mode = int(sys.argv[1])
# data_2g = data_2g[data_2g['mode'] == mode]


# In[128]:


train, label = make_rf_dataset(data_2g, eng_para)
from sklearn.cross_validation import train_test_split
tr_feature_r, te_feature_r, tr_label_, te_label_ = train_test_split(train, label, test_size=0.4,random_state=50)


# In[129]:


tr_feature_r.shape[0]


# In[130]:


def conf_pre(data):
    data=data.iloc[:,1:]
    label=data[['Longitude','Latitude']]
    data=data.drop(['Longitude', 'Latitude'],axis=1)
    
    return data, label


# In[131]:


con_tr_feature, con_te_feature, con_tr_p, con_te_p = train_test_split(te_feature_r, te_label_, test_size=0.4)


# In[132]:


from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor


# In[133]:


import grid
#from grid import RoadGrid
#importlib.reload(grid)
#from grid import RoadGrid
rg = grid.RoadGrid(np.vstack((tr_label_.values, te_label_.values)),100)
tr_label_g = rg.transform(tr_label_.values, False)
#rint tr_label_
con_tr_j = rg.transform(con_tr_p.values, False)
con_te_j = rg.transform(con_te_p.values, False)


# In[134]:


est=RandomForestClassifier( n_jobs=-1,
    n_estimators =50,
    max_features='sqrt').fit(tr_feature_r[col_name_rf].values, tr_label_g)
    #est=DecisionTreeRegressor(max_depth=4).fit(tr_f.values, tr_l.values)
pred_con_tr=est.predict(con_tr_feature[col_name_rf].values)
pred_con_te=est.predict(con_te_feature[col_name_rf].values)
tr_pred = np.array([rg.grid_center[idx] for idx in pred_con_tr])
te_pred = np.array([rg.grid_center[idx] for idx in pred_con_te])
error_tr = [distance(pt1, pt2) for pt1, pt2 in zip(tr_pred, con_tr_p.values)]
#error_tr = sorted(error_tr)
error_te = [distance(pt1, pt2) for pt1, pt2 in zip(te_pred, con_te_p.values)]
#error_te = sorted(error_te)


# In[137]:


tr_feature_r['Longitude'] = tr_label_.iloc[:, 0].values
tr_feature_r['Latitude'] = tr_label_.iloc[:, 1].values
tr_feature_r['gid'] = tr_label_g


# In[177]:


#tr_feature_r.to_csv("2g/total_conf_tr_sp2g.csv")
#con_tr_feature.to_csv("2g/conf_tr_sp2g.csv")
#con_te_feature.to_csv("2g/conf_te_sp2g.csv")


# In[140]:


conf_label_tr = conf_model_label(error_tr)
conf_label_te = conf_model_label(error_te)


# In[141]:


con_tr_feature["conf"]= conf_label_tr
con_te_feature["conf"]= conf_label_te
con_tr_feature["error"]= error_tr
con_te_feature["error"]= error_te


# In[142]:


con_tr_feature['Longitude'] = con_tr_p.iloc[:,0]
con_tr_feature['Latitude'] = con_tr_p.iloc[:,1]
con_te_feature['Longitude'] = con_te_p.iloc[:,0]
con_te_feature['Latitude'] = con_te_p.iloc[:,1]


# In[143]:


con_tr_feature['p_gid'] = pred_con_tr
con_tr_feature['gid'] = con_tr_j
con_te_feature['p_gid'] = pred_con_te
con_te_feature['gid'] = con_te_j


# In[144]:


def rss_level(dbm):
    if dbm>-50:
        return 1
    elif dbm >-60:
        return 2
    elif dbm >-70:
        return 3
    elif dbm>-80:
        return 4
    elif dbm>-90:
        return 5
    elif dbm>-100:
        return 6
    elif dbm>-110:
        return 7
    else:
        return 8


# In[145]:


for i in range(1, 7):
    con_tr_feature['rss_level_%d' % i] = con_tr_feature['Dbm_%d' % i].map(lambda x: rss_level(x))  
    con_te_feature['rss_level_%d' % i] = con_te_feature['Dbm_%d' % i].map(lambda x: rss_level(x))  


# In[146]:


con_te_feature.shape[0]


# In[147]:


total_ob = con_tr_feature[['BSID','rss_level_1','BSID2','rss_level_2',
               'BSID3','rss_level_3','BSID4','rss_level_4',
               'BSID5','rss_level_5','BSID6','rss_level_6',]].drop_duplicates()


# In[148]:


total_ob_bs = con_tr_feature[['BSID','BSID2','BSID3','BSID4','BSID5','BSID6',]].drop_duplicates()


# In[149]:


total_ob_rss = con_tr_feature[['rss_level_1','rss_level_2','rss_level_3',
                               'rss_level_4','rss_level_5','rss_level_6',]].drop_duplicates()


# In[150]:


total_ob_te = con_te_feature[['BSID','rss_level_1','BSID2','rss_level_2',
               'BSID3','rss_level_3','BSID4','rss_level_4',
               'BSID5','rss_level_5','BSID6','rss_level_6',]].drop_duplicates()


# In[151]:


total_ob_te.shape[0]


# In[152]:


def jaccard_sim(list1, list2):
    #print list1, list2
    union_set = len(set(list1)|set(list2))#并集长度
    intersection_set = len(set(list1)&set(list2))#交集长度

    Jaccard = float(intersection_set/union_set) #Jaccar
    return Jaccard


# In[153]:


def adaptive_emission_pro(jd_list, match, bs_list, ss_list, can_list, total_c, conf):
    pro_list = []
    weight_list = []
    if match.shape[0]>0:
        weight_list.append(math.log(1+match.shape[0])*1)
        match_ss= match[(match['rss_level_1']== int(ss_list[0])) & (match['rss_level_2']==int(ss_list[1])) & (match['rss_level_3']==ss_list[2])
             & (match['rss_level_4']==ss_list[3]) & (match['rss_level_5']==ss_list[4]) & (match['rss_level_6']==ss_list[5])]
        if match_ss.shape[0]>0:
            pro_list.append(float(match_ss[match_ss['conf']==conf].shape[0])/float(total_c))
        else:
            pro_list.append(float(match[match['conf']==conf].shape[0])/float(total_c))
        
        
    for can_temp, jd in zip(can_list, jd_list):
        match_c = con_tr_feature[(con_tr_feature['BSID']==int(can_temp[0])) & (con_tr_feature['BSID2']==int(can_temp[1]))
                      &(con_tr_feature['BSID3']==int(can_temp[2])) & (con_tr_feature['BSID4']==int(can_temp[3]))
                      &(con_tr_feature['BSID5']==int(can_temp[4])) & (con_tr_feature['BSID6']==int(can_temp[5]))]
        count = match_c.shape[0]
        weight_list.append(math.log(1+count)*jd)
        
        match_ss_c= match_c[(match_c['rss_level_1']== ss_list[0]) & (match_c['rss_level_2']== ss_list[1]) 
                            & (match_c['rss_level_3']==ss_list[2])& (match_c['rss_level_4']==ss_list[3]) 
                            & (match_c['rss_level_5']==ss_list[4]) & (match_c['rss_level_6']==ss_list[5])]
        if match_ss_c.shape[0]>0:
            pro_list.append(float(match_ss_c[match_ss_c['conf']==conf].shape[0])/float(total_c))
        else:
            pro_list.append(float(match_c[match_c['conf']==conf].shape[0])/float(total_c))


    weight_sum = np.sum(weight_list)
    ad_em_po = 0
    
    for x, y in zip(pro_list, weight_list):
        ad_em_po += x * (y / weight_sum)
    
    return ad_em_po


# In[154]:


zeros_list = []
i =0
j=0
zero_num = con_tr_feature[con_tr_feature['conf']==0].shape[0]
for idx, row in total_ob_te.iterrows():
    bs_list =row[['BSID','BSID2','BSID3','BSID4','BSID5','BSID6']].values
    ss_list = row[['rss_level_1','rss_level_2','rss_level_3','rss_level_4','rss_level_5','rss_level_6',]].values
    #print bs_list[0]
    match = con_tr_feature[(con_tr_feature['BSID']==int(bs_list[0])) & (con_tr_feature['BSID2']==int(bs_list[1]))
                          &(con_tr_feature['BSID3']==int(bs_list[2])) & (con_tr_feature['BSID4']==int(bs_list[3]))
                          &(con_tr_feature['BSID5']==int(bs_list[4])) & (con_tr_feature['BSID6']==int(bs_list[5]))]
    if match.shape[0]<5:
        can_bs_row = []
        can_j = []
        jaccd_max = []
        idx_list = []
        for idxx, roww in total_ob_bs.iterrows():
            can_bs_list = roww.values
            jd = jaccard_sim(bs_list, can_bs_list)
            jaccd_max.append(jd)
            idx_list.append(idxx)
            if jd>0.5:
                can_bs_row.append(can_bs_list)
                can_j.append(jd)
                
        if len(can_bs_row)==0:
            can_idx = jaccd_max.index(np.max(jaccd_max))
            can_temp = total_ob_bs.iloc[can_idx,:].values
            can_bs_row.append(total_ob_bs.iloc[can_idx,:].values)
            match_c = con_tr_feature[(con_tr_feature['BSID']==int(can_temp[0])) & (con_tr_feature['BSID2']==int(can_temp[1]))
                          &(con_tr_feature['BSID3']==int(can_temp[2])) & (con_tr_feature['BSID4']==int(can_temp[3]))
                          &(con_tr_feature['BSID5']==int(can_temp[4])) & (con_tr_feature['BSID6']==int(can_temp[5]))]
            
            match_ss_c= match_c[(match_c['rss_level_1']== ss_list[0]) & (match_c['rss_level_2']== ss_list[1]) 
                            & (match_c['rss_level_3']==ss_list[2])& (match_c['rss_level_4']==ss_list[3]) 
                            & (match_c['rss_level_5']==ss_list[4]) & (match_c['rss_level_6']==ss_list[5])]
            if match_ss_c.shape[0]>0:
                zeros_list.append(float(match_ss_c[match_ss_c['conf']==0].shape[0])/float(zero_num))
            else:
                zeros_list.append(float(match_c[match_c['conf']==0].shape[0])/float(zero_num))
        
            j+=1 
        else:
            zeros_list.append(adaptive_emission_pro(can_j, match, bs_list, ss_list, can_bs_row, zero_num, 0))
        #print i, len(can_bs_row)
        i+=1
    else:
        j+=1
        match_ss= match[(match['rss_level_1']== int(ss_list[0])) & (match['rss_level_2']==int(ss_list[1])) & (match['rss_level_3']==ss_list[2])
             & (match['rss_level_4']==ss_list[3]) & (match['rss_level_5']==ss_list[4]) & (match['rss_level_6']==ss_list[5])]
        if match_ss.shape[0]>0:
            zeros_list.append(float(match_ss[match_ss['conf']==0].shape[0])/float(zero_num))
        else:
            zeros_list.append(float(match[match['conf']==0].shape[0])/float(zero_num))


# In[155]:


total_ob_te['conf_ad_em_pro_0'] = zeros_list 


# In[156]:


one_list = []
one_num = con_tr_feature[con_tr_feature['conf']==1].shape[0]
for idx, row in total_ob_te.iterrows():
    bs_list =row[['BSID','BSID2','BSID3','BSID4','BSID5','BSID6']].values
    ss_list = row[['rss_level_1','rss_level_2','rss_level_3','rss_level_4','rss_level_5','rss_level_6',]].values
    #print bs_list[0]
    match = con_tr_feature[(con_tr_feature['BSID']==int(bs_list[0])) & (con_tr_feature['BSID2']==int(bs_list[1]))
                          &(con_tr_feature['BSID3']==int(bs_list[2])) & (con_tr_feature['BSID4']==int(bs_list[3]))
                          &(con_tr_feature['BSID5']==int(bs_list[4])) & (con_tr_feature['BSID6']==int(bs_list[5]))]
    if match.shape[0]<5:
        can_bs_row = []
        can_j = []
        jaccd_max = []
        idx_list = []
        for idxx, roww in total_ob_bs.iterrows():
            can_bs_list = roww.values
            jd = jaccard_sim(bs_list, can_bs_list)
            jaccd_max.append(jd)
            idx_list.append(idxx)
            if jd>0.5:
                can_bs_row.append(can_bs_list)
                can_j.append(jd)
                
        if len(can_bs_row)==0:
            can_idx = jaccd_max.index(np.max(jaccd_max))
            can_temp = total_ob_bs.iloc[can_idx,:].values
            can_bs_row.append(total_ob_bs.iloc[can_idx,:].values)
            match_c = con_tr_feature[(con_tr_feature['BSID']==int(can_temp[0])) & (con_tr_feature['BSID2']==int(can_temp[1]))
                          &(con_tr_feature['BSID3']==int(can_temp[2])) & (con_tr_feature['BSID4']==int(can_temp[3]))
                          &(con_tr_feature['BSID5']==int(can_temp[4])) & (con_tr_feature['BSID6']==int(can_temp[5]))]
            
            match_ss_c= match_c[(match_c['rss_level_1']== ss_list[0]) & (match_c['rss_level_2']== ss_list[1]) 
                            & (match_c['rss_level_3']==ss_list[2])& (match_c['rss_level_4']==ss_list[3]) 
                            & (match_c['rss_level_5']==ss_list[4]) & (match_c['rss_level_6']==ss_list[5])]
            if match_ss_c.shape[0]>0:
                one_list.append(float(match_ss_c[match_ss_c['conf']==1].shape[0])/float(one_num))
            else:
                one_list.append(float(match_c[match_c['conf']==1].shape[0])/float(one_num))
        
        else:
            one_list.append(adaptive_emission_pro(can_j, match, bs_list, ss_list, can_bs_row, one_num, 1))
        
    else:
       
        match_ss= match[(match['rss_level_1']== int(ss_list[0])) & (match['rss_level_2']==int(ss_list[1])) & (match['rss_level_3']==ss_list[2])
             & (match['rss_level_4']==ss_list[3]) & (match['rss_level_5']==ss_list[4]) & (match['rss_level_6']==ss_list[5])]
        if match_ss.shape[0]>0:
            one_list.append(float(match_ss[match_ss['conf']==1].shape[0])/float(one_num))
        else:
            one_list.append(float(match[match['conf']==1].shape[0])/float(one_num))


# In[157]:


total_ob_te['conf_ad_em_pro_1'] = one_list 


# In[158]:


con_te_feature['gid'] = con_te_j
con_te_feature['p_gid'] = pred_con_te
con_tr_feature['gid'] = con_tr_j
con_tr_feature['p_gid'] = pred_con_tr


# In[159]:


trajs_tr = con_tr_feature.groupby(['TrajID'])


# In[160]:


time_list = []
st_mat = np.zeros((14,2,2)) #0-0 0-1 1-0, 1-1
for trajid, traj in trajs_tr:
    traj = traj.sort_values(by=['MRTime'],ascending=True)
    t_time = traj['MRTime'].values
    conf = traj['conf'].values
    for i in range(traj.shape[0]-1):
        time_list.append(compute_time_interval(t_time[i], t_time[i + 1]))
        idx = int(compute_time_interval(t_time[i], t_time[i + 1])/5)
        if idx >12:
            idx = 13
        if conf[i]==0 and conf[i+1]==0:
            st_mat[idx, 0, 0] +=1
        if conf[i] ==0 and conf[i+1] ==1:
            st_mat[idx, 0, 1] +=1
        if conf[i]==1 and conf[i+1]==0:
            st_mat[idx, 1, 0] +=1
        if conf[i] ==1 and conf[i+1] ==1:
            st_mat[idx, 1, 1] +=1


# In[161]:


trajs_te = con_te_feature.groupby(['TrajID'])


# In[162]:


pred_list_t_idx = []
#st_mat = np.zeros((14,2,2)) #0-0 0-1 1-0, 1-1
pred_list = []
p_g_list = []
t_g_list = []
for trajid, traj in trajs_te:
    traj = traj.sort_values(by=['MRTime'],ascending=True)
    t_time = traj['MRTime'].values
    conf = traj['conf'].values
    idx_list = []
    for i in range(traj.shape[0]-1):
        time_list.append(compute_time_interval(t_time[i], t_time[i + 1]))
        idx = int(compute_time_interval(t_time[i], t_time[i + 1])/5)
        if idx >12:
            idx = 13
        idx_list.append(idx)
    pred_list.append(traj[['BSID','rss_level_1','BSID2','rss_level_2',
               'BSID3','rss_level_3','BSID4','rss_level_4',
               'BSID5','rss_level_5','BSID6','rss_level_6',]].values)
        
    pred_list_t_idx.append(idx_list)
    p_g_list.append(traj['p_gid'].values)
    t_g_list.append(traj[['Longitude','Latitude']].values)


# In[163]:


init_prob = [float(con_tr_feature[con_tr_feature['conf']==0].shape[0])/con_tr_feature.shape[0],
             float(con_tr_feature[con_tr_feature['conf']==1].shape[0])/con_tr_feature.shape[0]]


# In[164]:


def hmm_viterbi(st_mat, init_prob, emit_prob, obs_seq, t_list):
    Nstate = 2
    Nobs = int(emit_prob.shape[0])
    T = len(obs_seq)
    
    partial_prob = np.zeros((Nstate,T))

    path = np.zeros((Nstate,T))

    for i in range(Nstate):
        partial_prob[i,0] = init_prob[i] * emit_prob[obs_seq[0], i]
        path[i,0] = i


    for t in range(1,T,1):
        newpath = np.zeros((Nstate,T))
        for i in range(Nstate):
            prob = -1.0
            for j in range(Nstate):
                nprob = partial_prob[j,t-1] * st_mat[t_list[t-1], j, i] * emit_prob[obs_seq[t], i]
                if nprob > prob:
                    prob = nprob
                    partial_prob[i,t] = nprob
                    newpath[i,0:t] = path[j,0:t]
                    newpath[i,t] = i
        path = newpath
    
    prob = -1.0
    j = 0
    for i in range(Nstate):
        if(partial_prob[i,T-1] > prob):
            prob = partial_prob[i,T-1]
            j = i

    return path[j,:]


# In[165]:


em_pro = total_ob_te
em_pro = em_pro.reset_index(drop = True)


# In[166]:


emit_prob = em_pro[['conf_ad_em_pro_0', 'conf_ad_em_pro_1']].values


# In[167]:


num_g = rg.n_grid
w = np.zeros((num_g, num_g))

test = zip(pred_con_tr, con_tr_j, error_tr)
tj_r = {}
for ss in test:
    #print ss
    #w[ss[0]][num_g]=w[ss[0]][num_g]+1
    #w[ss[0]][ss[1]]=w[ss[0]][ss[1]]+1
    if ss[0]!=ss[1] and ss[2]>50:
         w[ss[0]][ss[1]] +=1
        #w[ss[0], ss[1]] += 1
    if not tj_r.has_key(ss[0]):
        tj_r[ss[0]]=set()
    tj_r[ss[0]].add((ss[1]))

standard_point=[]


for item in tj_r:
    if (len(tj_r[item])==1) and (item in tj_r[item]):
        standard_point.append(item)

for idx in range(num_g):
    if not tj_r.has_key(idx):
        tj_r[idx]=set()
        tj_r[idx].add((idx))

g_list=rg.gridlist


# In[168]:


g_obs = {}
for idx, row in con_tr_feature.iterrows():
    obs = row[['BSID','rss_level_1','BSID2','rss_level_2',
               'BSID3','rss_level_3','BSID4','rss_level_4',
               'BSID5','rss_level_5','BSID6','rss_level_6',]].values
    gid = int(row['gid'])
    match = em_pro[(em_pro['BSID']==obs[0]) & (em_pro['rss_level_1']==obs[1]) & 
                      (em_pro['BSID2']==obs[2]) & (em_pro['rss_level_2']==obs[3]) & 
                      (em_pro['BSID3']==obs[4]) & (em_pro['rss_level_3']==obs[5]) & 
                      (em_pro['BSID4']==obs[6]) & (em_pro['rss_level_4']==obs[7]) & 
                      (em_pro['BSID5']==obs[8]) & (em_pro['rss_level_5']==obs[9]) & 
                      (em_pro['BSID6']==obs[10]) & (em_pro['rss_level_6']==obs[11])]
    if match.shape[0]>0:
        obs_idx = (gid, int(match.index[0]))
        if not g_obs.has_key(obs_idx):
            g_obs[obs_idx]=1
        else:
            g_obs[obs_idx] +=1


# In[170]:


def repair(te_pred, cof_list, o_list):
    i=0
    te_pred_n=[]
    while i < len(cof_list):
        if cof_list[i]==0:
            pred_temp = te_pred[i]
            if pred_temp in standard_point:
                te_pred_n.append(te_pred[i])
            else:
                repair_r = list(w[te_pred[i], :])
                max_idx = repair_r.index(np.max(repair_r))
                if g_obs.has_key((max_idx, o_list[i])):
                    te_pred_n.append(int(max_idx))
                    #print 'ok'
                else:
                    
                    te_pred_n.append(te_pred[i])
                
        else:
            te_pred_n.append(te_pred[i])
        i+=1
            
    #print len(te_pred_n)
    return te_pred_n


# In[172]:


import operator
error_list = []
error_new =[]
i=0
for row, time, raw, gt in zip (pred_list, pred_list_t_idx, p_g_list, t_g_list):

    o_seq = []
    for obs in row:
        #print obs
        match = em_pro[(em_pro['BSID']==obs[0]) & (em_pro['rss_level_1']==obs[1]) & 
                      (em_pro['BSID2']==obs[2]) & (em_pro['rss_level_2']==obs[3]) & 
                      (em_pro['BSID3']==obs[4]) & (em_pro['rss_level_3']==obs[5]) & 
                      (em_pro['BSID4']==obs[6]) & (em_pro['rss_level_4']==obs[7]) & 
                      (em_pro['BSID5']==obs[8]) & (em_pro['rss_level_5']==obs[9]) & 
                      (em_pro['BSID6']==obs[10]) & (em_pro['rss_level_6']==obs[11])]
        o_seq.append(match.index[0])
    pred = raw
    true = gt
    cof_list = hmm_viterbi(st_mat, init_prob, emit_prob, o_seq, time)
    r_pred = repair(pred, cof_list, o_seq)
    
   
    te_predp = np.array([rg.grid_center[idx] for idx in r_pred])
   
   
    error_tep = [distance(pt1, pt2) for pt1, pt2 in zip(te_predp, true)]
   
   
    for t in error_tep:
        #print t
        error_new.append(t)
    i+=1
    


# In[173]:


len(error_new)


# In[174]:


err= sorted(error_new)


# In[176]:


print np.median(err), np.mean(err), err[int(len(err) * 0.9)]

