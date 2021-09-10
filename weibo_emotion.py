import jieba
import collections
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from pandas.plotting import register_matplotlib_converters
import coordinate_conversion
import shapely, fiona 
import geopandas as gp
from shapely.geometry import Point 
import contextily as ctx 
import math
from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType
      

def get_emotionword(angryfile, sadfile, scaredfile, happyfile, disgustedfile):
    '''
    param: 5种情绪字典
    return: 5个情绪关键词的列表
    '''
    angry = []
    sad = []
    scared = []
    happy = []
    disgusted = []
    with open(angryfile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            angry.append(line.strip())
    with open(sadfile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            sad.append(line.strip())
    with open(scaredfile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            scared.append(line.strip())
    with open(happyfile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            happy.append(line.strip())
    with open(disgustedfile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            disgusted.append(line.strip())
    return angry, sad, scared, happy, disgusted


def get_vector(content, angry, sad, scared, happy, disgusted):
    '''
    param content: 已分词的一条微博
    param the left: 5种情绪关键词的列表
    return: 每条微博的情绪向量（1*5） 
    '''
    vector = [0]*5
    for word in content:
        if word in angry:
            vector[0] += 1
        elif word in sad:
            vector[1] += 1
        elif word in scared:
            vector[2] += 1
        elif word in happy:
            vector[3] += 1
        elif word in disgusted:
            vector[4] += 1
    return vector


def cluster_emotion(filepath, angry, sad, scared, happy, disgusted):
    '''
    对每条微博进行分类
    param: 微博文本、5种情绪关键词的列表
    return: 嵌套的列表，第一层是5种情绪，第二层是每种情绪包含的所有微博
    '''
    angryclass = []
    disgustedclass = []
    happyclass = []
    sadclass = []
    scaredclass = []
    complexclass = []
    nonemotionclass = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            seg_word = jieba.lcut(line.strip())
            vector = get_vector(seg_word, angry, sad, scared, happy, disgusted)
            vector = np.array(vector)
            pos = np.where(vector == vector.max())
            length = np.size(pos)
            if length == 1:
                if 0 in pos:
                    angryclass.append(line)
                elif 1 in pos:
                    sadclass.append(line)
                elif 2 in pos:
                    scaredclass.append(line)
                elif 3 in pos:
                    happyclass.append(line)
                elif 4 in pos:
                    disgustedclass.append(line)
            elif length == 5:
                nonemotionclass.append(line)
            else:
                complexclass.append(line)
    emotion_list = [angryclass, sadclass, scaredclass, happyclass,
    disgustedclass, complexclass, nonemotionclass]
    return emotion_list



# 用正则表达式取出x,y,时间,火星坐标转换
def get_data(emotion, emotion_type):
    '''
    param emotion：某类情绪的所有微博列表
    param emotion_type：情绪类型
    return: time:时间列表、df坐标和情绪类型的dataframe
    '''
    length = len(emotion)
    i = 0
    coordx = []
    coordy = []
    time = []

    model_coordx = re.compile(r'116\.\d+')
    model_coordy = re.compile(r'39\.8\d{1,6}|39\.7\d{1,6}')
    model_time = re.compile(r'\d\d:\d\d:\d\d')
    for i in range(length):
        content = emotion[i]
        x = model_coordx.findall(content)       
        y = model_coordy.findall(content)
        t = model_time.findall(content)
        coordx.append(x)
        coordy.append(y)
        time.append(t)
    coordx= list(itertools.chain.from_iterable(coordx))
    coordy= list(itertools.chain.from_iterable(coordy))
    # 火星坐标转换
    for i in range(0, len(coordx)):
        coordx[i],coordy[i] = coordinate_conversion.gcj02towgs84(coordx[i],coordy[i])
    dict = {'lnt':coordx, 'lat':coordy, 'emotion':emotion_type}
    df = pd.DataFrame(dict) 
    return df, time

# 统计每30分钟内的微博数量
def time_distribution(t):
    length = len(t)
    t = list(itertools.chain.from_iterable(t))
    t = pd.to_datetime(pd.Series(t),format='%H:%M:%S')
    diction = {'time':t}
    data = pd.DataFrame(diction,index=np.arange(length))    
    m = data.resample('30T',on='time').count()
    return m


# 空间分布
def space_distribution(e_df):
    '''
    param e_df: 5000*3的dataframe，列属性为经纬度和所属情绪
    '''
    geometry = [Point(float(x),float(y)) for x, y in zip(e_df.lnt, e_df.lat)]
    e_gdf = gp.GeoDataFrame(e_df, geometry=geometry)
    # 转crs:GPS EPSG:4326转为google map EPSG:3857
    e_gdf.crs = '+proj=longlat +datum=WGS84 +no_defs'
    e_gdf = e_gdf.to_crs( crs='+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs') 

    # 将北京市shp文件转为dataframe
    fpath = r'C:\Users\bin\Desktop\python\map\beijing_boundary\bj2t.shp'
    beijing_gdf = gp.GeoDataFrame.from_file(fpath, encoding='ANSI')
    beijing_gdf.crs = '+proj=longlat +datum=WGS84 +no_defs'
    beijing_gdf = beijing_gdf.to_crs( crs='+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs')

    e_color = {'happy':'red', 'angry':'orange', 'sad':'m', 'scared':'magenta', 'disgusted':'blue', 'complex':'lime', 'nonemotion':'aqua'} 
    # 数据点画在不同的图中
    for e in e_color.keys(): 
        e_gdf_temp = e_gdf.loc[lambda df: df['emotion'] == e]
        ax = e_gdf_temp.plot(color=e_color[e], alpha=0.5, linewidth=0.5, markersize=5, label=e) 
        print('Downloading map from Stamen') 
        ctx.add_basemap(ax, url=ctx.providers.Stamen.Toner, zoom=12) 
        ax.set_title(e+" "+'Geographic Distribution') 
        plt.show()

    # 所有数据点画在一张图中
    ax = e_gdf.plot(color='w', linewidth=0.5, markersize=5)
    i = 0
    for e in e_color.keys():
        e_gdf_temp = e_gdf.loc[lambda df:df['emotion']==e]
        e_gdf_temp.plot(ax=ax,color=e_color[e],alpha=0.5,linewidth=0.5,markersize=5,label=e)
        i += 1
    print('Downloading map from Stamen')
    ctx.add_basemap(ax,url=ctx.providers.Stamen.Toner,zoom=12)
    ax.set_title('Geographic Distribution')
    ax.legend(loc='best',fontsize='x-small')
    plt.show()



def geo_distribution(e_df):
    geo = Geo(init_opts={'width':2000}).add_schema(maptype='北京')
    emotion_type = ['angry','sad','scared','happy','disgusted']
    for e in emotion_type:
        temp = e_df[lambda df: df['emotion'] == e]
        points = []
        for n,i in enumerate(temp.index):
            geo.add_coordinate(e+"_"+str(n),temp.loc[i,'lnt'],temp.loc[i,'lat'])
            points.append((e+"_"+str(n),1))
        geo.add(e, points, symbol_size=10)
        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    geo.render('emotion_geo_distribution.html')


def main():
    # 5种情绪字典加入结巴分词
    jieba.load_userdict("angry.txt")
    jieba.load_userdict("disgusted.txt")
    jieba.load_userdict("happy.txt")
    jieba.load_userdict("sad.txt")
    jieba.load_userdict("scared.txt")
    # 读取5种情绪，便于后续构建向量
    angry, sad, scared, happy, disgusted = get_emotionword('angry.txt', 'sad.txt', 'scared.txt', 'happy.txt', 'disgusted.txt')
    # 对每行微博情绪文本进行分类
    emotions_list = cluster_emotion('weibo_test.txt', angry, sad, scared, happy, disgusted )
    
    #画时间分布图，在同一张图中
    plt.figure(figsize=(20,10))   
    type_list = ['angry','sad','scared','happy','disgusted','complex','nonemotion']
    i= 0
    e_df = pd.DataFrame(columns=['lnt','lat','emotion'])  # 获得经纬度的dataframe
    for emotion in emotions_list:
        df, t = get_data(emotion, type_list[i])
        print(type_list[i],len(t))
        e_df = e_df.append(df)    # 不同情绪的经纬度的dataframe合并成一个dataframe
        m = time_distribution(t)
        plt.plot(m,label=type_list[i])
        i+=1
    plt.legend()
    plt.show()

    # 画时间分布图，在不同的图中
    i = 0
    for emotion in emotions_list:
        df, t = get_data(emotion, type_list[i])    
        temp = time_distribution(t)
        plt.plot(temp)
        plt.title(type_list[i]+'time distribution')
        plt.show()
        i+=1
    
    # 空间分布，采用contextily方法
    space_distribution(e_df)
    # 空间分布图，采用pycharts.geo方法
    geo_distribution(e_df)


if __name__ == "__main__":
    main()