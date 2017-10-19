import urllib.request
import urllib.parse
import json
import re
import os
from tqdm import tqdm
import argparse
dataset = ''
header={
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
     "referer":"https://image.baidu.com"
    }
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
def crawler(keyword, name):
    keyword=urllib.parse.quote(keyword,"utf-8")
    n = 0
    j = 0
    error = 0
    while n < 100:
        n+=30
        url1=url.format(word=keyword,pageNum=str(n))
        rep=urllib.request.Request(url1,headers=header)
        rep=urllib.request.urlopen(rep)
        try:
            html=rep.read().decode("utf-8")
        except:
            print("something wrong!")
            error=1
            print("-------------now page ="+str(n))
        if(error==1): continue
        #正则匹配，你需要的资源都是在 像这样的里面
        #"thumbURL":"https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=3734503404,179583637&fm=23&gp=0.jpg"
        p=re.compile("thumbURL.*?\.jpg")
        s=p.findall(html)
        if os.path.isdir("../data/%s/%s"%(dataset,name))!=True:
            os.makedirs('../data/%s/%s'%(dataset,name))
        for i in tqdm(s):
            i=i.replace("thumbURL\":\"","")
            try:
                urllib.request.urlretrieve(i,"../data/{ds}/{kw}/pic{num}.jpg".format(ds=dataset,kw=name,num=j))
                j+=1
            except:
                continue

if __name__ == '__main__':
    dataset = 'gaoqing'
    if os.path.isdir('../data/%s'%dataset) != True:
        os.makedirs('../data/%s'%dataset)
    subset = ['gaoqing']
    for idx, item in enumerate(subset):
        crawler(item, idx)

    """
    for idx, item in enumerate(['暹罗猫','布偶猫','苏格兰折耳猫','英国短毛猫','波斯猫','俄罗斯蓝猫','美国短毛猫',\
              '异国短毛猫','挪威森林猫','孟买猫','缅因猫','埃及猫','伯曼猫','斯芬克斯猫',\
             '缅甸猫','阿比西尼亚猫','新加坡猫','索马里猫','土耳其梵猫','中国狸花猫',\
             '美国短尾猫','西伯利亚森林猫','日本短尾猫','巴厘猫','土耳其安哥拉猫','褴褛猫',\
              '东奇尼猫','柯尼斯卷毛猫','马恩岛猫','奥西猫','沙特尔猫','德文卷毛猫','美国刚毛猫',\
             '呵叻猫','重点色短毛猫','哈瓦那棕猫','塞尔凯克卷毛猫','波米拉猫','拉邦猫','东方猫','美国卷毛猫','欧洲缅甸猫']):
        crawler(item, idx)
    """
