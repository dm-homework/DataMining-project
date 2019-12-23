# ### 一、数据分析

import json
import numpy as np
from queue import Queue
from collections import defaultdict
import pinyin.cedict
import difflib

#数据集路径
valid_row_data_path = 'C:/Users/xch/Desktop/DataMining/data/sna_data/sna_valid_author_raw.json'
valid_pub_data_path = 'C:/Users/xch/Desktop/DataMining/data/sna_data/sna_valid_pub.json'
# 合并数据
validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
for author in validate_data:
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]#输出的过程中发现还有中文姓名的共同作者
#对于validate_data中的每一个数据，设置validate_data[i] = 所有的发表的paper的集合

'''
作者名存在不一致的情况：
1、大小写
2、姓、名顺序不一致
3、下划线、横线
4、简写与不简写
5、姓名有三个字的表达: 名字是否分开

同理：机构的表达也存在不一致的情况
因此：需要对数据做相应的预处理统一表达
'''

import re
# 数据预处理

# 预处理名字
def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def get_equal_rate(str1, str2):
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


def check_contain_english(check_str):
    return bool(re.search('[a-z]', check_str)) or bool(re.search('[A-Z]', check_str))


def check_contain_space(check_str):
    for ch in check_str:
        if ch == ' ':
            return True
    return False


def precess_ch_name(name) :
    name = pinyin.get(name, format="strip", delimiter=" ")
    temp = name.split(' ')
    ch_temp =''
    for i in range(1,len(temp)):
        ch_temp += temp[i] 
    return ch_temp+' '+temp[0]


def precessname(name):
    if check_contain_chinese(name):
        if check_contain_english(name):
            if u'\u4e00' <= name[0] <= u'\u9fff':#开头是中文
                temp = name.split()
                name = temp[0]
                name = precess_ch_name(name)
            else :
                temp = name.split()
                name = temp[0]
                for i in range(1,len(temp)-1) :
                    name += ' ' + temp[i]
        else :#仅含中文
            if check_contain_space(name) :#包含空格
                name = pinyin.get(name, format="strip", delimiter=" ")
                name = name.replace('  ', ' ')
            else :
                name = precess_ch_name(name)
    #改完结果并没有变好嘻嘻嘻，果然是要从方法入手
    name = name.lower().replace(' ', '_')
    name = name.replace('.', '_')
    name = name.replace('-', '')
    name = re.sub(r"_{2,}", "_", name)
    return name


# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace(',',' ')
        org = org.replace('Sch.', 'School')
        org = org.replace('Dept.', 'Department')
        org = org.replace('Coll.', 'College')
        org = org.replace('Inst.', 'Institute')
        org = org.replace('Univ.', 'University')
        org = org.replace('Lab ', 'Laboratory ')
        org = org.replace('Lab.', 'Laboratory')
        org = org.replace('Natl.', 'National')
        org = org.replace('Comp.', 'Computer')
        org = org.replace('Sci.', 'Science')
        org = org.replace('Tech.', 'Technology')
        org = org.replace('Technol.', 'Technology')
        org = org.replace('Elec.', 'Electronic')
        org = org.replace('Engr.', 'Engineering')
        org = org.replace('Aca.', 'Academy')
        org = org.replace('Syst.', 'Systems')
        org = org.replace('Eng.', 'Engineering')
        org = org.replace('Res.', 'Research')
        org = org.replace('Appl.', 'Applied')
        org = org.replace('Chem.', 'Chemistry')
        org = org.replace('Prep.', 'Petrochemical')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Phys.', 'Physics')
        org = org.replace('Mech.', 'Mechanics')
        org = org.replace('Mat.', 'Material')
        org = org.replace('Cent.', 'Center')
        org = org.replace('Ctr.', 'Center')
        org = org.replace('Behav.', 'Behavior')
        org = org.replace('Atom.', 'Atomic')
        #org = org.split(';')[0]  # 可以去掉这一个注释，但是速度会变慢
        org = org.lower()
        # result = org.split(';')#多个机构全部存储在result数组中 然后返回
    return org


def get_org_same(org_from,org_to):
    from_ = org_from.split(";")
    to_ = org_to.split(";")
    from_=from_[:2]
    to_ = to_[:2]
    for i in range(len(from_)):
        for j in range(len(to_)):
            if from_[i] == to_[j] :
                return True
    return False


#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content


def get_similarity(list1,list2):
    for i in range(len(list1)):
        for j in range(len(list2)):
            if(list1[i]==list2[j]):
                return True
    return False


def get_count_similarity(list1,list2):
    count = 0
    for i in range(len(list1)):
        for j in range(len(list2)):
            if(list1[i]==list2[j]):
                count += 1
                break
    return count

# ### 二、解决方案：
from sklearn import metrics
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import math
#papers 中所包含的数据有 
# id(论文id)  title(文章标题)   abstract(简介)    keywords(关键词(一般多个))  authors(合作者(多个，包含name和org两个属性))  venue(发表的会议或者期刊)  year(年份)


## 基于图的人名消歧
def disambiguate_by_graph():
    res_dict = {}
    N_graph=[]
    for author in validate_data:
        count = 0
        papers = validate_data[author]
        res_dict[author] = []
        #打印作者和文章的数目
        print(author)
        n = len(papers)
        print(n)
        if len(papers) == 0:
            res_dict[author] = []
            N_graph.append(0)
            continue
        graph = [ [0] * len(papers) for i in range(len(papers))]  #i->j

        coauther_orgs = []
        paper_author = []
        paper_keywords = []
        paper_orgs = []
       
        for ipaper in range(len(papers)):
            paper_ = papers[ipaper]
            iauthor = paper_['authors']
            paper_author.append([precessname(paper_author['name']) for paper_author in iauthor])
            orgs=[]
            for _author in iauthor :
                if 'org' in _author:
                    orgs.append(preprocessorg(_author['org']))
                else:
                    orgs.append('')
            paper_orgs.append(orgs)
            # coauther_orgs.append(etl(' '.join(paper_author[ipaper] + orgs) + ' '+ abstract))
            coauther_orgs.append(etl(' '.join(paper_author[ipaper] + orgs)))
        
        graph = [ [0] * len(papers) for i in range(len(papers))]
        for i in range(n):
            flag = False
            print(i)
            for j in range(i,n):
                if i==j :
                    graph[i][j]==1
                    continue
                coauther_num = 0
                for iname in range(min(5,len(paper_author[i]))):
                    if flag :
                        break
                    for jname in range(min(5,len(paper_author[j]))) :
                        if flag :
                            break
                        if graph[i][j]!=0 and get_equal_rate(paper_author[i][iname],paper_author[j][jname]):
                            coauther_num += 1
                            if coauther_num > 2 or get_org_same(paper_orgs[i][iname],paper_orgs[j][jname]):
                                graph[i][j]=1
                                graph[j][i]=1
                                flag = True
        

                
        #广搜
        visit = n*[-1]
        queue = Queue()
        for start in range(n):#起始点start
            if visit[start] != -1:#start如果没有被访问过
                continue
            queue.put(start)
            while not queue.empty() :
                temp = queue.get()
                for sub in range(n):
                    if graph[temp][sub] == 1 and visit[sub] == -1:
                        queue.put(sub)
                        visit[sub] = count
            count += 1 
        N_graph.append(count)

    # N_graph_mean = np.mean(N_graph)#图算法求出的均值
    # print(18 *  "-")
    # print(N_graph)
    
    # i_author =0 #看看k
    # for author in validate_data:
    #     k = int(round(N_graph[i_author] / N_graph_mean * N[i_author] / 104)) + 1 #N算的有问题删了
    #     i_author += 1
    #     print(k)
    i_author = 0
    for author in validate_data:
        papers = validate_data[author]
        res_dict[author] = []
        #打印作者和文章的数目
        print(author)
        n = len(papers)
        if n == 0:
            res_dict[author] = []
            i_author+=1
            continue
        paper_dict = {}
        graph = [ [0] * len(papers) for i in range(len(papers))]  #i->j

        coauther_orgs = []
        paper_author = []
        paper_keywords = []
        paper_orgs = []
       
        for ipaper in range(len(papers)):
            paper_ = papers[ipaper]
            iauthor = paper_['authors']
            if 'keywords' in paper_:
                paper_keywords.append(paper_['keywords'])
            else:
                paper_keywords.append([])
            paper_author.append([precessname(paper_author['name']) for paper_author in iauthor])
            abstract = paper_["abstract"] if 'abstract' in paper_ else ''
            orgs=[]
            for _author in iauthor :
                if 'org' in _author:
                    orgs.append(preprocessorg(_author['org']))
                # else:
                #     orgs.append('')
            paper_orgs.append(orgs)
            coauther_orgs.append(etl(' '.join(paper_author[ipaper] + orgs) + ' '.join(paper_keywords[ipaper][:2])))
            # coauther_orgs.append(etl(' '.join(paper_author[ipaper] + orgs)))
        
        # k = int(round(N_graph[i_author] / N_graph_mean * N[i_author] / 104)) + 1 #10GM_ratio_keywords_abstract_0.3039 及之前用这个更新
        # k = max(int(round(N_graph[i_author] / N_graph_mean * N[i_author]/ 104)),1)
        k = max(round(n/9),1)#/104
        k = min(k,N_graph[i_author])
        print()#这个k应该没有多少问题 但是希望根据k分出的结果满足sigma = 37
        i_author+=1

        Gaussian = False
        if Gaussian:
            clf = GaussianMixture(n_components=k) 
        else :
            # clf = AgglomerativeClustering(n_clusters=k) 
            clf = MeanShift()
            # clf = KMeans(n_clusters=k)
            # clf = SpectralClustering(n_clusters=k, eigen_solver='arpack',affinity="nearest_neighbors")
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2,use_idf=True)#向量化
        X = vectorizer.fit_transform(coauther_orgs)
        svd = TruncatedSVD(10)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        if Gaussian:
            clf.fit(X)
            label_pre= clf.predict(X)
        else :
            label_pre = clf.fit_predict(X)

        for label, paper in zip(label_pre, papers):
            if str(label) not in paper_dict:
                paper_dict[str(label)] = [paper['id']]
            else:
                paper_dict[str(label)].append(paper['id']) 
        res_dict[author] = list(paper_dict.values())
        
    json.dump(res_dict, open('C:/Users/sdu/Desktop/DataMining/result/10GM_key2.json', 'w', encoding='utf-8'), indent=4)
disambiguate_by_graph()
