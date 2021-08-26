#-*- encoding:utf-8 -*-
__date__ = '20/04/21'
'''
CV_INTER_NN - 最近邻插值,  
CV_INTER_LINEAR - 双线性插值 (缺省使用)  
CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..  
CV_INTER_CUBIC - 立方插值
'''
 
import os, codecs, json
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
 
match_data = []
file_sign_list = []
pq_size = 4
fea_size = 128

def takeSecond(elem):
    return elem[1]

def strToFloat(number):
    try:
        return float(number)
    except:
        return number

def linalg_norm(v_a, v_b):
    v_a = np.array(v_a)
    v_b = np.array(v_b)
    return np.sqrt(np.sum(np.square(v_b - v_a))).astype(np.float32)

def linalg_list(v_a, v_b):
    v_a = np.array(v_a)
    v_b = np.array(v_b)
    return v_b - v_a


def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in file_sign_list]
    return dict(zip(files, labels))
 
def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding = 'utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))


def second_cluster_centers(labels, cluster_centers, input_x, cluster_nums, randomState = None):
    '''
    计算原生向量和一级聚类中心的距离，生成残差向量，产出二级聚类中心
    '''
    all_pq = []
    #fr为原始特征，index为labels内的序号，all_pq为残差结果
    for index, fr in enumerate(input_x):
        cluster_index = cluster_centers[labels[index]]
        tmp_q = linalg_list(fr, cluster_index)
        all_pq.append(tmp_q)
    print("二级聚类中心特征提取完毕,长度:", len(all_pq))
    
    kmeans = MiniBatchKMeans(
        n_clusters = cluster_nums, 
        max_iter=15,
        n_init=1,
        init="k-means++",
        batch_size=10000,
        random_state = randomState).fit(all_pq)
    
    print("二级聚类完毕")

    all_pq = []
    for index, fr in enumerate(input_x):
        cluster_index = kmeans.cluster_centers_[kmeans.labels_[index]]
        tmp_q = linalg_list(fr, cluster_index)
        all_pq.append(tmp_q)

    print("二级聚类中心残差提取完毕")
    
    # 用原始向量和二级聚类中心的残差，进行PQ量化
    all_pq, all_kmeans_cluster = second_pq(all_pq, kmeans.labels_)

    # 返回二级聚类中心，all_pq为二级聚类中心量化结果
    return kmeans.labels_, kmeans.cluster_centers_, all_pq, all_kmeans_cluster

def second_pq(all_q, labels_2, randomState = None):
    '''
    二级聚类残差结果中心进行PQ量化，切4份
    '''
    all_list = dict()
    range_num = int(fea_size / pq_size)
    for i, fr in enumerate(all_q):
        for index in range(pq_size):
            if index not in all_list.keys():
                all_list[index] = []
            tmp_list = fr[index * range_num : ((index + 1) * range_num) - 1]
            # print(index * range_num, ((index + 1) * range_num) - 1)
            all_list[index].append(tmp_list)

    all_kmeans_cluster = []
    all_key = []
    for fr in all_list:
        fr = np.array(all_list[fr])
        kmeans = MiniBatchKMeans(
            n_clusters = 50, 
            max_iter=5,
            n_init=5,
            init="k-means++",
            batch_size=10000,
            random_state = randomState).fit(fr)
        all_kmeans_cluster.append(kmeans.cluster_centers_)
        all_key.append(kmeans.labels_)
    
    all_pq_list = dict()

    # 产出PQ聚类中心，计算每个样本所在聚类中心，进行编码
    for label_index, labels in enumerate(all_key):
        # num_index为第几位，分4分就是4位
        for num_index, num in enumerate(labels):
            #编码
            if num_index not in all_pq_list.keys():
                all_pq_list[num_index] = []
            all_pq_list[num_index].append(num)

    # all_pq_list为PQ码本，非对称距离计算时，此处还需要一个距离码本（样本到聚类中心的距离）
    for index, label in enumerate(labels_2):
        all_pq_list[index].append(labels_2[index])

    for index in all_pq_list:
        all_pq_list[index].append(file_sign_list[index])

    pq_list = dict()
    # key为二级聚类中心index
    for index, val in all_pq_list.items():
        if val[4] not in pq_list:
            pq_list[val[4]] = []
        pq_list[val[4]].append(val)

    return pq_list, all_kmeans_cluster

def knn_detect(file_list, cluster_nums, randomState = None):
    features = []
    num = 0
    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            sign = line[0]
            fr = line[1].strip().split(" ")
            if len(fr) != 128:
                continue
            fr = list(map(strToFloat,fr))
            features.append(fr)
            file_sign_list.append(sign)
            num = num + 1
            if num % 50000 == 0:
                print("当前进度:", num)
    input_x = np.array(features)
    print("一级聚类中心特征提取完毕,长度:", len(input_x))

    # 参数说明 https://www.pinggu.com/post/details/5ece258995fbaa64c2ed2475
    # 200w数据 max_iter=30 需要1个小时左右
    kmeans = MiniBatchKMeans(
        n_clusters = cluster_nums, 
        max_iter=15,
        n_init=1,
        init="k-means++",
        batch_size=10000,
        random_state = randomState).fit(input_x)

    print("一级聚类完毕")
    # score = metrics.calinski_harabaz_score(X, y_pred)
    # print("score,", score)
    return kmeans.labels_, kmeans.cluster_centers_, input_x
 

def search(search_data, cluster_centers, cluster_centers2, all_kmeans_cluster, all_pq):
    '''
    计算一级聚类中心，计算残差，计算二级聚类中心，计算PQ编码
    all_kmeans_cluster为PQ聚类中心，需要先计算
    all_pq为每条数据的量化结果[key:二级聚类中心，value为PQ编码]
    '''
    min_q = 0
    min_index = -1
    for index, centers in enumerate(cluster_centers):
        q = linalg_norm(search_data, centers)
        if min_q == 0:
            min_q = q
            min_index = index
        if q < min_q:
            min_q = q
            min_index = index
        #print("1距离:", index, q)
    
    index_1 = min_index
    print("一级聚类中心:", min_index, "打分:", min_q)

    #残差计算
    tmp_q = linalg_list(search_data, cluster_centers[min_index])

    #残差所在的二级聚类中心
    min_q = 0
    min_index = -1
    for index, centers in enumerate(cluster_centers2):
        q = linalg_norm(tmp_q, centers)
        if min_q == 0:
            min_q = q
            min_index = index
        if q < min_q:
            min_q = q
            min_index = index
        # print("2距离:", index, q)

    index_2 = min_index
    print("二级聚类中心:", min_index, "打分:", min_q)


    # 原始数据跟二级聚类中心做残差，取PQ结果
    tmp_pq_list = []
    q_2 = linalg_list(search_data, cluster_centers2[index_2])

    range_num = int(fea_size / pq_size)
    for index in range(pq_size):
        # PQ的前32维数据
        tmp_list = q_2[index * range_num : ((index + 1) * range_num) - 1]
        # print(index * range_num, ((index + 1) * range_num) - 1)

        min_q = 0
        min_index = -1
        # 取PQ结果
        cluster_list = all_kmeans_cluster[index]
        for index, cluster in enumerate(cluster_list):
            q = linalg_norm(tmp_list, cluster)
            if min_q == 0:
                min_q = q
                min_index = index
            if q < min_q:
                min_q = q
                min_index = index
        tmp_pq_list.append(min_index)
    print('二级聚类中心:', index_2, ' PQ编码:', tmp_pq_list)

    sort_list = []
    for index, pq in enumerate(all_pq[index_2]):
        tmp_list_pq = pq[0:4]

        # 这个距离公式可以变换
        # q = linalg_list(tmp_pq_list, tmp_list_pq)
        q = linalg_norm(tmp_pq_list, tmp_list_pq)
        sort_list.append([all_pq[index_2][index][5], q])
    sort_list.sort(key=takeSecond)
    
    print('----------')
    print("查询结果前十:")
    for val in sort_list[0:10]:
        print(val)

def main():
    filenames = "data/part-00086"
    print('----------')
    print("开始构建码本")
    labels, cluster_centers, input_x = knn_detect(filenames, 100)
    labels2, cluster_centers2, all_pq, all_kmeans_cluster = second_cluster_centers(labels, cluster_centers, input_x, 50)
    print('----------')
    print("开始查询:", file_sign_list[998])
    search(input_x[998], cluster_centers, cluster_centers2, all_kmeans_cluster, all_pq)
    res_dict = res_fit(filenames, labels)
    save('./', 'knn_res1.txt', res_dict)

    res_dict = res_fit(filenames, labels2)
    save('./', 'knn_res2.txt', res_dict)

 
if __name__ == "__main__":
    main()
