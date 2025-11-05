# 2025CCF大数据与计算智能大赛-信也科技DGraph金融反欺诈图算法挑战赛数据集说明

2025CCF大数据与计算智能大赛-信也科技DGraph金融反欺诈图算法挑战赛初赛数据集存储在当前路径下的"phase1_gdata.npz"文件中。

数据集的npz文件中包含以下keys：
（以下说明中的N_node为用户节点的数目，N_edge为有向边的条数，用户的id对应用户特征x中的行数）
- **x**:
    节点特征，shape为(N_node,17)；
- **y**:
    节点共有(0,1,2,3)四类label，shape为(N_node,)，其中测试样本对应的label被标为-100；
- **edge_index**:
    有向边信息，shape为(N_edge,2)，其中每一行为(id_a, id_b)，代表用户id_a指向用户id_b的有向边；
- **edge_type**:
    边类型，shape为(N_edge,)；
- **edge_timestamp**：
    边连接日期，shape为(N_edge,)，其中边日期为从1开始的整数，单位为天；
- **train_mask**：
    包含训练样本id的一维array;
- **test_mask**：
    包含测试样本id的一维array;


