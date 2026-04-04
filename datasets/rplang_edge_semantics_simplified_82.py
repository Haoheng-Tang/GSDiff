import os
from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import tiny_graph
from house_diffusion.utils import edges_to_coordinates, get_cycle_basis_and_semantic_3_semansimplified

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_82(Dataset):
    '''
    hould have storaged data without padding, augmentation and normalization in advance.
   data is graphs, (V, E), V has attributes(coords, ...), E has adjacency matrices.
   (although we use ordered data structure like ndarray, we only use the order in adjacency matrices(instead of adjacency lists)
   to facilitate the data loading. we don't use the order in nn, to meet permutation invariability of graph nodes.)
    '''
    def __init__(self, mode):
        '''(1)data reading. np.load()
           (2)data filtering(generate_corner_number in eval and not in train).
           (3)normalization(you could also do this in __getitem__() even after batch loading, but this will lead to longer time in training and sampling).
           purpose: make nn do not need to learn distributions scale shifting, more easier to converge.
           (4)padding and attn mask generating.

           but if dataset very big, memory can't stand, you should storage each one as a file and read it'''
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/train')
        elif self.mode == 'val':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/val')
        elif self.mode == 'test':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/test')
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)

    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):
        '''(1)get ndarray item by index.
          (2)random augmentation.
          return all unbatched things in ndarray in a dict'''

        if self.mode == 'train':
            graph = np.load('../datasets/rplang-v3-withsemantics/train/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load('../datasets/rplang-v3-withsemantics/val/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load('../datasets/rplang-v3-withsemantics/test/' + self.files[index], allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (53, 16)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
        # 初始化一个n*9的新数组(53, 9)
        corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
        # 复制第0、1列
        corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
        # 计算新的第2列
        corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
        # 计算新的第3列
        corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
        # 计算新的第4列
        corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
        # 复制第4、5、11、15列
        corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
        corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
        corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
        corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

        '''attn 1 matrix, (53, 53)'''
        global_attn_matrix = graph['global_matrix_np_padding'].astype(bool)
        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask']

        '''edges, (2809, 1)'''
        edges = graph['edges']

        '''edge_semantics, (2809, 7)'''
        # print(graph)
        corners_withsemantics_0_temp = corners_withsemantics_simplified[None, :, :].clip(-1, 1)
        corners_0_temp = (corners_withsemantics_0_temp[0, :, :2] * 128 + 128).astype(int)
        semantics_0_temp = corners_withsemantics_0_temp[0, :, 2:].astype(int)
        global_attn_matrix_temp = global_attn_matrix[None, :, :]
        corners_padding_mask_temp = corners_padding_mask[None, :, :]
        corners_0_temp_depadded = corners_0_temp[corners_padding_mask_temp.squeeze() == 1][None, :, :]  # (n, 2)
        semantics_0_temp_depadded = semantics_0_temp[corners_padding_mask_temp.squeeze() == 1][None, :, :]  # (n, 7)
        # print(edges.shape, edges[None, :, :].shape, global_attn_matrix_temp.shape, global_attn_matrix_temp.reshape(1, -1, 1).shape)
        # assert 0
        edges_temp_depadded = edges[None, :, :][global_attn_matrix_temp.reshape(1, -1, 1)][None, :, None]
        edges_temp_depadded = np.concatenate((1 - edges_temp_depadded, edges_temp_depadded), axis=2)

        ''' get planar cycles'''
        semantics_gt_i_transform_temp = np.where(semantics_0_temp_depadded == 1,
                                                np.indices(semantics_0_temp_depadded.shape)[-1], 99999)

        gt_i_points_temp = [tuple(corner_with_seman_val) for corner_with_seman_val in
                           np.concatenate((corners_0_temp_depadded, semantics_gt_i_transform_temp), axis=-1).tolist()[0]]

        gt_i_edges_temp = edges_to_coordinates(
            np.triu(edges_temp_depadded[0, :, 1].reshape(len(gt_i_points_temp), len(gt_i_points_temp))).reshape(-1),
            gt_i_points_temp)

        d_rev_temp, simple_cycles_temp, simple_cycles_semantics_temp = get_cycle_basis_and_semantic_3_semansimplified(
            gt_i_points_temp,
            gt_i_edges_temp)

        polygons = [[np.array([[p[0], p[1]] for p in polygon], dtype=np.int32)] for polygon in simple_cycles_temp]

        # 用于存储每条边界线及其两侧房间类型的列表
        edges_with_room_types = []

        # 遍历所有多边形和房间类型
        for i, polygon in enumerate(polygons):
            current_room_type = simple_cycles_semantics_temp[i]
            for points in polygon:
                # 遍历多边形的每条边界线
                for j in range(len(points) - 1):
                    # 当前边界线的起点和终点
                    start_point = tuple(points[j])
                    end_point = tuple(points[j + 1])
                    # print(start_point, end_point)
                    # 检查这条边界线的另一侧是否有相邻的房间
                    adjacent_room_type = None
                    for k, other_polygon in enumerate(polygons):
                        if k != i:  # 不检查同一个多边形
                            for other_points in other_polygon:
                                # 检查其他多边形的边界线
                                for l in range(len(other_points) - 1):
                                    other_start = tuple(other_points[l])
                                    other_end = tuple(other_points[l + 1])
                                    # 如果找到共享端点的边界线
                                    if start_point in (other_start, other_end) and end_point in (
                                    other_start, other_end):
                                        adjacent_room_type = simple_cycles_semantics_temp[k]
                                        break
                                if adjacent_room_type is not None:
                                    break
                        if adjacent_room_type is not None:
                            break
                    # 将边界线和两侧的房间类型添加到列表中
                    if ((start_point, end_point), current_room_type, adjacent_room_type) not in edges_with_room_types and \
                        ((end_point, start_point), current_room_type, adjacent_room_type) not in edges_with_room_types and \
                        ((start_point, end_point), adjacent_room_type, current_room_type) not in edges_with_room_types and \
                        ((end_point, start_point), adjacent_room_type, current_room_type) not in edges_with_room_types:
                        edges_with_room_types.append(((start_point, end_point), current_room_type, adjacent_room_type))
        # 替换 None 值并调整坐标
        normalized_edges_with_room_types = np.array([
            [
                (start[0] - 128) / 128,
                (start[1] - 128) / 128,
                (end[0] - 128) / 128,
                (end[1] - 128) / 128,
                6 if room_type is None else room_type,
                6 if adjacent_room_type is None else adjacent_room_type
            ]
            for ((start, end), room_type, adjacent_room_type) in edges_with_room_types
        ])

        # pass
        # 创建一个字典来映射节点坐标到它们的索引
        node_indices = {tuple(coord): idx for idx, coord in enumerate(corners_withsemantics_simplified[corners_padding_mask_temp.squeeze() == 1, :2])}

        # 初始化边的ndarray，所有元素设置为7
        edges_matrix = np.full((53, 53, 2), 7)

        # 遍历每条边，更新edges_matrix
        for edge in normalized_edges_with_room_types:
            # 端点坐标
            start_coord = (edge[0], edge[1])
            end_coord = (edge[2], edge[3])
            # 房间类型
            room_types = edge[4:]

            # 获取端点的索引
            start_idx = node_indices.get(start_coord)
            end_idx = node_indices.get(end_coord)

            # 如果两个端点的索引都在范围内，则更新edges_matrix
            if start_idx is not None and end_idx is not None and start_idx < 53 and end_idx < 53:
                edges_matrix[start_idx, end_idx, :] = room_types
                edges_matrix[end_idx, start_idx, :] = room_types  # 无向图，所以两个方向都要更新
        assert np.sum(np.triu(edges_matrix.sum(2), k=0) + np.tril(np.full((53, 53), 14), k=-1) < 14) == len(normalized_edges_with_room_types)

        # 创建一个形状为(53, 53, 8)的全零数组
        edges_seman_multi_hot = np.zeros((53, 53, 7), dtype=int)
        # 使用这些非7索引来填充multi-hot数组，这里我们利用了np.where来找到所有非7的位置
        indices = np.where(edges_matrix != 7)
        # 根据索引设置multi-hot数组，注意indices[2]代表值
        edges_seman_multi_hot[indices[0], indices[1], edges_matrix[indices]] = 1
        # edges_seman_multi_hot = edges_seman_multi_hot.reshape(2809, 7)

        # 判断multi-hot里面有阳台（5）的边数是不是4 ,
        # 如果是，满不满足其中三个是外部（6），一个既不是阳台也不是外部，如果满足，找出与内部相连的那条边的对面那条边的索引
        # 然后在corners_withsemantics_simplified里修改这两个点的坐标
        bal_hot = np.triu(edges_seman_multi_hot[:, :, 5])
        if bal_hot.sum() == 4:
            baledge_indices = np.where(bal_hot == 1)
            # print(baledge_indices)
            # 记录索引为6（外侧）的数量
            index_six_count = 0
            # 记录非6（内侧）的数量
            non_six_count = 0
            # 遍历数组的每一行
            index_six_count = []
            non_six_count = []
            for row_i, row in enumerate(edges_seman_multi_hot[baledge_indices]):
                # 找到除了索引为5之外值为1的列的索引
                for index, value in enumerate(row):
                    if value == 1 and index != 5:
                        if index == 6:
                            index_six_count.append(row_i)
                        else:
                            non_six_count.append(row_i)
            # 判断结果
            if len(index_six_count) == 3 and len(non_six_count) == 1:
                # 找出与内部相连的那条边的对面那条边的索引
                # 在corners_withsemantics_simplified里修改这两个点的坐标
                duimian = list(set(baledge_indices[0].tolist()+baledge_indices[1].tolist()).difference({baledge_indices[0][non_six_count[0]], baledge_indices[1][non_six_count[0]]}))
                c1 = corners_withsemantics_simplified[duimian[0]:duimian[0]+1, 0:2] * 128 + 128
                c2 = corners_withsemantics_simplified[duimian[1]:duimian[1]+1, 0:2] * 128 + 128
                p2 = ((c1 * 0.191 + c2 * 0.809) - 128) / 128
                p1 = ((c2 * 0.191 + c1 * 0.809) - 128) / 128
                corners_withsemantics_simplified[duimian[0]:duimian[0]+1, 0:2] = p1
                corners_withsemantics_simplified[duimian[1]:duimian[1]+1, 0:2] = p2
            else:
                pass
        
            
            
        


        return corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges