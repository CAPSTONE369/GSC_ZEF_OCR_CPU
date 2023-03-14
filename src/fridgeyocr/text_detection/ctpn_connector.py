"""ctpn_connector.py
1. TextProposalGraphBuilder
2. fit_y
3. TextProposalConnector
4. Graph
"""
import numpy as np
from typing import Tuple, List
import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clip_boxes(bboxes, image_size):
    ## 원본 이미지의 크기의 가로, 세로보다 넘치거나 0보다 작은 길이를 가질수도 있어서 그부분 예외 처리
    H, W = image_size
    zero = 0.0
    W_diff, H_diff = W - 1., H - 1.
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], W_diff), zero)
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], H_diff), zero)

    return bboxes

def fit_y(X, Y, x1, x2):
    """
    X: horizontal coordinate를 저장한 numpy array
    Y: vertical coordinate를 저장한 numpy array
    x1: horizontal coordinate of point 1
    x2: horizontal coordinate of point 2
    """
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


class Graph:
    def __init__(self, graph):
        self.graph = graph
    
    def sub_graphs_connected(self):
        sub_graphs = []
        for idx in range(self.graph.shape[0]):
            if not self.graph[:, idx].any() and self.graph[idx, :].any():
                v = idx
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs

""" Text Proposal Builder
- MIN_V_OVERLAP
    : 겹치는 최대의 iou영역을 의미한다. (이웃 anchor로 간주 할 수 있는)
- MIN_SIZE_SIM
    : 이 값은 side refinement를 보정할 때에 높이 길이의 비율을 의미한다.
- MAX_HORI_GAP
    : 역시나 side refinement 보정하는 부분에서 제일 멀리 떨어져 있을 수 있는 이웃으로 간주되는 픽셀 단위의 거리를 의미한다.
"""
class TextProposalGraphBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if isinstance(cfg, dict):
            self.MIN_V_OVERLAP=cfg['MIN_V_OVERLAP']
            self.MIN_SIZE_SIM=cfg['MIN_SIZE_SIM']
            self.MAX_HORI_GAP=cfg['MAX_HORI_GAP']
        else:
            self.MIN_V_OVERLAP=cfg.MIN_V_OVERLAP ## 0.7
            self.MIN_SIZE_SIM=cfg.MIN_SIZE_SIM ## 0.7
            self.MAX_HORI_GAP=cfg.MAX_HORI_GAP ## 50 
    
    def get_precursors(self, index):
        """ 현재 text box과 동일한 group에 존재하는 이전 text proposal을 구한다.
        - index: 현재 vertex의 아이디
        - returns: suitable text proposal의 인덱스를 저장한 리스트
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - self.MAX_HORI_GAP), 0) - 1, -1):
            neighbor_box_indices = self.boxes_table[left]
            for negighbor_box_index in neighbor_box_indices:
                if self.meet_v_iou(negighbor_box_index, index):
                    results.append(negighbor_box_index)

            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        presursors = self.get_precursors(succession_index) ## 같은 group에 존재하는 모든 right-side text proposal을 구함
        """
        text proposal이 right-side text proposal에 비해서 높거나 같은 점수를 갖는다면 True
        """
        if self.scores[index] >= np.max(self.scores[presursors]):
            return True

        return False

    def build_graph(self, text_proposals, scores, image_size):
        self.text_proposals = text_proposals ## numpy array
        self.scores = scores ## numpy array
        self.image_size = image_size
        ## text proposal이란 CTPN 모델이 입력 이미지를 기반으로 
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1


        boxes_tables = [[] for _ in range(self.image_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_tables[int(box[0])].append(index)
        self.boxes_table = boxes_tables
        
        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True
        G = Graph(graph)

        return G

    def get_successions(self, index):
        """ Find the text proposals that belongs to the same group of the current text proposal
        index: id of the current vertex
        Returns: list of integer that contains the suitable text proposals
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + self.MAX_HORI_GAP+1, self.image_size[1])):
            neighbor_box_indices = self.boxes_table[left]
            for negighbor_box_index in neighbor_box_indices:
                if self.meet_v_iou(negighbor_box_index, index):
                    results.append(negighbor_box_index)

            if len(results) != 0:
                return results

        return results

    def meet_v_iou(self, first_index, second_index):
        def vertical_overlap(first_index, second_index):
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            y0 = max(self.text_proposals[second_index][1], self.text_proposals[first_index][1])
            y1 = min(self.text_proposals[second_index][3], self.text_proposals[first_index][3])

            return max(0, y1-y0+1) / min(h1, h2)
        
        def size_similarity(first_index, second_index):
            h1 = self.heights[first_index]
            h2 = self.heights[second_index]
            return min(h1, h2) / max(h1, h2)
        
        return vertical_overlap(first_index, second_index) >= self.MIN_V_OVERLAP and \
            size_similarity(first_index, second_index) >= self.MIN_SIZE_SIM
    

class TextProposalConnector(object):
    def __init__(self, cfg):
        self.graph_builder = TextProposalGraphBuilder(cfg)
        self.cfg = cfg
        if isinstance(cfg, dict):
            self.refine = cfg['REFINEMENT']
        else:
            self.refine = cfg.REFINEMENT
    
    def group_text_proposals(self, text_proposals, scores, image_size):
        graph = self.graph_builder.build_graph(text_proposals=text_proposals, scores=scores,image_size=image_size)
        return graph.sub_graphs_connected()
    
    def get_text_lines(self, text_proposals, scores, image_size):
        tp_groups = self.group_text_proposals(text_proposals, scores, image_size)
        print(f"TP GROUPS: {len(tp_groups)}")
        text_lines = np.zeros((len(tp_groups), 4), dtype = np.float32)
        average_scores = [] ## 각각의 예측된 text box의 score은 곧 각각의 anchor text score의 평균이다.

        for idx, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            xmin = np.min(text_line_boxes[:, 0]) ## 전체 text line에서의 왼쪽 x축 좌표
            xmax = np.max(text_line_boxes[:, 2]) ## 전체 text line에서 오른쪽 x축 좌표

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) / 2.
            lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], xmin + offset, xmax - offset)
            lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], xmin + offset, xmax - offset)

            average_scores.append(scores[list(tp_indices)].sum() / float(len(tp_indices)))

            text_lines[idx, 0] = xmin
            text_lines[idx, 1] = min(lt_y, rt_y)
            text_lines[idx, 2] = xmax
            text_lines[idx, 3] = max(lb_y, rb_y)
        
        text_lines = clip_boxes(text_lines, image_size)
        average_scores = np.array(average_scores)

        return text_lines, average_scores
