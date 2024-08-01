import os

class CrossValidSetting:
    def __init__(self):
        self.name = 'colon'
        self.batch_size = 32  #  testing batch_size
        self.do_eval = False
        self.sample_time = 1
        self.sample_ratio = 1
        self.root = './proto'
        self.save_path = 'CELL_output'
        self.log_path = os.path.join(self.save_path,'log' )
        self.result_path = os.path.join(self.save_path, 'result')
        self.dataset = 'CELL'
        self.max_edge_distance = 180  #  CRC=150, LUAD=180
        self.max_num_nodes = 2081 # the maximum number of nodes in one graph

show_flag = False
image_txt="./txts/CRC_datasets.txt"
show_graph_txt="./txts/pt_CRC_new_test.txt"
picture_graph_save_dir = os.path.join('picture_graph', 'CRC_new_test_ResGCN14_dynamic')