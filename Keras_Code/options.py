import argparse
import os
import time

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_folder', type=str, default='data/', help='The data folder which contains training and testing folders.')
        self.parser.add_argument('--train_folder_file', type=str, default='train_folders.txt', help='The file storing list of database folders for training.')
        self.parser.add_argument('--test_folder_file', type=str, default='test_folders.txt', help='The file storing list of test folders for testing.')
        self.parser.add_argument('--image_extension', type=str, default='.tiff', help='The extension of image files.')
        self.parser.add_argument('--train_session', type=int, default=1, help='The training session.')
        self.parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for kfold cross validation.')
        self.parser.add_argument('--weight_decay', type=float, default=0.0005, help='The weight decay.')
        self.parser.add_argument('--use_bias', type=int, default=0, help='Layers use bias or not.')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='The dropout rate of model.')
        self.parser.add_argument('--kernel_initializer', type=str, default='he_uniform', help='The kernel_initializer.')
        self.parser.add_argument('--warmup_batch_size', type=int, default=24, help='The warm_up batch size.')
        self.parser.add_argument('--fine_tune_batch_size', type=int, default=24, help='The fine tuning batch size')
        self.parser.add_argument('--warmup_epochs', type=int, default=50, help='The number of warmup epochs')
        self.parser.add_argument('--fine_tune_epochs', type=int, default=50, help='The number of fine tuning epochs')
        self.parser.add_argument('--warmup_optimizer', type=str, default='rmsprop', help='The Warmup optimizer')
        self.parser.add_argument('--fine_tune_optimizer', type=str, default='adam', help='The fine tuning optimizer')
        self.parser.add_argument('--warmup_lr', type=float, default=0.001, help='The warmup learning rate')
        self.parser.add_argument('--fine_tune_lr', type=float, default=0.0001, help='The fine tuning learning rate')
        self.parser.add_argument('--embedding_dim', type=int, default=128, help='The dimension of feature embedding')
        self.parser.add_argument('--embedding_layer_name', type=str, default='embeddings', help='The name of feature embedding layer')
        self.parser.add_argument('--model_name', type=str, default='mpsnet', help='The name of model need to train.')
        self.parser.add_argument('--start_fine_tune_layer_id', type=int, default=-5, help='The index of start fine tuning layer')
        self.parser.add_argument('--end_fine_tune_layer_id', type=int, default=-4, help='The index of end fine tuning layer')
        self.parser.add_argument('--distance_metric', type=str, default='cosine', help='The name of distance metric using for verification process')
        self.parser.add_argument('--random_state', type=int, default=42, help='Random state number')
       
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])
        
        with open(self.opt.data_folder + self.opt.train_folder_file,"r") as f:
            self.opt.train_folders = f.readlines()
       
        with open(self.opt.data_folder + self.opt.test_folder_file,"r") as f:
            self.opt.test_folders = f.readlines()

        self.opt.output_folder = 'results\\' + 'session_' + str(self.opt.train_session) + '\\' + self.opt.model_name + '\\'
        os.makedirs(self.opt.output_folder, exist_ok = True)
        
        return self.opt

    def __string__(self):
        args = vars(self.opt)
        doc = '------------ Options -------------\n'
        for k, v in sorted(args.items()):
            doc += f'{str(k)}: {str(v)}\n'
        doc += '-------------- End ----------------'