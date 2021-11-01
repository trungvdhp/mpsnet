import os
import pathlib
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle, class_weight

class DataLoader:
    
    def __init__(self, config):
        
        self.data_folder = config.data_folder
        self.image_extension = config.image_extension
        self.train_data = []
        self.train_labels = []
        curr_class = 0
        label = -1
        i=0
        print('Load training databases.')
        
        for folder in config.train_folders:
            folder=folder.strip()
            path = os.path.abspath(self.data_folder + folder)
            pattern = path + '**/*' + config.image_extension
            print(pattern)
            paths = [ os.path.abspath(path) for path in pathlib.Path(path).glob('**/*' + self.image_extension)]
            print('Num images:', len(paths))
            
            for path in paths:
                path = os.path.abspath(path)
                img = cv2.imread(path, 0)
                new_class = int(path.split('\\')[-2])

                if(new_class != curr_class):
                    curr_class = new_class
                    label += 1
                self.train_data.append(img_to_array(img)/255.0)
                self.train_labels.append(label)
                i+= 1
                if(i%1000==0):
                    print('Processed: ', i, end='\r')
        print('Total training images: ', i)
        self.train_data = np.asarray(self.train_data)
        self.train_labels = np.asarray(self.train_labels)
        self.sample_shape = self.train_data.shape[1:]
        self.n_train_classes = (self.train_labels[-1]-self.train_labels[0])+1
        
        self.folds = list(StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state).split(self.train_data, self.train_labels))
        
        self.class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.train_labels), y=self.train_labels)
        self.class_weights = {l:c for l,c in zip(np.unique(self.train_labels), self.class_weights)}
        
    def load_test_data(self, test_folder):
        self.test_data = []
        self.test_labels = []
        curr_class = 0
        label = -1
        i = 0  
        print('Load test databases.')
        
        for path in pathlib.Path(self.data_folder + test_folder).glob('**/*' + self.image_extension):
            path = os.path.abspath(path)
            img = cv2.imread(path, 0)
            new_class = int(path.split('\\')[-2])

            if(new_class != curr_class):
                curr_class = new_class
                label += 1
            self.test_data.append(img_to_array(img)/255.0)
            self.test_labels.append(label)
            i+=1
            
            if(i%100==0):
                print('Processed: ', i, end='\r')
        print('Total test images: ', i)
        self.test_data = np.asarray(self.test_data)
        self.test_labels = np.asarray(self.test_labels)
        self.test_name = test_folder
        self.n_test_samples = self.test_data.shape[0]
        self.n_test_classes = (self.test_labels[-1]-self.test_labels[0])+1
        self.test_class_size = self.n_test_samples//self.n_test_classes
       
    def get_fold_data(self, fold_id):
        train_ids, test_ids = self.folds[fold_id]
        return self.train_data[train_ids], self.train_labels[train_ids], self.train_data[test_ids], self.train_labels[test_ids]
        