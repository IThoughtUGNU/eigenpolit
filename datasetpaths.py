# -*- coding: utf-8 -*-

# MY OWN DATASET

output100         = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
output100_test    = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test'
output100lbp      = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
output100lbp_test = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test'

output100_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test-strict'
output100_test_complement = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test-complement'

output100lbp_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test-strict'
output100lbp_test_complement = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test-complement'

output100_classes = ['berlusconi', 'calenda', 'conte', 'dimaio',
                         'gentiloni', 'mattarella', 'meloni', 'renzi', 'salvini',
                         'zingaretti','io','_baddetections']

output100strict_classes = ['berlusconi', 'calenda', 'conte', 'dimaio',
                     'gentiloni', 'mattarella', 'meloni', 'renzi', 'salvini',
                     'zingaretti']   
#FACES94 DATASET
import os
from pathlib import Path

dataset_mainfolder = r'C:\dev\python\nm4cs\eigenfaces\dataset'
faces94 = os.path.join(dataset_mainfolder, 'faces94_mixed')
faces94_complement = os.path.join(dataset_mainfolder, 'faces94_complement')

faces94_own_classes = [Path(f.path).stem for f in os.scandir(faces94) if f.is_dir() ]
faces94_compl_classes = [Path(f.path).stem for f in os.scandir(faces94_complement) if f.is_dir() ]
faces94_ALL_classes = faces94_own_classes + faces94_compl_classes


dirs_at_dir = [Path(f.path).stem for f in os.scandir(output100lbp_test) if f.is_dir() ]
output100_all_classes = output100_classes.copy()
output100_all_classes.extend(dirs_at_dir)
output100lbp_classes = list(set(output100_all_classes))

