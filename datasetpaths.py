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

# FACES94 

faces94 = os.path.join(dataset_mainfolder, 'faces94_mixed')
faces94_complement = os.path.join(dataset_mainfolder, 'faces94_complement')

faces94_own_classes = [Path(f.path).stem for f in os.scandir(faces94) if f.is_dir() ]
faces94_compl_classes = [Path(f.path).stem for f in os.scandir(faces94_complement) if f.is_dir() ]
faces94_ALL_classes = faces94_own_classes + faces94_compl_classes

# FACES95

faces95 = os.path.join(dataset_mainfolder, 'faces95')
faces95_complement = os.path.join(dataset_mainfolder, 'faces95_complement')

faces95_fe97 = os.path.join(dataset_mainfolder, 'faces95-fe97')
faces95_complement_fe97 = os.path.join(dataset_mainfolder, 'faces95_complement-fe97')

faces95_own_classes = [Path(f.path).stem for f in os.scandir(faces95) if f.is_dir() ]
faces95_complement_classes = [Path(f.path).stem for f in os.scandir(faces95_complement) if f.is_dir() ]
faces95_all_classes = faces95_own_classes + faces95_complement_classes

# FACES96
faces96 = os.path.join(dataset_mainfolder, 'faces96')
faces96_complement = os.path.join(dataset_mainfolder, 'faces96_complement')

faces96_own_classes = [Path(f.path).stem for f in os.scandir(faces96) if f.is_dir() ]
faces96_complement_classes =  [Path(f.path).stem for f in os.scandir(faces96_complement) if f.is_dir() ]
faces96_all_classes = faces96_own_classes + faces96_complement_classes

faces96_fe75  = os.path.join(dataset_mainfolder, 'faces96-fe75')
faces96_complement_fe75  = os.path.join(dataset_mainfolder, 'faces96_complement-fe75')

faces96_fe100  = os.path.join(dataset_mainfolder, 'faces96-fe100')
faces96_complement_fe100  = os.path.join(dataset_mainfolder, 'faces96_complement-fe100')

faces96_fe100lbp  = os.path.join(dataset_mainfolder, 'faces96-fe100lbp')
faces96_complement_fe100lbp  = os.path.join(dataset_mainfolder, 'faces96_complement-fe100lbp')


dirs_at_dir = [Path(f.path).stem for f in os.scandir(output100lbp_test) if f.is_dir() ]
output100_all_classes = output100_classes.copy()
output100_all_classes.extend(dirs_at_dir)
output100lbp_classes = list(set(output100_all_classes))

# POLIT

polit100 = os.path.join(dataset_mainfolder, 'polit100')
polit100_complement = os.path.join(dataset_mainfolder, 'polit100_complement')

polit100lbp = os.path.join(dataset_mainfolder, 'polit100lbp')
polit100lbp_complement = os.path.join(dataset_mainfolder, 'polit100_complement-lbp')

polit100blur_lbp = os.path.join(dataset_mainfolder, 'polit100blur_lbp')
polit100blur_lbp_complement = os.path.join(dataset_mainfolder, 'polit100_complement-blur-lbp')

polit100med_lbp = os.path.join(dataset_mainfolder, 'polit100med-lbp')
polit100med_lbp_complement = os.path.join(dataset_mainfolder, 'polit100_complement-med-lbp')

polit100_own_classes = [Path(f.path).stem for f in os.scandir(polit100) if f.is_dir() ]
polit100_complement_classes = [Path(f.path).stem for f in os.scandir(polit100_complement) if f.is_dir() ]
polit100_all_classes = polit100_own_classes + polit100_complement_classes





