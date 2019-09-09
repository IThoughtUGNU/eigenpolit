# -*- coding: utf-8 -*-

if __name__ == "__main__":
    # raw      -> output      -> output100      -> output100lbp
    # raw-test -> output-test -> output100-test -> output100lbp-test
    
    from pathlib import Path
    import DatasetTransform
    from DatasetTransform import resize100_transform, lbp_transform, resize200_transform
    
    dataset_outputStart_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output'
    dataset_output100_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
    dataset_output100lbp_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
    
    dataset_output200_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200'
    dataset_output200lbp_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200lbp'
    
    dataset_output_testStart_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output-test'
    dataset_output100test_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test'
    dataset_output100lbp_test_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test'
    
    dataset_output200test_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200-test'
    dataset_output200lbp_test_folder = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200lbp-test'
    
    
    # To perform output      -> output100      -> output100lbp
    
    DatasetTransform.Apply(dataset_outputStart_folder, dataset_output100_folder, 
                           image_transform_function=resize100_transform)
    
    DatasetTransform.Apply(dataset_output100_folder, dataset_output100lbp_folder, 
                           image_transform_function=lbp_transform)
    
    # raw-test -> output-test -> output100-test -> output100lbp-test
    
    DatasetTransform.Apply(dataset_output_testStart_folder, dataset_output100test_folder, 
                           image_transform_function=resize100_transform)
    
    DatasetTransform.Apply(dataset_output100test_folder, dataset_output100lbp_test_folder, 
                           image_transform_function=lbp_transform)
    
    # 200 x 200 works too bad with current dataset (many pics low resoluted)
    """
    # --------------------------- 200 x 200 -----------------------------------
    
    # To perform output      -> output200      -> output200lbp
    
    DatasetTransform.Apply(dataset_outputStart_folder, dataset_output200_folder, 
                           image_transform_function=resize200_transform)
    
    DatasetTransform.Apply(dataset_output200_folder, dataset_output200lbp_folder, 
                           image_transform_function=lbp_transform)
    
    # raw-test -> output-test -> output100-test -> output100lbp-test
    
    DatasetTransform.Apply(dataset_output_testStart_folder, dataset_output200test_folder, 
                           image_transform_function=resize200_transform)
    
    DatasetTransform.Apply(dataset_output200test_folder, dataset_output200lbp_test_folder, 
                           image_transform_function=lbp_transform)
    """