# -*- coding: utf-8 -*-

def Apply(input_path, output_path, image_transform_function, allowOverwrite=False,supported_exts=['.jpg','.jpeg','.png']):
    """
    Parameters
    ----------
    input_path : str
        Folder which cointains dataset files (also in folders) to be transformed
    output_path : str
        Folder where to output transformed dataste files
    image_transform_function
        Callable function which accepts one image in input and returns the modified image
    allowOverwrite : bool
        Allow this function to overwrite a file in the output destination. (Default disabled)
    supported_exts : list
        Limit the kind of extension file for the dataset which to apply transform function (otherwise ignored)
    
    """
    
    import os
    from pathlib import Path
    from cv2 import imread, imwrite
    
    assert( input_path is not None)
    assert(output_path is not None)
    
    input_path = input_path.replace("\\","/")
    # ------------------ File gathering -----------------------

    files_to_process = []
    
    for root, subdirs, files in os.walk(input_path):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in supported_exts:
                files_to_process.append(os.path.join(root, filename))
                
    # ------------------ Path setting -------------------------
    relative_filepaths = []
    for file_to_process in files_to_process:
        filepath = Path(file_to_process)#.parents[0]
        
        relative_filepath = filepath.relative_to(input_path)
        
        
        relative_filepaths.append(relative_filepath)
    
    # ------------------ Transformation & Output --------------
    for i, file_to_process in enumerate(files_to_process):
        #print("%d"%i,end=";")
        relative_path = relative_filepaths[i].parents[0]
        # Ensure path existance
        Path(output_path / relative_path).mkdir(parents=True, exist_ok=True)
        
        my_file = Path(output_path / relative_filepaths[i])
        if my_file.is_file() and not allowOverwrite:
            # file exists
            continue
        
        input_image = imread(file_to_process)
        output_image = image_transform_function(input_image)
        
        imwrite((output_path / relative_filepaths[i]).__str__(), output_image)
        

if __name__ == "__main__":
    def lbp_transform(image_gray):
        import cv2
        import numpy as np
        from ImageProcessing.lbp import lbp_calculated_pixel
        
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        [height, width] = image_gray.shape[0:2]
        img_lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                 img_lbp[i, j] = lbp_calculated_pixel(image_gray, i, j)
            
        return cv2.cvtColor(img_lbp, cv2.COLOR_BGR2GRAY)
        
    def resize100_transform(image):
        import cv2
         
        # resize image
        return cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)

    import argparse
    parser = argparse.ArgumentParser(description="Face extractor from images")
    
    # ------------------ Input parsing  ------------------ 
    parser.add_argument('--inputpath', type=str,
                        help='Folder which contains images (even nested in folders)')
    
    parser.add_argument('--outputpath', type=str, 
                        help='Folder to save extracted faces')
    
    parser.add_argument('--lbp', action='store_true',
                default=False,
                dest='lbp',
                help='Apply Local Binary Pattern function')
    
    parser.add_argument('--resize100', action='store_true',
                default=False,
                dest='resize100',
                help='Apply Local Binary Pattern function')
    args = parser.parse_args()

    input_path  = args.inputpath
    output_path = args.outputpath
    lbp = args.lbp
    resize100 = args.resize100
    
    if input_path is None or output_path is None:
        raise SystemExit
    
    if resize100:
        Apply(input_path, output_path, image_transform_function=resize100_transform)
    if lbp:
        Apply(input_path, output_path, image_transform_function=lbp_transform)
    
    












