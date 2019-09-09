# -*- coding: utf-8 -*-

def _clean(input_path, limit_to_exts=['.jpg','.jpeg','.png']):
    """
    Parameters
    ----------
    input_path : str
        Folder which cointains dataset files (also in folders) to be DELETED
    limit_to_ext : list
        List of files types to DELETE if they can't be opened using OpenCV.
        
    DANGEROUS FUNCTION. USE WITH CAUTION.
    
    """
    
    import os
    from pathlib import Path
    from cv2 import imread, imwrite
    
    assert( input_path is not None)
    
    input_path = input_path.replace("\\","/")
    # ------------------ File gathering -----------------------

    files_to_process = []
    
    for root, subdirs, files in os.walk(input_path):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in limit_to_exts:
                files_to_process.append(os.path.join(root, filename))
      
    """          
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
    """
    for file in files_to_process:
        img = imread(file)
        if img is None:
            os.remove(file)
            print("[!!!!] Deleting file %s " % file)