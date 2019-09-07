# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from cv2 import imread
    from FaceRecognition.FaceRecognition import readFilesRecursively, faceRecognition, weights
    from FaceRecognition import KnownFaceClassifier
    import os
    # ------------------ CONSTANTS --------------------------
    
    INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
    
    TEST_IMAGE_PATH0 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\salvini\salvini003.jpg'
    TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\io\rugg001.png'
    TEST_IMAGE_PATH2 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\io\rugg001.png'
    TEST_IMAGE_PATH3 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\renzi\renzi001.jpg'
    
    test_path = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test'
    
    MAX_N_EIGENVECTORS = 55
    # ------------------ Actual code ------------------------
    input_path = INPUT_PATH
    
    images = readFilesRecursively(input_path)
    knownFaceClassifier = KnownFaceClassifier.MakeFromLbpModel(images,threshold=25)
    
    knownFaceClassifier.fit(3, 50, log=True)
    print("FIT!")
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH), already_processed=True)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH0), already_processed=True)
    print("is test a known face:", is_a_known_face)
    

    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH2), already_processed=False)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH3), already_processed=True)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(os.path.join(test_path,'renzi\\renzi002-1.jpg')), already_processed=True)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(os.path.join(test_path,'renzi\\renzi003.jpg')), already_processed=True)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(os.path.join(test_path,'salvini\\salvini001.jpg')), already_processed=True)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(os.path.join(test_path,'salvini\\salvini002.jpg')), already_processed=True)
    print("is test a known face:", is_a_known_face)