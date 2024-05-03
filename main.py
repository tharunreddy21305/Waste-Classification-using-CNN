import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import os


cap = cv2.VideoCapture(0)
Classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)
new_width = 75
new_height = 75
imgArrow = cv2.resize(imgArrow, (new_width, new_height))
classIDBin = 0

imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED)
    # Resize the image to a smaller size, e.g., (width, height)
    img = cv2.resize(img, (120, 120))
    imgWasteList.append(img)


imgBinList = []
pathFolderBin = "Resources/Bins"
pathList = os.listdir(pathFolderBin)
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderBin, path), cv2.IMREAD_UNCHANGED)
    # Resize the image to a smaller size, e.g., (width, height)
    img = cv2.resize(img, (120, 120))
    imgBinList.append(img)


classDic = {0: None,
            1: 0,
            2: 1,
            3: 3,
            4: 0,
            5: 1,
            6: 2,
            7: 1,
            8: 2,
            9: 3,
            10:2,
            11:2,
            12:1,
            13:2,
            14:3,
            15:1,
            16:0,
            17:1,
            18:1,
            19:0,
            20:1}



while True:
    _, img = cap.read()
    imgResize = cv2.resize(img,(485,400))

    imgBackground = cv2.imread('Resources/latest bg (4).png')

    predection = Classifier.getPrediction(img)
    print(predection)
    classID = predection[1]

    if classID !=0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1],(749,100))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (775, 250))

        classIDBin = classDic[classID]
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinList[classIDBin], (749, 400))


    imgBackground[120:120+400,43:43+485] = imgResize
    #cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)