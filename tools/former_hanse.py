import numpy as np
import cv2
def getGray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    equ = clahe.apply(img)
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    equ = cv2.LUT(equ,table)
    return equ

data_path = '05_test.tif'
data_img = cv2.imread(data_path)
data_img = cv2.resize(data_img,(512,512),interpolation=cv2.INTER_CUBIC)
data_img = getGray(data_img)
cv2.imshow('image',data_img)
cv2.imwrite('7.tif',data_img)
cv2.waitKey(0)
cv2.destroyAllWindows()