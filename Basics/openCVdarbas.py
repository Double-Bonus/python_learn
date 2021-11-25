import cv2
 # B G R, kartais buna dar alfa

#img = cv2.imread(r'D:\\Backup\\Desktop\\I\\KTU\\7Semestras\\Intelektika\\Praktika\\flower.png',cv2.IMREAD_COLOR)
img = cv2.imread(r'flower.png',cv2.IMREAD_COLOR)

langovardas = 'img'

cv2.namedWindow(langovardas, cv2.WINDOW_FREERATIO)
cv2.imshow(langovardas, img)
cv2.waitKey(1000)
print(img.shape)
print(img.size)

roi = img[100:400, 100:400,:]
cv2.imshow(langovardas, roi)

cv2.waitKey()

cv2.destroyAllWindows()

cam = cv2.VideoCapture(0)
cv2.namedWindow(langovardas, cv2.WINDOW_FREERATIO)
while True:
    ret, frame = cam.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray,50,50)

        r = frame[:,:,2]
        b = frame[:,:,0]
        g = frame[:,:,1]
        frame[:,:,1] = r
        frame[:,:,0] = g

        cv2.imshow(langovardas, frame)
    if cv2.waitKey(30) == ord('q'): #ord pakeicia i asci is raides!
        break
cam.release()
cv2.destroyAllWindows()
