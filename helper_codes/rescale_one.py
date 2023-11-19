import cv2

ratio = 0.25
A = cv2.imread('A.png')
heightA, widthA, channelsA = A.shape
heightA, widthA = int(heightA*ratio), int(widthA*ratio)
A = cv2.resize(A, (widthA, heightA), interpolation=cv2.INTER_AREA)
cv2.imwrite('Ar.png',A)

Ap = cv2.imread('Ap.png')
heightAp, widthAp, channelsAp = Ap.shape
heightAp, widthAp = int(heightAp*ratio), int(widthAp*ratio)
Ap = cv2.resize(Ap, (widthAp, heightAp), interpolation=cv2.INTER_AREA)
cv2.imwrite('Arp.png',Ap)