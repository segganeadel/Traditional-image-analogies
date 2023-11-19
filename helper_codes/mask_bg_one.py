import cv2
import numpy as np



A = cv2.imread('Ar.png')
Ap = cv2.imread('Arp.png')
mask = np.where(np.any(A!=0, axis=2))
Ap[mask] = A[mask]
cv2.imwrite('Arbg.png',Ap)