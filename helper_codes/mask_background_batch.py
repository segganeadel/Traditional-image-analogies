from glob import glob
from natsort import natsorted
import cv2
import numpy as np

frames = natsorted(glob('adel_frames/*.*'))
images = [cv2.imread(img) for img in frames]

segmented_frames = natsorted(glob('segmentedframes/*.*'))
segmented_images = [cv2.imread(img) for img in segmented_frames]

masks = [np.where(np.any(img!=0, axis=2)) for img in segmented_images]

for i,(Ap,A,mask) in enumerate(zip(images,segmented_images,masks)):
    Ap[mask] = A[mask]
    cv2.imwrite('withbg\B'+str(i)+'.jpg',Ap)
