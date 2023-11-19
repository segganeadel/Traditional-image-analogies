import cv2 

def rescale_frame(frame_input, ratio=0.5):
    width = int(frame_input.shape[1] * ratio)
    height = int(frame_input.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

# Créer un objet de videoCapture
cap = cv2.VideoCapture('video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Boucle pour extraire les frames
i = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    # si on a récupérer la trame
    frame = rescale_frame(frame,0.25)
    cv2.imwrite('adel_frames\B'+str(i)+'.jpg',frame)
    i+=1
 
# libération 
cap.release()