import cv2

#our image and video
image_file = 'Car-PedestrianMotionTracker/images.jfif'
# image_file = 'Car-PedestrianMotionTracker/images (2).jfif'
# image_file = 'Car-PedestrianMotionTracker/images (3).jfif'
# image_file = 'Car-PedestrianMotionTracker/images (4).jfif'



# Our pre-trained car classifier
# use the ml trained xml file
classifier_file = 'Car-PedestrianMotionTracker/car.xml'


# create opencv image
img = cv2.imread(image_file)


# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)


#convert to grayscale (needed for haar cascade)
# BGR is RGB backwards in open cv
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# DETECT THE CAR IN THE IMAGES
# detect multi scale will detect car of any size no matter what size it is 
car_detector = car_tracker.detectMultiScale(black_and_white)

# draw rectangle around the cars 
for(x, y, w, h) in car_detector:
    cv2.rectangle(img,(x,y), (x+w, y+h), (0, 0, 255), 2)



#Display the image with the faces spotted
cv2.imshow('Dishant Israni Car Detector', img)

#Dont autoclose (wait here in the code and listen for a keypress)
cv2.waitKey()

print("Code Completed")
