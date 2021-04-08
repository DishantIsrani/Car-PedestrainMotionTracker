import cv2

#our image and video
image_file = 'project/images.jfif'
Video = cv2.VideoCapture('project/carTraffic2.mp4')
# Video = cv2.VideoCapture('project/mumbaiTraffic3.mp4')
# Video = cv2.VideoCapture('project/onlycar.mp4')

# Our pre-trained car and pedestrains classifier
# use the ml trained xml file
car_classifier_file = 'project/car.xml'
pedestrains_classifier_file = 'project/human.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrains_tracker = cv2.CascadeClassifier(pedestrains_classifier_file)


#iterate forever over frames 
while True:

    #read the current frame
    (read_successful, frame) = Video.read()

    if read_successful:
        # must convert to gray scale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break


    # DETECT CARS
    car_detector = car_tracker.detectMultiScale(grayscaled_frame, scaleFactor = 1.1, minNeighbors =2)

    # to print the coordinates of all the cars 
    # print(car_detector)

    # # DETECT PEDESTRAIN 
    pedestrains_detector = pedestrains_tracker.detectMultiScale(grayscaled_frame, scaleFactor = 1.1, minNeighbors =2)

    # DRAW RECTANGLE AROUND THE CARS
    for (x, y, w, h) in car_detector:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        
    for (x, y, w, h) in pedestrains_detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    #Display the image with the faces spotted
    cv2.imshow('Dishant Israni Car Detector', frame)

    #Dont autoclose (wait here in the code and listen for a keypress)
    cv2.waitKey(1)


# create opencv image
img = cv2.imread(image_file)


#convert to grayscale (needed for haar cascade)
# BGR is RGB backwards in open cv
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# DETECT THE CAR IN THE IMAGES
# detect multi scale will detect car of any size no matter what size it is 
car_detector = car_tracker.detectMultiScale(black_and_white)

# draw rectangle around the cars 
for(x, y, w, h) in car_detector:
    cv2.rectangle(img,(x,y), (x+w, y+h), (0, 0, 255), 2)


print("Code Completed")