import cv2


def sift_method(picture_address):
    # reading the image
    img = cv2.imread(picture_address)
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img, None)
    print(descriptors.shape)
    print(keypoints)

    # draw the detected key points
    sift_image = cv2.drawKeypoints(gray, keypoints, img)
    # show the image
    cv2.imshow('image', sift_image)
    # save the image
    # cv2.imwrite("table-sift.jpg", sift_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return descriptors


sift_method('Belha Ciao.jpg')
