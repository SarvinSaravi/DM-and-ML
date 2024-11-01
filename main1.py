import os
import computing
import matplotlib.pyplot as plt
import time
from PIL import Image as img

# list contatining  all images
images = []
folder = 'Dataset/PASCAL'

st = time.time()

for filename in os.listdir(folder):
    images.append(filename)

    # define path to read file
    path = folder + '/' + str(filename)
    # define path to save segmented image
    save_path = folder + '/' + 'segmented_images'

    # clustering for segmented image
    transformed = computing.main_computing(path)
    print('end of processing for image : ', filename)

    # show segmented image
    plt.imshow(transformed)
    plt.show()

    # save segmented image
    try:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            print('create directory for save!')

        im = img.fromarray(transformed)
        im.save((save_path + '/' + filename))
    finally:
        pass

    print()

total_time = time.time() - st
print('Total time of computing : ', total_time)
