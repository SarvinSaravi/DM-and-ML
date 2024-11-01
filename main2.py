import os
import computing
import matplotlib.pyplot as plt
import time
import random
from PIL import Image as img

# list containing  all images
images = []
# path to dataset 2
directory = 'Dataset/flowers'

# set start timer
st = time.time()

for root, dirs, files in os.walk(directory):
    # randomly choose n file from every directories
    if len(files) > 0:
        files = random.sample(files, 100)

    # start process for each image
    for name in files:
        if name.endswith(".jpg"):
            images.append(name)

            # define path to read file
            path = root + '/' + str(name)
            # define path to save segmented image
            save_path = root + '/' + 'segmented_images'

            # clustering for segmented image
            transformed = computing.main_computing(path)
            print('end of processing for image : ', name)

            # show segmented image
            plt.imshow(transformed)
            plt.show()

            # save segmented image
            try:
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    print('create directory for save!')

                im = img.fromarray(transformed)
                im.save((save_path + '/' + name))
            finally:
                pass

            print()

total_time = time.time() - st
print('Total time of computing : ', total_time)

print('Total number of computing images : ', len(images))
