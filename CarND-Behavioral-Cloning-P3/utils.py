import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


# Randomly decrease data having low steering angle
def randomly_drop_low_steering_data(data):
    
    index = data[abs(data['steer'])<.05].index.tolist()
    rows = [i for i in index if np.random.randint(10) < 8]
    data = data.drop(data.index[rows])
    print("Dropped %s rows with low steering"%(len(rows)))
    return data

#Returns croppped image
def preprocess_img(img):
    return img[60:135, : ]

#Change image brigtness
def img_change_brightness(img):
    # Convert the image to HSV
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = 0.25 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def trans_image(image, steer):
    """ Returns translated image and 
    corrsponding steering angle.
    """
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (320,75))
    return image_tr, steer_ang



#translate image and compensate for the translation on the steering angle
def translate_image(image, steering, horz_range=30, vert_range=5):
    rows, cols, chs = image.shape
    tx = np.random.randint(-horz_range, horz_range+1)
    ty = np.random.randint(-vert_range, vert_range+1)
    steering = steering + tx * 0.004 # multiply by steering angle units per pixel
    tr_M = np.float32([[1,0,tx], [0,1,ty]])
    image = cv2.warpAffine(image, tr_M, (cols,rows), borderMode=1)
    return image, steering


def read_image(image_path):
    image = mpimg.imread(image_path)
    return preprocess_img(image)

# Validation Generator
def get_images(data, data_path):

    while 1:
        for i in range(len(data)):
            img_path = data['center'][i].split('/')[-1].strip()
            img = read_image(os.path.join(data_path + '/IMG', img_path))
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steer'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang

def get_random_image_and_steering_angle(data, value, data_path):
    """ Returns randomly selected right, left or center images
    and their corrsponding steering angle.
    The probability to select center is twice of right or left. 
    """ 
    random = np.random.randint(4)
    if (random == 0):
        img_path = data['left'][value].split('/')[-1].strip()
        shift_ang = .25
    if (random == 1 or random == 3):
        img_path = data['center'][value].split('/')[-1].strip()
        shift_ang = 0.
    if (random == 2):
        img_path = data['right'][value].split('/')[-1].strip()
        shift_ang = -.25
    img = read_image(os.path.join(data_path + '/IMG', img_path))
    steer_ang = float(data['steer'][value]) + shift_ang
    return img, steer_ang


#Train data generator
def training_image_generator(data, batch_size, data_path):
   
    while 1:
        #Returns randomly sampled data from given pandas df  .
        batch = data.sample(n=batch_size)
        features = np.empty([batch_size, 75, 320, 3])
        labels = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            # Randomly select right, center or left image
            img, steer_ang = get_random_image_and_steering_angle(data, value, data_path)
            img = img.reshape(img.shape[0], img.shape[1], 3)          
            # Random Translation Jitter
            img, steer_ang = trans_image(img, steer_ang)
            # Randomly Flip Images
            random = np.random.randint(1)
            if (random == 0):
                img, steer_ang = np.fliplr(img), -steer_ang
            features[i] = img
            labels[i] = steer_ang
            yield np.array(features), np.array(labels)