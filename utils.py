from PIL import Image
import random
from skimage.io import imread
import os
import numpy as np
from skimage.transform import resize


def generator_train_batch(batch_size, size, root_path):

	path_to_fold = os.path.join(root_path, "train_data")

	img_name_1 = "santa_rosa.png"
	img_name_2 = "ventura.png"

	mask_name_1 = img_name_1.split(".")[0] + ".tif"
	mask_name_2 = img_name_2.split(".")[0] + ".tif"

	img1 = os.path.join(path_to_fold, img_name_1)
	img2 = os.path.join(path_to_fold, img_name_2)

	img1_mask = os.path.join(path_to_fold, mask_name_1)
	img2_mask = os.path.join(path_to_fold, mask_name_2)

	im = Image.open(img1_mask)
	mask_np_1 = np.array(im)

	im = Image.open(img2_mask)
	mask_np_2 = np.array(im)


	im = Image.open(img1)
	imarray_1 = np.array(im)
	border_x = imarray_1.shape[1] - 513
	border_y = imarray_1.shape[0] - 512*2

	im = Image.open(img2)
	imarray_2 = np.array(im)
	border_x1 = imarray_2.shape[1] - 513
	border_y1 = imarray_2.shape[0] - 512*2

	while True:
		batch_x = np.zeros((batch_size,3,256,256))
		batch_y = np.zeros((batch_size,1,256,256))
		count = 0

		while count<batch_size:
			choise_img = random.randint(0,1)
			if choise_img == 0:
				rand_x = random.randint(0, border_x)
				rand_y = random.randint(0, border_y)

				im_write = imarray_1[rand_y:rand_y+512, rand_x:rand_x+512]
				label_write = mask_np_1[rand_y:rand_y+512, rand_x:rand_x+512]

				label_write = resize(label_write, size, mode='constant', anti_aliasing=False) > 0.5
				im_write = resize(im_write, size, mode='constant', anti_aliasing=True)

				#print(im_write.shape)
				batch_x[count,:] = np.rollaxis(im_write, 2)
				batch_y[count,:] = label_write

				count = count+1
			else:
				rand_x = random.randint(0, border_x1)
				rand_y = random.randint(0, border_y1)

				im_write = imarray_2[rand_y:rand_y+512, rand_x:rand_x+512]
				label_write1 = mask_np_2[rand_y:rand_y+512, rand_x:rand_x+512]

				label_write1 = resize(label_write1, size, mode='constant', anti_aliasing=False) > 0.5
				im_write = resize(im_write, size, mode='constant', anti_aliasing=True)

				batch_x[count,:] = np.rollaxis(im_write, 2)
				batch_y[count,:] = label_write1

				count = count+1

		yield np.array(batch_x, np.float32), np.array(batch_y, np.float32)
