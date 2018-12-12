import numpy as np
import pickle
import progressbar
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input	

TOKEN_PATH = "Flickr8K_Text/Flickr8k.token.txt"
TRAIN_PATH = "Flickr8K_Text/Flickr_8k.trainImages.txt"
TRAIN_WRITE_PATH = "Flickr8K_Text/trainimgs.txt"
TEST_PATH = "Flickr8K_Text/Flickr_8k.testImages.txt"
TEST_WRITE_PATH = "Flickr8K_Text/testimgs.txt"

def tokenize():
	image_captions = open(TOKEN_PATH).read().split('\n')
	caption = {}
	for i in range(len(image_captions)-1):
		id_capt = image_captions[i].split("\t")
		id_capt[0] = id_capt[0][:len(id_capt[0])-2]
		try:
			caption[id_capt[0]].append(id_capt[1])
		except:
			caption[id_capt[0]] = [id_capt[1]]
	return caption

def preprocess_data(caption, file_path, write_path):
	train_imgs_id = open(file_path).read().split('\n')[:-1]
	train_imgs_captions = open(write_path,'wb')
	for img_id in train_imgs_id:
		for captions in caption[img_id]:
			desc = "<start> "+captions+" <end>"
			train_imgs_captions.write(img_id+"\t"+desc+"\n")
			train_imgs_captions.flush()
	train_imgs_captions.close()

def model_gen():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	return model

def encodings(model, path):
	processed_image = image.load_img(path, target_size=(224,224))
	processed_image = image.img_to_array(processed_image)
	processed_image = np.expand_dims(processed_image, axis=0)
	ready_image = np.asarray(preprocess_input(processed_image))
	prediction = model.predict(ready_image)
	prediction = np.reshape(prediction, prediction.shape[1])
	return prediction

def encode_image():
	model = model_gen()
	image_encodings = {}
	images = open(TRAIN_PATH).read().split('\n')[:-1]

	# visualize progress
	bar = progressbar.ProgressBar(maxval=len(images), \
    		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

	# start encoding
	print("Encoding images")
	bar.start()
	process_progess = 1
	for img in images:
		path = "Flickr8K_Data/"+str(img)
		image_encodings[img] = encodings(model, path)
		bar.update(process_progess)
		process_progess += 1
	bar.finish()

	# dump the encoding file
	with open( "image_encodings.p", "wb" ) as pickle_f:
		pickle.dump(image_encodings, pickle_f)
	print("Finishing Encoding")

if __name__=="__main__":
	tokens = tokenize()
	print("Start data preprocess")
	preprocess_data(tokens, TRAIN_PATH, TRAIN_WRITE_PATH)
	preprocess_data(tokens, TEST_PATH, TEST_WRITE_PATH)
	print("Finish data preprocess")

	print("Start encoding")
	encode_image()
	print("Finish encoding")