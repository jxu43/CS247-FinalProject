import numpy as np
from keras import layers
from keras import models
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation
from keras.preprocessing import image, sequence
import pickle
import sys

EMBEDDING_DIM = 128

class decoder():
	def __init__(self):
		self.captions = None
		self.img_id = None
		self.vocab_size = None
		self.num_samples = None
		self.max_length = 0
		self.word2id = {}
		self.id2word = {}
		self.image_encodings = pickle.load(open("image_encodings.p", "rb"))
		# if the code above does not work, replace with following one
		# self.image_encodings = pickle.load(open("image_encodings.p", "rb"), encoding='latin1')
		self.parse()

	def parse(self):
		self.captions = []
		self.img_id = []
		self.num_samples = 0
		tokens, vocab = [], []
		with open('Flickr8K_Text/trainimgs.txt', 'r') as train_imgs:
			train_data = train_imgs.read().split('\n')
			for line in train_data:
				if not line: continue
				line = line.split('\t')
				self.img_id.append(line[0])
				self.captions.append(line[1])
				caption_length = len(line[1].split())
				self.max_length = max(self.max_length, caption_length)

		for caption in self.captions:
		    self.num_samples += len(caption.split())-1
		    vocab.extend(caption.split())
		print (len(vocab))
		vocab = list(set(vocab))
		self.vocab_size = len(vocab)
		self.word2id = {w: i for i, w in enumerate(list(vocab))}
		self.id2word = {i: w for i, w in enumerate(list(vocab))}

	def generator(self, batch_size):
		
	    num_steps = int(self.num_samples/batch_size)
	    cur, cap_index = 0, 0
	    features_img,features_cap,labels = [],[],[]

	    while True:
	    	for i in range(num_steps):
	    		features_img = []
	    		features_cap = []
	    		labels = []
	    		for j in range(batch_size):
	    			caption = self.captions[cur]
	    			features_cap.append([self.word2id[word] for word in caption.split()[:cap_index]])
	    			label = np.zeros(self.vocab_size)
	    			label[self.word2id[caption.split()[cap_index]]] = 1
	    			labels.append(label)
	    			features_img.append(self.image_encodings[self.img_id[cur]])
	    			if cap_index + 1 > len(caption.split()) - 1:
	    				cap_index = 0
	    				cur = 0 if cur >= len(self.captions) - 1 else cur + 1
	    			else:
	    				cap_index += 1
	    		labels = np.asarray(labels)
	    		features_img = np.asarray(features_img)
	    		features_cap = sequence.pad_sequences(features_cap, maxlen=self.max_length, padding='post')
	    		features = [features_img, features_cap]
	    		yield [features, labels]


	def model_gen(self, ret_model = False):
	       
		input_image = layers.Input(shape=(4096,))
		image_model = layers.Dense(256,activation='relu',name="ImageFeature")(input_image)
		image_model = RepeatVector(self.max_length)(image_model)

		input_txt = layers.Input(shape=(self.max_length,))
		text_model = layers.Embedding(self.vocab_size, EMBEDDING_DIM, mask_zero=True)(input_txt)
		decoder = layers.concatenate([text_model,image_model])
		decoder = layers.LSTM(1000,name="CaptionFeature",return_sequences=False)(decoder)
		output = layers.Dense(self.vocab_size,activation='softmax')(decoder)
		model = models.Model(inputs=[input_image, input_txt],outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		print ("Model created!")
		
		return model
