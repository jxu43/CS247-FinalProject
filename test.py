import encode_image as ei
import decode_caption 
import nltk
import numpy as np
import sys
from keras.preprocessing import sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_sentense():
	image_captions = open("Flickr8K_Text/Flickr8k.token.txt").read().split('\n')
	caption = {}
	for i in range(len(image_captions)-1):
		id_capt = image_captions[i].split("\t")
		id_capt[0] = id_capt[0][:len(id_capt[0])-2] 	# to rip off the #0,#1,#2,#3,#4 from the tokens file
		try:
			caption[id_capt[0]].append(id_capt[1])
		except:
			caption[id_capt[0]] = [id_capt[1]]
	return caption


def text(caption):
	encode = ei.model_gen()
	weight = 'Output/Weights.h5'
	decoder = decode_caption.decoder()
	model = decoder.model_gen(ret_model = True)
	model.load_weights(weight)
	bleu_sum = 0.0
	count = 0
	cc = SmoothingFunction()

	test_imgs_id = open("Flickr8K_Text/Flickr_8k.testImages.txt").read().split('\n')[:-1]
	for img_id in test_imgs_id:
		image_path = "Flickr8K_Data/" + img_id

		encoded_images = ei.encodings(encode, image_path)
		image_captions = generate_captions(decoder, model, encoded_images, beam_size=3)
		print (image_captions)

		# bleuscore-4
		bleus = []
		image_captions = image_captions.split()

		for true_sentense in caption[img_id]:
			true_sentense = true_sentense.split()
			try:
				bleu = sentence_bleu([true_sentense],image_captions, weights=(0.25,0.25,0.25,0.25), smoothing_function=cc.method4)
				bleus.append(bleu)
			except:
				pass
		if len(bleus) > 0:
			print (np.mean(bleus))
			bleu_sum += np.mean(bleus)
			count += 1

	print (bleu_sum / count)

def process_caption(dec, caption):
    tokens = caption.split()
    tokens = tokens[1:]
    terminate_id = tokens.index('<end>')
    tokens = tokens[:terminate_id]
    return " ".join([word for word in tokens])

def generate_captions(dec, model, encoded_images, beam_size):
    first_word = [dec.word2id['<start>']]
    prob_level = 0.0
    capt_seq = [[first_word, prob_level]]
    max_cap_length = dec.max_length
    while len(capt_seq[0][0]) < max_cap_length:
        temp_capt_seq = []
        for caption_id in capt_seq:
            iter_capt = sequence.pad_sequences([caption_id[0]], max_cap_length, padding = 'post')
            next_word_prob = model.predict([np.asarray([encoded_images]), np.asarray(iter_capt)])[0]
            next_word_ids = np.argsort(next_word_prob)[-beam_size:]
            for word_id in next_word_ids:
                new_iter_capt, new_iter_prob = caption_id[0][:], caption_id[1]
                new_iter_capt.append(word_id)
                new_iter_prob+=next_word_prob[word_id]
                temp_capt_seq.append([new_iter_capt,new_iter_prob])
        capt_seq = temp_capt_seq
        capt_seq.sort(key = lambda l:l[1])
        capt_seq = capt_seq[-beam_size:]
    best_caption = capt_seq[len(capt_seq)-1][0]
    best_caption = " ".join([dec.id2word[index] for index in best_caption])
    image_desc = process_caption(dec, best_caption)
    return image_desc


if __name__ == '__main__':
	caption = get_sentense()
	text(caption)
