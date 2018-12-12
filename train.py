import decode_caption
import sys

def train(epoch):
    decoder = decode_caption.decoder()
    model = decoder.model_gen()
    batch_size = 512
    model.fit_generator(decoder.generator(batch_size=batch_size), steps_per_epoch=decoder.num_samples/batch_size, epochs=epoch, verbose=2, callbacks=None)
    model.save('Output/Model.h5', overwrite=True)
    model.save_weights('Output/Weights.h5',overwrite=True)
 
if __name__=="__main__":
    train(int(sys.argv[1]))
