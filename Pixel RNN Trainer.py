import MNISTClassifier as classifier
import MNISTGen as gen

'''
  The purpose of this file is to train multiple different networks back to back
without human intervention, to be run on the main workstation...
'''

gen.train_generation(epochs = 1, batch_size = 100, model_name = 'Gen TEST.pt')
gen.generator()

# classifier.train_pixel_rnn(epochs = 1, model_name = "Test.pt")
# classifier.evaluate(train_network = None, dataset = None, filepath = 'Test.pt')
# for _ in range(10):
#     classifier.manual_evaluation(filepath = 'Test.pt')