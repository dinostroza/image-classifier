import argparse
from utils import *

# init parse script arguments
parser = argparse.ArgumentParser(
    description='Flower Classifier'
)
# arguments settings
parser.add_argument('image_file', help='Image file (.jpg, .png)')
parser.add_argument('model_file', help='Model file in format HDF5 (.h5)')
parser.add_argument('--top_k', action="store", dest="top_k", default=1, type=int,
                    help='Number of best predictions to display')
parser.add_argument('--category_names', action="store", default='label_map.json', dest="category_names",
                    help='Json file with class names mapping ')
args = parser.parse_args()

# load scripts arguments (no checks for now)
image_file = args.image_file
model_file = args.model_file
top_k = args.top_k
category_names = args.category_names

# get image from path
image = get_image(image_file)

# format image for using in model
processed_image = process_image(image)

# load model h5
model = load_model(model_file)

# get probabilities using model
probs = get_probs(processed_image, model)

# get index of max prob values
top_k_prob_index = argtopmax(probs, top_k)

# get top k probabilities
top_k_probs = probs[top_k_prob_index]

# get class names mapping dict
class_names = get_classnames(category_names)

# get corresponding class names of top probabilities
top_k_classnames = list(map(lambda p: class_names[str(p)], top_k_prob_index))

# get the best prediction
best_prediction = top_k_classnames[0]

# get the best prediction probability
best_probability = top_k_probs[0]

# print results
if top_k > 1:
    print(f'The top {top_k} predictions are:')
    for k in range(top_k):
        print(f'- classname  : {top_k_classnames[k]} ')
        print(f'  probability: {100 * top_k_probs[k]}')
else:
    print(f'The best prediction is:')
    print(f'- classname  : {best_prediction} ')
    print(f'  probability: {100 * best_probability}')

