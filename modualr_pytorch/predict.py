
import torch
import torchvision
import argparse
import model_builder

# creating a parser
parser = argparse.ArgumentParser()

# get image path
parser.add_argument('--image',
                    help= 'Path directory of image to predict on')

# get model path
parser.add_argument('--model_path',
                    type=str,
                    default= 'models/test_modular_tinyvgg.pth',
                    help= 'Target Model filepath to use for prediction')

args = parser.parse_args()

# setup class names
class_names = ['pizza', 'steak', 'sushi']

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get the image path
IMG_PATH= args.image
print(f'[INFO] Predicting on {IMG_PATH}')

# function to load the model
def load_model(filepath = args.model_path):

  # same hyperparamas as saved model
  model= model_builder.TinyVGG(input_shape=3,
                               hidden_units = 96,
                               output_shape=3).to(device)

  print(f'[INFO] loading the saved model from: {filepath}')

  # load the saved model state_dict
  model.load_state_dict(torch.load(filepath))

  return model

# function to load in model and make prediction on the image
def predict_image(image_path= IMG_PATH,
                  filepath = args.model_path):
  
  # load the model
  model = load_model(filepath)

  # load the image and preprocess it
  image = torchvision.io.read_image(str(image_path)).type(torch.float32)/255

  # make transform -> resize the image
  transform = torchvision.transforms.Resize(size = (64,64))
  image = transform(image)

  # predict on image
  model.eval()
  with torch.inference_mode():
    image = image.to(device)

    # add batch_size dim and make pred
    pred_logits = model(image.unsqueeze(dim=0))

    # pred probs
    pred_probs = torch.softmax(pred_logits, dim=1)

    #pred labels
    pred_label = torch.argmax(pred_probs, dim=1)
    pred_label_class = class_names[pred_label]

  print(f'[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}')

if __name__ == '__main__':
  predict_image()
