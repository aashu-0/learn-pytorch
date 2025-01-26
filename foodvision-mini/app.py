
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# class_names
class_names = ['pizza', 'steak', 'sushi']


#create effnet model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes = len(class_names),
    seed=42
)

# load saved weights
effnetb2.load_state_dict(torch.load(
    f = '09_pretrained_effnet_feature_extractor_20_percent.pth',
    map_location=torch.device('cpu')
    )
)

# predict

def predict(img) -> Tuple[Dict, float]:

  # start timer
  start_timer = timer()

  # perform transform aadd batch dim
  img = effnetb2_transforms(img).unsqueeze(0)

  # modle on eval mode
  effnetb2.eval()
  with torch.inference_mode():

    # pass the img to the model and get pred prob
    pred_probs = torch.softmax(effnetb2(img), dim=1)

  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # pred time
  pred_time = round(timer() - start_timer, 5)

  # return pred dict and pred time
  return pred_labels_and_probs, pred_time

# gradio app

title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Made with ❤️ and 🍕"

# create examples list
example_list = [['examples/' +example] for example in os.listdir('examples')]

# gradio demo
demo = gr.Interface(fn= predict,
                    inputs = gr.Image(type= 'pil'),
                    outputs= [gr.Label(num_top_classes=3, label='Predictions'),
                              gr.Number(label="Predictions time(in seconds)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article
                    )

# launch
demo.launch()
