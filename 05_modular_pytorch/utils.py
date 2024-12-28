
import torch
from pathlib import Path

# func to save the model after training
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  # create target dir
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok= True)

  # create model save path
  assert model_name.endswith('.pth') or model_name.endswith('pt')
  model_save_path = target_dir_path /model_name

  # save model state_dict()
  print(f'Saving model to: {model_save_path}')
  torch.save(obj= model.state_dict(),
             f = model_save_path)
