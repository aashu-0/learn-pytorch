import os
import argparse
import torch
import torchmetrics
import get_data, data_setup, engine, model_builder, utils
from torchvision import transforms

# create a parser
parser = argparse.ArgumentParser(description= 'Get some hyperparameters')

# get an arg for num of epochs
parser.add_argument('--num_epochs',
                    default = 10,
                    type=int,
                    help='The number of epochs to train for')

# get an arg for batch size
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='The number of samples to train per batch')

# get an arg for hidden units
parser.add_argument('--hidden_units',
                    default=10,
                    type=int,
                    help='The number of hidden units in hidden layers')

# get an arg for learning rate
parser.add_argument('--learning_rate',
                    default=0.003,
                    type=float,
                    help='Learning rate to train the model')

# get an arg for train directory
parser.add_argument('--train_dir',
                    default=get_data.image_path/'train',
                    type=str,
                    help='Path of file directory to train the model')

# get an arg for test directory
parser.add_argument('--test_dir',
                    default=get_data.image_path/'test',
                    type=str,
                    help='Path of file directory to test the model')

# get an argument from the parser
args = parser.parse_args()

# hyperparams
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
NUM_WORKERS= os.cpu_count()


# directories
train_dir = args.train_dir
test_dir = args.test_dir

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transforms
data_transform = transforms.Compose([transforms.Resize((64,64)),
                                     transforms.ToTensor()])

# dataloaders from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = data_transform,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS
)

# model from mode_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units= HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)

# loss, optimizer and accuracy
loss_fn = torch.nn.CrossEntropyLoss()
accuracy_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr= LEARNING_RATE)

# training using engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             accuracy_fn= accuracy_fn,
             optimizer=optimizer,
             epochs = NUM_EPOCHS,
             device = device)

# save model using utils.py
utils.save_model(model=model,
                 target_dir='models',
                 model_name='test_modular_tinyvgg.pth')
