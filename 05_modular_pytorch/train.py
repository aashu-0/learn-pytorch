import os
import torch
import torchmetrics
import get_data, data_setup, engine, model_builder, utils
from torchvision import transforms

# hyperparams
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
NUM_WORKERS= os.cpu_count()

# directories
train_dir = get_data.image_path/'train'
test_dir = get_data.image_path/'test'

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
