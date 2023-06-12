import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
num_classes = 10
model.fc = nn.Linear(num_ftrs, num_classes)

PATH = "model.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

def transform_image(input_file):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    image = Image.open(input_file)
    t_img = data_transform(image)
    t_img.unsqueeze_(0)
    return t_img

def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction

def render_prediction(prediction_idx):   
    classes = [
    "Древнегреческая архитектура",
    "Древнеримская архитектура",
    "Модерн",
    "Барокко",
    "Готика",
    "Модернизм",
    "Классицизм",
    "Постмодернизм",
    "Ренессанс",
    "Романский стиль",
]
    class_name = classes[prediction_idx]
    return class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_name = render_prediction(prediction_idx)
            return render_template('result.html', class_name=class_name)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
