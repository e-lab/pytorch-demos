import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# way to get intermediate output of the network:
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Define the model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
# print(model)
model.layer1 = torch.nn.Identity()
model.layer2 = torch.nn.Identity()
model.layer3 = torch.nn.Identity()
model.layer4 = torch.nn.Identity()
model.avgpool = torch.nn.Identity()
model.fc = torch.nn.Identity()
# print(model)
model.maxpool.register_forward_hook(get_activation('maxpool'))

# Define the preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((640, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    # Capture two frames from the webcam
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    # Preprocess the frame
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame = preprocess(frame).unsqueeze(0)

    # Run the model on the batch
    with torch.no_grad():
        output = model(frame)

    maxpool_out = activation['maxpool']
    size = maxpool_out.shape

    # Generate the saliency map
    saliency_image = maxpool_out.sum(1).squeeze(0).unsqueeze(2).expand(size[2],size[3],3)
    mframe1 = cv2.resize(frame1, (size[3],size[2]))
    mframe2 = cv2.resize(frame2, (size[3],size[2]))
    saliency_motion = (mframe2 - mframe1)

    # Compute and Normalize the saliency map
    saliency_image = (saliency_image / torch.max(saliency_image)).numpy()
    # print(saliency_image.min(), saliency_image.max())
    saliency_motion = saliency_motion / np.max(saliency_motion)
    saliency_map = (saliency_image + saliency_motion)/2

    # saliency_map = cv2.resize(saliency_map, (512,512)) # for large demo screen

    # Display the saliency map
    cv2.imshow('Saliency Map', saliency_map)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy all windows
cv2.destroyAllWindows()