import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_v2_l

model = efficientnet_v2_l(weights=None).to('cuda')
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=2,  # same number of output units as our number of classes
                    bias=True)).to('cuda')
# Load local weights
try:
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")
except RuntimeError as e:
    print(f"Error loading weights: {e}")

preprocess = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model.eval()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y - 20:y + h + 20, x - 20:x + w + 20]

    try:
        img_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)
        input_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor.to('cuda'))
            _, predicted = torch.max(output, 1)  # Get the index of the highest score

        class_names = ['non_smile', 'smile']
        class_label = class_names[predicted.item()]

        cv2.putText(frame, f'Class: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(class_label)

    except:
        pass

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
