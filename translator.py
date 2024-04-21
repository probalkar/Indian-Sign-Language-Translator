from keras.models import load_model
import cv2
import numpy as np

# Load the pre-trained model
model = load_model("model.h5")

# Define class labels
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + [chr(i) for i in range(65, 91)]

def extract_features(image):
    # Assuming the input image is already preprocessed and resized to 48x48
    feature = np.array(image)
    feature = feature.reshape(1,224,224,3)
    return feature/255.0

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2RGB)
    cropframe = cv2.resize(cropframe, (224, 224))
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe) 
    prediction_label = label[np.argmax(pred)]
    cv2.flip(frame, 1)
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred)*100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
