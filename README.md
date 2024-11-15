# Smile Detection Project

This project implements a real-time smile detection system using a fine-tuned `EfficientNet V2-L model`. The workflow includes fine-tuning `EfficientNet V2-L` in `detection_model.ipynb`, where the trained weights are saved as `model.pth`, and using these weights in `smile_detection.py` for live webcam detection.


## Features

* *Fine-Tuning*: Train the smile detection model using detection_model.ipynb on your custom dataset.
* *Real-Time Smile Detection*: Run the fine-tuned model on webcam feeds for live smile classification.
* *EfficientNet Architecture*: Uses a modified EfficientNet for accurate and efficient smile detection.

## File Structure

```
├── detection_model.ipynb       # Notebook for fine-tuning the model  
├── model.pth                   # Fine-tuned model weights (generated by the notebook)  
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection  
├── smile_detection.py          # Real-time detection script  
├── requirements.txt            # Python dependencies  
├── README.md                   # Project documentation  
```


## How to Use

### 1. Install Dependencies

Install the required Python packages using:

```
pip install -r requirements.txt  
```


### 2. Fine-Tune the Model

1. Open `detection_model.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the steps in the notebook to:
  * Load and preprocess your dataset.
  * Fine-tune the EfficientNet-based model for smile detection.
  * Save the trained model as `model.pth`.

### 3. Run Smile Detection

After fine-tuning the model, run the real-time detection script:

```
python smile_detection.py  
```

### 4. Exit the Webcam Feed

Press `q` to quit the webcam feed.


## Notes

* Ensure the `model.pth` file is present in the project directory before running `smile_detection.py`.
* Modify the Haar cascade or preprocessing parameters if needed for your specific use case.
* The project supports custom datasets for training in `detection_model.ipynb`.


## Acknowledgments

* OpenCV for face detection.
* PyTorch and torchvision for model fine-tuning and inference.
