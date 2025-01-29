# FacialExpressionDetector

**FacialExpressionDetector** is a Python project for detecting facial expressions using a pre-trained face detection model and a custom-trained emotion recognition model. The project utilizes OpenCV for face detection and TensorFlow for emotion recognition.

## Code Structure

- `face_detect.py` - The main script that runs the project.
- `train_model.py` - The script used to train the emotion recognition model (training is performed in Google Colab).
- `haarcascade_frontalface_default.xml` - Pre-trained face detection model weights (Haar Cascade Classifier).
- `data/fer2013_train.csv` - Training dataset for emotion recognition.
- `data/fer2013_test.csv` - Testing dataset for emotion recognition.

## Instructions

To use the **FacialExpressionDetector**, follow these steps:

1. Ensure you have Python 3.9 installed:
   - Check your Python version with `python3.9 --version`.
   - If Python 3.9 is not installed, download it from the [official Python website](https://www.python.org/).

2. Create a virtual environment:
   - Run `python3.9 -m venv <your_venv_name>` to set up a virtual environment.

3. Activate the virtual environment:
   - On Linux/MacOS: `source <your_venv_name>/bin/activate`
   - On Windows: `<your_venv_name>\Scripts\activate`

4. Install required packages:
   - Use the following command to install all necessary dependencies:
     ```bash
     pip install pandas numpy scikit-learn tensorflow opencv-python opencv-contrib-python
     ```

5. Run the main script:
   - Execute the program with `python3.9 face_detect.py`.

6. Enjoy real-time facial expression detection:
   - The program will display a window with live emotion detection. Close the window by pressing the `q` key.

## Notes

- It is recommended to use Python 3.9 and the specified dependencies, as older versions of Python or TensorFlow may not support loading `.keras` files.
- If you encounter issues, try running the code in your current development environment before setting up a new one.

Happy coding! 🎉
