William Warner - 10889072

The language is Python3.9.21

Code Structure:

    face_detect.py - the script that runs the project
    train_model.py - the script used to train the emotion recognition model (done in CoLab)
    haarcascade_frontalface_default.xml - weights for a pretrained face detection model
    data/fer2013_test.csv - the testing data for recognizing emotions
    data/fer2013_test.csv - the training data for recognizing emotions

Instructions:

    1. Ensure you have python3.9
    2. Create a virtual environment: python3.9 -m venv <your_venv_name>
    3. Source the virtual environment: source <your_venv_name>/bin/activate
    4. Install required packages: pip install pandas numpy scikit-learn tensorflow opencv-python opencv-contrib-python
    5. Run the code: python3.9 face_detect.py
    6. Enjoy! You can close the window by pressing 'q'

It might be worth it to just try running the code in whatever development environment you 
currently have, but older versions of python/tensorflow are not able to open .keras files
