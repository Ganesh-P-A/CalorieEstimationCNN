from flask import Flask, request, render_template
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.models import load_model

from keras.utils import load_img,img_to_array
import os

app = Flask(__name__)

# Load the model globally to avoid reloading with each request
model = load_model('FV.h5')

# ... (rest of the code with definitions for labels, fruits, vegetables,
# fetch_calories, and prepare_image)
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
     7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
     14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
     19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
     26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
     32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
     'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
       'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
       'Tomato', 'Turnip']


def fetch_calories(prediction):
  try:
    url = 'https://www.google.com/search?&q=calories in ' + prediction
    req = requests.get(url).text
    scrap = BeautifulSoup(req, 'html.parser')
    calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
    return calories
  except :
    return "unable to fetch calorie"

def prepare_image(img_path):
  # img = load_img(img_path, target_size=(224, 224, 3))
  # img = img_to_array(img)
  # img = img / 255
  # img = np.expand_dims(img, [0])
  img = Image.open(img_path)
  img = img.resize((224, 224))
  img = img_to_array(img)
  img = img / 255
  img = np.expand_dims(img, axis=0)
  answer = model.predict(img)
  y_class = answer.argmax(axis=-1)
  print("yclass",y_class)
  y = " ".join(str(x) for x in y_class)
  y = int(y)
  res = labels[y]
  print(res)
  return res.capitalize()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the uploaded image file
            img_file = request.files['image']
            if img_file:
                # Process the image and get the prediction
                result = prepare_image(img_file)

                # Fetch calorie information if available
                cal = fetch_calories(result)

                # Render the results template
                return render_template('result.html', prediction=result, category='Vegetables' if result in vegetables else 'Fruits', calories=cal)

        except Exception as e:
            return render_template('error.html', error_message=str(e))

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
