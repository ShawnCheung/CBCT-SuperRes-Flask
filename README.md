# Cycle GAN super resolution model deployment using flask

## Web app
Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. Run:

`$ python3 webapp.py --port 5000`

then visit http://localhost:5000/ in your browser:


## Rest API
Simple rest API exposing the model for consumption by another service. Run:

`$ python3 restapi.py --port 5000`
