from werkzeug.wrappers import Request, Response
import io
import numpy
import cv2
import json
import tensorflow as tf

def read_input(request):
    # Ensure that we've received a file named 'image' through POST
    # If we have a valid request proceed, otherwise reeturn None
    if request.method == 'POST' and 'image' in request.files:
        photo = request.files['image']
        # Save file to memory
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        # Read the file bytes
        data = numpy.frombuffer(in_memory_file.getvalue(), dtype=numpy.uint8)
        # Use OpenCV to read read the image as grayscale
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 28x28
        img = cv2.resize(img, (28,28))
        return img
    return None

def mypredictor(environ, start_response):
    # Get the request object from the environment
    request = Request(environ)

    # Get the image file from our request
    inputfile = read_input(request)

    # If read_input didn't find a valid file
    if(inputfile is None) :
        response = Response("No image\n", content_type='text/html')
        return response(environ, start_response)

    # Load our model
    model_path = 'model.h5'
    new_model = tf.keras.models.load_model(model_path)

    # Use our model to predict the class of the file sent over a form.
    # We're reshaping the model as our model is expecting 3 dimensions (with the first one describing the number of images)
    prediction = new_model.predict_classes(inputfile.reshape(1,28,28))

    # Generate a JSON output with the prediction
    json_response = json.dumps("{Predicted_Digit: %s}" % prediction[0])

    # Send a response back with the prediction
    response = Response(json_response, content_type='application/json') 
    return response(environ, start_response)

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("localhost", 8000, mypredictor)