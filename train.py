import tensorflow as tf
import os
import json
import argparse 

# Get the output path from the Valohai machines environment variables
output_path = os.getenv('VH_OUTPUTS_DIR')

# A function to write JSON to our output logs with the epoch number with the loss and accuracy from each run.
def logMetadata(epoch, logs):
    print()
    print(json.dumps({
        'epoch': epoch,
        'loss': str(logs['loss']),
        'acc': str(logs['acc']),
    }))

def getArgs():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser()
    # Define two arguments that it should parse
    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Now run the parser that will return us the arguments and their values and store in our variable args
    args = parser.parse_args()

    # Return the parsed arguments
    return args

# Call our newly created getArgs() function and store the parsed arguments in a variable args. We can later access the values through it, for example args.learning_rate
args = getArgs()

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

metadataCallback = tf.keras.callbacks.LambdaCallback(on_epoch_end=logMetadata)

# Build the tf.keras.Sequential model by stacking layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Choose Adam as the optimizer and choose a loss function for training:
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=args.epoch_num, callbacks=[metadataCallback])

model.evaluate(x_test,  y_test, verbose=2)

# Save our file to that directory as model.h5
model.save(os.path.join(output_path, 'model.h5'))