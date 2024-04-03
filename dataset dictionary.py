import numpy as np
import keras
from keras import layers
from keras import ops

# https://datascience.stackexchange.com/questions/71751/applying-a-keras-model-working-with-greyscale-images-to-rgb-images

num_categories = 11
num_materials, num_water_resistance, num_silhouettes = 6
num_warmth = 7

category_input = keras.Input(
    shape=(num_categories,), name="category"
)
material_input = keras.Input(
    shape=(num_materials,), name="material"
)  
water_resistance_input = keras.Input(
    shape=(num_water_resistance,), name="water_resistance"
)
silhouette_input = keras.Input(
    shape=(num_silhouettes,), name="silhouette"
)
warmth_input = keras.Input(
    shape=(num_warmth,), name="warmth"
)_input = keras.Input(
    shape=(num_water_resistance,), name="water_resistance"
)



dataset_dictionary = {
    'category': {
        0: 'hat',
        1: 'shirt',
        2: 'sweater',
        3: 'hoodie',
        4: 'coat',
        5: 'shorts',
        6: 'pants',
        7: 'skirt',
        8: 'socks',
        9: 'shoes',
        10: 'accessory', },

    'material': {
        0: 'cotton',
        1: 'wool',
        2: 'polyester',
        3: 'silk',
        4: 'linen',
        5: 'other' },
    
    'warmth': {
        0: 'not_applicable',
        1: 'for_hot_weather',
        2: 'for_warm_weather',
        3: 'versitile',
        4: 'for_cool_weather',
        5: 'for_cold_weather',
        6: 'for_arctic_weather' },

    'water_resistance': {
        0: 'not_applicable',
        1: 'no_resistance',
        2: 'mild_resistance',
        3: 'moderate_resistance',
        4: 'high_resistance',
        5: 'waterproof' },

    'silhouette': {
        0: 'not_applicable',
        1: 'skinny_fit',
        2: 'slim_fit',
        3: 'standard_fit',
        4: 'loose_fit',
        5: 'oversized_fit' }

}


dark_academia_color_palette = {
'#422d20', '#421220', '#5a3925,' '#81613b', '#774c3a',
'#271d20', '#b6aca8', '#2b271c', '#321f19', '#5a2827',
'#a8835d', '#241b24', '#5c343b', '#22151f', '#5b3630',
'#bdac86', '#54342b', '#3c3838', '#3d3232', '#202430',
'#8f5e5f', '#120f12', '#8d7458', '#704a2e', '#b69b7d',
'#2b2832', '#b89c8c', '#472f26', '#43473e', '#341f22'
}

def calculate_color_similarity(color, palette):
    # Example: Calculate color similarity using Euclidean distance
    distances = [np.linalg.norm(np.array(color) - np.array(palette_color)) for palette_color in palette]
    return np.mean(distances)

inputs = keras.Input(shape=(784,))

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])




num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs={"priority": priority_pred, "department": department_pred},
)

keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
