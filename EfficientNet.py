import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import pandas as pd

train_dir = "/Data/hala.gamouh/cheese_dataset/train/simple_prompts"
test_dir = "/Data/hala.gamouh/cheese_dataset/val"
submission_test_dir = "/Data/hala.gamouh/cheese_dataset/test"  

base_model = EfficientNetB0(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(37, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_generator, epochs=1, validation_data=test_generator)

loss, accuracy = model.evaluate(test_generator)
print("Test Accuracy:", accuracy)
model.save('model.h5')


#Create Submission
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd


submission_datagen = ImageDataGenerator(rescale=1./255)

submission_generator = submission_datagen.flow_from_directory(
    submission_test_dir,
    target_size=(224, 224), 
    batch_size=32,
    class_mode=None,
    shuffle=False
)

model = load_model('model.h5')

predictions = model.predict(submission_generator, verbose=1)
predicted_classes = predictions.argmax(axis=1)

label_map = {v: k for k, v in train_generator.class_indices.items()} 
predicted_labels = [label_map[k] for k in predicted_classes]

filenames = submission_generator.filenames
ids = [os.path.basename(filename) for filename in filenames] 
results = pd.DataFrame({"id": ids, "label": predicted_labels})

results.to_csv("submission_EfficientNet.csv", index=False)
print("Submission file has been created and saved as 'submission_EfficientNet.csv'")
