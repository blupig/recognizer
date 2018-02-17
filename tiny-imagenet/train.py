# Train
import data_input
import model

# Config
epochs = 50
workers = 3

# Build data generators
train_gen, val_gen = data_input.data_generators()

# Build and compile model
m = model.cnn_model()

# Train
m.fit_generator(train_gen, validation_data=val_gen, epochs=epochs, workers=workers)

# Save model
m.save('model.h5')
