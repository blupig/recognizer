# Train
import data_input
import model

# Config
epochs = 100
workers = 6
data_path = 'tiny-imagenet-200'

# Build data generators
# Sample data for generators first
x_samples = data_input.load_sample_data(data_path)
print(x_samples[0])
train_gen, val_gen = data_input.data_generators(data_path, x_samples=x_samples)
for x_batch, y_batch in train_gen:
    print(x_batch[0])
    break


# Build and compile model
m = model.cnn_model()

# Train
m.fit_generator(train_gen, validation_data=val_gen, epochs=epochs, workers=workers)

# Save model
m.save('model.h5')
