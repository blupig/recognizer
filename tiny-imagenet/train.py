# Train
import data_input
import model

train_gen, val_gen = data_input.data_generators()
m = model.cnn_model()
m.fit_generator(train_gen, epochs=10, validation_data=val_gen)
m.save('model.h5')
