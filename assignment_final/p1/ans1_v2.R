# Install required packages
# install.packages(c("keras", "tidyverse", "jpeg", "tools"))

# Load required libraries
library(keras)
library(tidyverse)

# Set the seed for reproducibility
set.seed(123)

# Define the paths to your data folders
train_path <- "data/train"
val_path <- "data/val"
test_path <- "data/test"

# Image dimensions
img_width <- 150
img_height <- 150

# Define the model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(img_width, img_height, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metrics = c("accuracy")
)

# Data preprocessing
datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_path,
  generator = datagen,
  target_size = c(img_width, img_height),
  class_mode = "binary",
  batch_size = 20
)

validation_generator <- flow_images_from_directory(
  val_path,
  generator = datagen,
  target_size = c(img_width, img_height),
  class_mode = "binary",
  batch_size = 20
)

# Train the model
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# Evaluate the model on the test set
test_generator <- flow_images_from_directory(
  test_path,
  generator = datagen,
  target_size = c(img_width, img_height),
  class_mode = "binary",
  batch_size = 1
)

model %>% evaluate_generator(test_generator, steps = 624)

# Save the model
save_model_hdf5(model, "lung_classification_model2.h5")





# Plot training and validation accuracy
plot(history$metrics$accuracy, type = "l", col = "blue", ylab = "Accuracy", xlab = "Epoch", main = "Training and Validation Accuracy")
lines(history$metrics$val_accuracy, col = "red")
legend("topright", legend = c("Training", "Validation"), col = c("blue", "red"), lty = 1)

# Save the plot as an image file
dev.copy(png, "accuracy_plot.png")
dev.off()

# Plot training and validation loss
plot(history$metrics$loss, type = "l", col = "blue", ylab = "Loss", xlab = "Epoch", main = "Training and Validation Loss")
lines(history$metrics$val_loss, col = "red")
legend("topright", legend = c("Training", "Validation"), col = c("blue", "red"), lty = 1)

# Save the plot as an image file
dev.copy(png, "loss_plot.png")
dev.off()

