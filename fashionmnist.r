mg_rows <- img_cols <- 28
x_train <- as.matrix(train.data[, 2:dim(train.data)[2]])
y_train <- as.matrix(train.data[, 1])
dim(x_train) <- c(nrow(x_train), img_rows, img_cols, 1)
x_test <- as.matrix(test.data[, 2:dim(train.data)[2]])
y_test <- as.matrix(test.data[, 1])
dim(x_test) <- c(nrow(x_test), img_rows, img_cols, 1)
clothes.labels <-c( "T-shirt/top", "Trouser", "Pullover","Dress", "Coat", "Sandal", "Shirt", "Sneaker","Bag", "Ankle boot")
rotate <- function(x) t(apply(x, 2, rev))
plot_image <- function(x, title = "", title.color = "black") {
dim(x) <- c(img_rows, img_cols)
image(rotate(rotate(x)), axes = FALSE,
col = grey(seq(0, 1, length = 256)),
main = list(title, col = title.color))
}
par(mfrow=c(4, 4), mar=c(0, 0.2, 1, 0.2))
for (i in 1:16) {
nr <- i * 10 plot_image(x_train[nr, , , 1],
clothes.labels[as.numeric(train.data[nr, 1] + 1)])
}
batch_size <- 128
num_classes <- 10
epochs <- 13
input_shape <- c(img_rows, img_cols, 1)
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
model <- keras_model_sequential()
model %>%
layer_conv_2d(filters = 32, kernel_size = c(5,5), activation= 'relu',
input_shape = input_shape) %>%
layer_conv_2d(filters = 64, kernel_size = c(3,3), activation= 'relu') %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_dropout(rate = 0.25) %>%
layer_flatten() %>%
layer_dense(units = 128, activation = 'relu') %>%
layer_dropout(rate = 0.5) %>%
layer_dense(units = num_classes, activation = 'softmax')
model %>% compile(
loss = loss_categorical_crossentropy,
optimizer = optimizer_adadelta(),
metrics = c('accuracy')
)
model %>% fit(
x_train, y_train,
batch_size = batch_size,
epochs = epochs, verbose = 1, validation_data = list(x_test, y_test))
scores <- model %>% evaluate(
x_test, y_test, verbose = 0
)
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
for (i in 1:16) {
nr <- i * 10
tmpimg <- x_train[nr, , , 1]
dim(tmpimg) <- c(1, img_rows, img_cols, 1)
pred <- model %>% predict(tmpimg)
plot_image(x_train[nr, , , 1],
clothes.labels[which.max(pred)], "blue")
}
par(mfrow=c(1, 1))