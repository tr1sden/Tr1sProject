# Установка и подключение пакетов
install.packages('devtools')
library('devtools')
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
library('keras')
library('tensorflow')

# загрузка данных
mnist <- dataset_mnist()

# для удобства разиваем их на 4 объекта
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# строим архитектуру нейронной сети

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = 'softmax')

# Добавляем для нейронной сети оптимизатор, функцию потерь, какие метрики выводить на экран (в примере выводится только точность)
network %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Изначально массивы имеют размерность 60000, 28, 28, сами значения изменяются в пределах от 0 до 255
# для обучения нейронной сети потребуется преобразовать форму 60000, 28*28, а значения перевести в размерность от 0 до 1

train_images <- array_reshape(train_images, c(60000, 28*28)) # меняем размерность к матрице
train_images <- train_images/255 # меняем область значений
str(train_images)
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# создаем категории для ярлыков

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# после подготовки данных тренируем нейронную сеть

network %>% fit(train_images, train_labels, epochs = 20, batch_size = 128)

# точность модели по 30 эпохам составила %98,31, принято решение использовать 20 эпох, посколькоу изменение незначительно.
# точность модели по 20 эпохам составила %98,25


metric <- network %>% evaluate(test_images, test_labels)
metric

test_labels1 <- mnist$test$y

# Предсказываем значения для первой и последней сотни чисел
First100 <- network %>% predict_classes(test_images[1:100,])
Last100 <- network %>% predict_classes(test_images[9901:10000,])

# Создание базы данных со столбцами "Последние сто исходных чисел", "Предсказанные первые сто чисел", "Первые сто исходных чисел", "Предсказанные последние сто  чисел") для более удобного сравнения
Comparison <- data.frame(First100based=test_labels1[1:100],PredictedF100 = First100, Last100Based = test_labels1[9901:10000],PredictedL100=Last100)

# обучение сетей с добавлением валидации
history <- network %>% fit(train_images, train_labels,
                           epochs = 5, batch_size = 128,
                           validation_split = 0.2)
# простой точечный график процесса обучения
plot(history)