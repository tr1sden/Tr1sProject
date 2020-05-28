##########Start#####################
#Собираем все нужные для работы пакеты
library(plotly)
library(BatchGetSymbols)
library('keras')
library('tensorflow')
library('minimax')

#Формируем период данных для сбора 
tickers <- c('DJI')
first.date <- Sys.Date() - 360*6
last.date <- Sys.Date()

# Загрузка данных
yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

# подготовка данных
y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)#выборка составила 1327 наблюдений
myts <-  myts[complete.cases(myts), ]#Удалили пропуски
myts <-  myts[-seq(nrow(myts) - 1200), ]#сократили выборку до 1200
myts$index <-  seq(nrow(myts))


# автокорреляция
acf(myts$price, lag.max = 1200)

# интерактивный график
plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)

# стандартизация данных
msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]
summary(myts)


# разбиваем совокупность на тестовую и тренировочную 1000 - тренировка и 200- тест
datalags = 10
train <-  myts[seq(1000), ]
test <-  myts[1000+ seq(200), ]
batch.size <-  50

###########SM##########################
#Работа с обычными последовательными моделями (sequential models)
#Стандартизация минимакс:
myts1 <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))
myts1

#разбиваем выборку с лагами
train1 <-  myts1[seq(1000 + datalags), ]
test1 <-  myts1[1000 + datalags + seq(200 + datalags), ]

train1 <-  myts1[seq(1000 + datalags), ]
test1 <-  myts1[1000 + datalags + seq(200 + datalags), ]

x.train1 <- array(data = lag(cbind(train1$price, train1$vol), datalags)[-(1:datalags), ], dim = c(nrow(train1) - datalags, datalags))
y.train1 <- array(data = train1$price[-(1:datalags)], dim = c(nrow(train1)-datalags))

#строим модели по различным спецификациям
#################
SM_rmsprop_mse = 0.0833

SM1 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM1 %>% compile(optimizer = "rmsprop",loss = "mse",) 

SM1 %>% fit(x.train1, y.train1, epochs = 10, batch_size = 16)

#################
SM_rmsprop_mae = 0.2411

  SM5 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM5 %>% compile(optimizer = "rmsprop",loss = "mae",) 

SM5 %>% fit(x.train1, y.train1, epochs = 10, batch_size = 16)

#################
SM_rmsprop_mape = 92.08

SM2 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM2 %>% compile(optimizer = "rmsprop",loss = "mape",) 

SM2 %>% fit(x.train1, y.train1, epochs = 30, batch_size = 16)

#################
SM_adam_mse = 0.0806

SM3 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM3 %>% compile(optimizer = "adam",loss = "mse",) 


SM3 %>% fit(x.train1, y.train1, epochs = 10, batch_size = 16)

#################
SM_adam_mape = 95.8814

SM4 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM4 %>% compile(optimizer = "adam",loss = "mape",) 


SM4 %>% fit(x.train1, y.train1, epochs = 10, batch_size = 16)

#################
SM_adam_mae =  0.2412

SM6 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.train1)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SM6 %>% compile(optimizer = "adam",loss = "mae",) 


SM6 %>% fit(x.train1, y.train1, epochs = 10, batch_size = 16)

##########RNN#######################
#Начинаем работу с рекуррентными нейронными сетями.
#Исходные данные подготовлены во время работы с последовательными сетями, используем их для последущих моделей рекуррентных нейронных сетей.
###################
RNN_rmsprop_mse = 0.0835

RNN1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN1 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)
RNN1 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)

###################
RNN_rmsprop_mape = 93.9368
  
RNN2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN2 %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)
RNN2 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)

###################
RNN_rmsprop_mae = 0.2535

RNN5 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN5 %>% compile(
  optimizer = "rmsprop",
  loss = "mae",
)
RNN5 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)

###################
RNN_adam_mse = 0.0749

  RNN3 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN3 %>% compile(
  optimizer = "adam",
  loss = "mse",
)
RNN3 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)
###################
RNN_adam_mape = 95.5410

  RNN4 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN4 %>% compile(
  optimizer = "adam",
  loss = "mape",
)
RNN4 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)

###################
RNN_adam_mae = 0.2385

RNN6 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")
RNN6 %>% compile(
  optimizer = "adam",
  loss = "mae",
)
RNN6 %>% fit(x.train1, y.train1, epochs = 10,batch_size = 50, validation_split = 0.2)
####################################
##########LSTM######################
datalags = 10
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <-  50

x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

dim(x.train)
dim(y.train)

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

###################
LSTM_rmsprop_mse = 0.0917
  
LSTM1 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

LSTM1 %>% compile(loss = 'mse', optimizer = 'rmsprop')

LSTM1 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)

###################
LSTM_rmsprop_mape = 68.8106
  
  LSTM2 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

LSTM2 %>% compile(loss = 'mape', optimizer = 'rmsprop')

LSTM2 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)

###################
LSTM_rmsprop_mae = 0.2230

LSTM5 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

LSTM5 %>% compile(loss = 'mae', optimizer = 'rmsprop')

LSTM5 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)
###################
LSTM_adam_mse = 0.0888
  
  LSTM3 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

  LSTM3 %>% compile(loss = 'mse', optimizer = 'adam')

  LSTM3 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)
###################
LSTM_adam_mape = 69.7568
  
  LSTM4 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

LSTM4 %>% compile(loss = 'mape', optimizer = 'adam')

LSTM4 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)
####################################
LSTM_adam_mae = 0.2249

LSTM6 <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

LSTM6 %>% compile(loss = 'mae', optimizer = 'adam')

LSTM6 %>% fit(x.train, y.train, epochs = 20, batch_size = batch.size)

####################################
######FINAL#########################
#Формируем финальную сводную таблицу с полученными результатами от всех моделей и выбираем лучший вариант.
Final <- data.frame(c("SM", "SM","RNN", "RNN","LSTM","LSTM"),
                     c("rmsprop","adam","rmsprop","adam","rmsprop","adam"),
                     c(SM_rmsprop_mse,SM_adam_mse,RNN_rmsprop_mse,RNN_adam_mse,LSTM_rmsprop_mse,LSTM_adam_mse),
                     c(SM_rmsprop_mape,SM_adam_mape,RNN_rmsprop_mape,RNN_adam_mape,LSTM_rmsprop_mape,LSTM_adam_mape),
                     c(SM_rmsprop_mae,SM_adam_mae,RNN_rmsprop_mae,RNN_adam_mae,LSTM_rmsprop_mae,LSTM_adam_mae))

#Лучшая модель - LSTM-rmsprop. Второе место - RNN-adam


##########SM-lag#################################################
# сцена после титров - последовательная модель - без лага :)
# разбиваем совокупность на тестовую и тренировочную 1000 - тренировка и 200- тест

datalags = 10
train <-  myts[seq(1000), ]
test <-  myts[1000+ seq(200), ]
batch.size <-  50


x.trainSM <- as.matrix(cbind(train$vol, train$price))
y.trainSM <- as.matrix(train$price)


x.testSM <-  as.matrix(cbind(test$vol,test$price))
y.testSM <-  as.matrix(test$price)

SML_rmsprop_mse = 0.0633

SML1 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.trainSM)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SML1 %>% compile(optimizer = "rmsprop",loss = "mse",) 


SML1 %>% fit(x.trainSM, y.trainSM, epochs = 10, batch_size = 16)
SML_rmsprop_mape = 37.5549

SML2 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.trainSM)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SML2 %>% compile(optimizer = "rmsprop",loss = "mape",) 


SML2 %>% fit(x.trainSM, y.trainSM, epochs = 10, batch_size = 16)

SML_adam_mse = 0.0428

SML3 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.trainSM)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SML3 %>% compile(optimizer = "adam",loss = "mse",) 


SML3 %>% fit(x.trainSM, y.trainSM, epochs = 10, batch_size = 16)

SML_adam_mape = 38.5399

SML4 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(x.trainSM)[[2]]) %>%
  layer_dense(units = 64, activation = "softmax") %>%
  layer_dense(units = 1)
SML4 %>% compile(optimizer = "adam",loss = "mape",) 


SML4 %>% fit(x.trainSM, y.trainSM, epochs = 10, batch_size = 16)
####################################