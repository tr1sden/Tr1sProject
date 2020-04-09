library("kohonen")
library("boot")

setwd("C:/Users/Danilin Andrey/Desktop/RRRR")
data('nuclear')
# Цикл
a <- nuclear$cap

b <- NULL
for (i in 1:length(a)) {
  if (a[i] > 0 & a[i] < 682){b[i] = 1}
  if (a[i] > 683 & a[i] < 905) {b[i] = 2}
  if (a[i] > 906) {b[i] = 3}
}

Capacity <- as.factor(b) #задаем вектор переменной - мощности станции, где 1 - мощность до 682ед,  2 = 682 до 905, 3 = 905 и выше
set.seed(1)

som.nuclear <- som(scale(nuclear), grid = somgrid(3, 3, 'hexagonal')) 
#grid - таблица somgrid - размерность. 

plot(som.nuclear, main = 'nuclear Konohen')

plot(som.nuclear, type = 'changes', main = 'changes')#график ошибок , снизу указаны эпохи

#test vyborka


32*0.7
#sample - выборка с бесповторным отбором.

train <- sample(nrow(nuclear), 22)#тренировочная выборка с бесповторным отбором для 22 наблюдений. Получаем номера строк которые попадут в тестовую выборку.

train

x_train <- scale(nuclear[train,])#берем порядковые номера из исходной выборки с учетом найденных в процессе трейна строк
#формируем тестовую выборку.

x_test <- scale(nuclear[-train,],
                center = attr(x_train, 'scaled:center'),
                scale = attr(x_train, "scaled:center"))

#создадим таблицу со всеми исходными показателями и вектор значений.
#список будет состоять из двух позиций : матрица, с помощью которой будем обучать нейронку, 2 - вектор, который содержит исходные данные мощности станций.
train_data <- list(measurements = x_train, Capacity = Capacity[train])

test_data <- list(measurements = x_test, Capacity = Capacity[-train])# будут все, кроме train

som_grid <- somgrid(3,3,'hexagonal')


#supersom занимается обучением и весами
som.nuclear <- supersom(train_data, grid = som_grid)
plot(som.nuclear)


som_predict <- predict(som.nuclear, newdata = test_data) #берём обученную нейронку и вносим в неё исходные данные.

som_predict$predictions

table(Capacity[-train], som_predict$predictions[['Capacity']]) #с помощью матрицы будем подсчитывать количество правильных и неправильных ответов. берем все порядковые номера тестовой выборки

som.nuclear[('distances')]
#каждый вектор можно оценить по расстояниям. 

#создали датафрейм и два столбца векторов
classif <- data.frame(Capacity = Capacity[train],
                      class = som.nuclear[['unit.classif']]) #class- нейрон.