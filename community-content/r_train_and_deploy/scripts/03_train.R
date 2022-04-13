# 02_train.R --------------------------------------------------------------

library(here)
library(randomForest)

# train model
model <- randomForest(Species ~ ., data = iris)

# save model
# TODO @justijm - resolve directory for model and image via docker file 
save(model, file = "./build/model.RData")

# upload to cloud storage 
# TODO @justinjm - 
