# 02_train.R --------------------------------------------------------------

library(here)
library(randomForest)

# train model
model <- randomForest(Species ~ ., data = iris)

# save model
save(model, file = here("container","model.RData"))

