# get_data.R --------------------------------------------------------------

library(here)
library(dplyr)
library(readr)
library(jsonlite)

source(here("scripts", "functions.R"))

# Source https://datahub.io/machine-learning/bank-marketing
# https://datahub.io/machine-learning/bank-marketing/r/1.html

json_file <- "https://datahub.io/machine-learning/bank-marketing/datapackage.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))

# get list of all resources:
print(json_data$resources$name)

# print all tabular data(if exists any)
for(i in 1:length(json_data$resources$datahub$type)){
  if(json_data$resources$datahub$type[i]=='derived/csv'){
    path_to_file = json_data$resources$path[i]
    data <- read.csv(url(path_to_file))
    print(data)
  }
}

data_out <- data 

column_names <- c(V1 = "age",
                  V2 = "job",
                  V3 = "marital",
                  V4 = "education",
                  V5 = "default",
                  V6 = "balance",
                  V7 = "housing",
                  V8 = "loan",
                  V9 = "contact",
                  V10 = "day",
                  V11 = "month",
                  V12 = "duration",
                  V13 = "campaign",
                  V14 = "pdays",
                  V15 = "previous",
                  V16 = "poutcome",
                  Class = "subscribed")

colnames(data_out) <- set_names(data_frame = data_out, 
                                lookup_names = column_names)

write_csv(data_out, here("data", "raw", "bank_marketing.csv"))
