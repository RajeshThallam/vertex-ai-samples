# install.R --------------------------------------------------------------

# install packages from CRAN ---------------------------------------------------
## create list of packages not already installed and install missing ones 
required_packages <- c("here",
                       "readr",
                       "gargle",
                       "googleAuthR",
                       "googleCloudStorageR",
                       "randomForest",
                       "plumber")

install.packages(setdiff(required_packages, rownames(installed.packages())))
