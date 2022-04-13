# install.R --------------------------------------------------------------

# install packages from CRAN ---------------------------------------------------
## create list of packages not already installed and install missing ones 
required_packages <- c("here",
                       "devtools",
                       "dpylr",
                       "readr",
                       "gargle",
                       "googleAuthR",
                       "googleCloudStorageR",
                       "googleCloudRunner",
                       "randomForest",
                       "plumber")

install.packages(setdiff(required_packages, rownames(installed.packages())))

# install required packages from GitHub  --------------------------------------
# devtools::install_github("username/reponame")