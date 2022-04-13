# deploy_dockerfile.R ---------------------------------------------------------
library(googleCloudRunner)

cr_deploy_docker(local = "build/",
                 image_name = "gcr.io/user-name/repo-name",
                 timeout = 1000L,
                 tag = "latest")