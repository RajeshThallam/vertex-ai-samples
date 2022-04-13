FROM rstudio/plumber

# install random forest
RUN R -e 'install.packages(c("randomForest"), repos = "https://cran.rstudio.com/")'

# Copy model and script
RUN mkdir /app
COPY model.RData /app
COPY serving.R /app
COPY startServer.R /app
WORKDIR /app

# plumber & run server
EXPOSE 7080

ENTRYPOINT ["R", "-f", "/app/startServer.R"]