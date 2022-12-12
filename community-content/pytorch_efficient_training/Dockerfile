FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest

COPY ./custom /custom

WORKDIR /custom

RUN pip install -r requirements.txt

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.main"]
