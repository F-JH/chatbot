FROM continuumio/anaconda3
USER root
WORKDIR /app
COPY ./* ./
RUN conda install -y pytorch torchvision torchaudio -c pytorch \
    && conda install -y transformers
CMD ["python","bot.py"]