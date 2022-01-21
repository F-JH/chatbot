FROM python:3.8
USER root
WORKDIR /app
COPY ./* ./
RUN pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip3 install transformers
CMD ["python","bot.py"]