FROM python:3.8
USER root
WORKDIR /app
COPY ./checkpoint/* ./checkpoint/
COPY ./config/* ./config/
COPY ./models/* ./models/
COPY ./scripts/* ./scripts/
COPY ./utils/* ./utils/
COPY ./bot.py ./
RUN pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip3 install transformers \
    && pip3 install flask numpy
CMD ["python","bot.py"]