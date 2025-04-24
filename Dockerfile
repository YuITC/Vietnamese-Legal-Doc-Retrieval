FROM continuumio/miniconda3

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN conda install -y \
      -c pytorch -c nvidia \
      python=3.10 \
      pytorch torchvision torchaudio pytorch-cuda=12.1 \
      faiss-gpu=1.9.0 && \
    conda clean -afy && \
    pip install --no-cache-dir -r /code/requirements.txt

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "main.py"]