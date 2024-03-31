FROM python:3.11


WORKDIR /usr/src/app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


COPY . .

CMD [ "python", "API.py" ]