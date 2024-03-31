# provafiesc

Arquivos utilizados durante o teste técnico para vaga de Pesquisador de Visão Computacional.

## Pré-requisitos

- [Docker](https://www.docker.com/)

## Instalação

- Dar build na imagem, com o comando:


    docker build --no-cache -t juliocezarmendonca/provafiesc:1.0 .


- Rodar a imagem, com o comando:


    docker run -d -p 5000:5000 juliocezarmendonca/provafiesc:1.0


## Como utilizar
- Importar o arquivo da collection no seu API Client de preferência
- Testar as rotas de acordo com o teste
