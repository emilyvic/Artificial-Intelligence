{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Sin Framework - Emilia_Jácome-A00828347.ipynb",
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyMApPnj5hHtROAUZBw+TWKB",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/emilyvic/Artificial-Intelligence/blob/main/Sin_Framework_Emilia_J%C3%A1come_A00828347.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Emilia Victoria Jácome Iñiguez\n",
        "\n",
        "A00828347\n",
        "\n",
        "**MOMENTO DE RETROALIMENTACIÓN: IMPLEMENTACIÓN DE UNA TÉCNICA DE APRENDIZAJE MÁQUINA SIN EL USO DE UN FRAMEWORK**"
      ],
      "metadata": {
        "id": "tUEizQNciIIQ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Preliminar"
      ],
      "metadata": {
        "id": "oEaUrfhSjnJN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Conexión a Drive"
      ],
      "metadata": {
        "id": "HcrPkc-WjSOC"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DO0HXMx4hpym",
        "outputId": "0134d200-e17c-4c18-87aa-5c21c5f5e4ca"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cd /content/drive/Shareddrives/Los Nietos de Gauss"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0Hgdc0WIjXpo",
        "outputId": "3cc2b9fa-c0d6-4ed8-a7d4-39eda71a0cfb"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/Shareddrives/Los Nietos de Gauss\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Cargar los datos a data frames"
      ],
      "metadata": {
        "id": "ysyRpHb1jih8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "df_train = pd.read_csv('train.csv') #generar el dataframe con los datos del titanic\n",
        "df_test = pd.read_csv('test.csv') #generar el dataframe con los datos del titanic\n",
        "df_gs = pd.read_csv('gender_submission.csv') #generar el dataframe con los datos del titanic"
      ],
      "metadata": {
        "id": "Kt0sVeC-jaYs"
      },
      "execution_count": 5,
      "outputs": []
    }
  ]
}