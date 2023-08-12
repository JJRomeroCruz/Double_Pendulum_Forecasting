# Double Pendulum Forecasting

## 1. Short description

This project consists in a new method to forecast the trajectory of a dynamical system based in artificial neural networks. I try to use this method with a toy model: the double pendulum. 

## 2. Motivation

In this case the dynamical system is the known double pendulum, this system is really interesting since
it has a chaotic beheaviour so, although is a deterministic problem (it is posible to give a prediction
by using the differential equations giving some initial conditions), is very difficult because it exhibits a extremely different 
movement if we change a little bit the initial conditions. In the world of dynamical systems, the ones which exhibit a chaotic behaaviour are usually the most likely to 
be forecasted with other models like artificial networks.

The aim of this project, in a selfish point of view, is to learn more about dynamical systems treated like a time serie

## 3. How can I install and execute it?

First, it is important to download the software 

In addition, this project requires the Deep Learning libraries keras and scikit-learn. And the most typical Python libaries such as matplotlib, numpy, simpy, scipy or pandas. 

## 4. Description of the code

This project has several Python scripts: 

\begin{enumerate}
  \item \textbf{Elastic_pendulum.py}: This was a failed attempt of doing all that I am doing with the pendulum but using the elastic pendulum.

  \item \textbf{Pendulum.py}: Here are the Pendulum's Ordinary Differential Equations Set. 

  \item \textbf{Search.py}: Here I am searching the best hyperparemeters in order to obtain the most accurate ANN model. Once they are found, the model is trained.

  \item \textbf{Functions.py}: In this script are all the functions related to the data analysis, use of other time series prediction models like arima or arma, tests of stationarity, use of the ANN model for predict the time series with the two both methods exposed, calculate the cuadratic error, calculate the system's energy and do some spectral analysis.

  \item \textbf{Train.py}: The aim of this script is training an ANN model with the hyperparemeters chosen by the user. This is a good option if you already know what hyperparemeters you want for your model.

  \item \textbf{create_model.py}: This script is just a vestige of having done something smaller than this project. It is useless. Don't use it.

  \item \textbf{main.py}: This script is the most important in terms of running the program. This script uses the ANN model previously trained, use the second forecasting method (see the Theory_resume.pdf) to predict the time serie, then it predicts this time serie with other methods (ARIMA, ARMA, VAR), then ir calculates the errors, the kinetic energy, the spectra. Then it plottes all the graphs of the time series for each prediction method, and then it generates the animation of the pendulum for each prediction method, in which the first half of the animation is the pendulum behaving according to the ODEs' solutions, and the second half consists in the pendulum moving according to other prediction model (ARMA, ARIMA, VAR, ANN, etc).
\end{enumerate}

## 5. License

## 6. Bibliography

- https://www.youtube.com/watch?v=JfeB_n4zsRM&t=1024s 
- A. Géron: “Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow“ (2019).
- P. I. Viñuela, I. M. Galván: “Redes de neuronas artificiales. Un enfoque práctico” (2004).
- https://matplotlib.org/stable/gallery/animation/double_pendulum.html
