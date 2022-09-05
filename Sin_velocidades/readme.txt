Este proyecto recoge todo el trabajo que se ha hecho a la hora de usar modelos para la prediccion de series temporales: ANN, ARIMA, VAR, tanto multipaso como paso a paso.

This project includes all the work done in order to use models for the time series forecasting: ANN (Artificial Neural Networks, ARIMA, VAR (vectorial autorregresive model), both multistep and stepper.

This is the functions used in this project, as well as its output and input:

	separar_set(x, y, n, r) --> np.array(xTrain), np.array(xTest), np.array(yTrain), np.array(yTest):  the pandas DAtaFrames x, y; its maximum size, n; and the proportion we want to split up in, r; are given to separate the two both datasets into two datasets each one, this function gives the both four sets in form of numpy arrays.
	
	arima_univariant(datos, p, d, q) --> theta (numpy array): Where datos is a Pandas DataFrame, and p, d, q are floats. This function separates the dataset datos in two halfs, and we use the first half to produce a new second half with an arima(p, d, q) defined previously in multistep. Theta is the array result of the joint of the first datos's half and the second half forecasted withe the arima model.
	
	var_multi(datos) --> y (numpy array): datos is a Pandas DataFrame, this function does the same as arima_univariant but with a VAR model.
	
	var_paso(datos) --> y_verdad (numpy array): datos is a Pandas DataFrame, this function does the same as arima_univariant but with a VAR model and forecasting step by step.
	
	crear_ANN(taza_aprendizaje, taza_abandono) --> model (tf.keras.Sequential): in this function a model of artificial neural network is defined.
	
	ANN_paso(model, datos) --> y_verdad (numpy array): model is a tf.keras.Sequential and datos is a Pandas DataFrame. This function does the same as var_paso, but using an artificial neural network instead of a VAR model.
	
	ANN_multipaso(model, datos) --> y_verdad (numpy array): model is a tf.keras.Sequential and datos is a Pandas DataFrame. This function does the same as var_multi, but using an artificial neural network instead of VAR model.
	
	
