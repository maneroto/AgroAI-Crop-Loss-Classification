import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_file(path, names):
  df = pd.read_csv(path, header=None, names=names, encoding='latin-1')
  return df

def load_dataset(test_proportion=0.3):
  scaler = MinMaxScaler()
  headers = ["Anio","Idestado","Nomestado","Idddr","Nomddr","Idcader","Nomcader","Idmunicipio","Nommunicipio","Idciclo","Nomcicloproductivo","Idmodalidad","Nommodalidad","Idunidadmedida","Nomunidad","Idcultivo","Nomcultivo","Sembrada","Cosechada","Siniestrada","Volumenproduccion","Rendimiento","Preciomediorural","Valorproduccion"]

  file_path = "./data.csv"
  data_frame = read_file(file_path, headers)

  data_frame = data_frame.drop(columns=["Nomestado", "Idddr", "Nomddr", "Idcader", "Nomcader", "Nommunicipio", "Nomcicloproductivo", "Nommodalidad", "Nomunidad", "Nomcultivo", "Volumenproduccion", "Rendimiento", "Preciomediorural", "Valorproduccion", "Cosechada"]).drop(0)

  data_frame["Sembrada"] = data_frame["Sembrada"].str.replace(",", "")
  data_frame['Perdida'] = (data_frame['Siniestrada'].astype(float) > (data_frame["Sembrada"].astype(float, errors='ignore') * 0.2)).astype(int)

  has_no_wrecked = data_frame[data_frame['Siniestrada'].astype(float) == 0].sample(n=350)
  has_wrecked = data_frame[data_frame['Siniestrada'].astype(float) != 0]

  has_no_wrecked_test = has_no_wrecked.sample(frac=test_proportion)
  has_no_wrecked_train = has_no_wrecked.drop(has_no_wrecked_test.index)

  has_wrecked_test = has_wrecked.sample(frac=test_proportion)
  has_wrecked_train = has_wrecked.drop(has_wrecked_test.index)

  train_dataset = pd.concat([has_no_wrecked_train, has_wrecked_train]).drop(columns=["Siniestrada"]).sample(frac=1)
  test_dataset = pd.concat([has_no_wrecked_test, has_wrecked_test]).drop(columns=["Siniestrada"]).sample(frac=1)
  print(train_dataset)

  features = train_dataset.drop(columns=['Perdida']).columns.tolist()
  classes = train_dataset.columns[-1:].tolist()

  X_train = scaler.fit_transform(train_dataset.drop(columns=['Perdida']))
  y_train = train_dataset['Perdida']

  X_test = scaler.fit_transform(test_dataset.drop(columns=['Perdida']))
  y_test = test_dataset['Perdida']

  return X_train, y_train, X_test, y_test, features, classes

