import csv
import pandas as pd
from pathlib import Path

PATH = str(Path.cwd()) + "/"

def promedio_csv(n):
    # Obtener la lista de archivos CSV especificados por el usuario
    archivos_csv = []
    for i in range(n):
        archivo = input("Ingrese la ruta del archivo CSV: ")
        archivos_csv.append(archivo)

    # Calcular el promedio de los resultados
    resultados = []
    for archivo_csv in archivos_csv:
        with open(archivo_csv, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Suponiendo que los resultados están en la primera columna de cada fila
                resultado = float(row[0])
                resultados.append(resultado)
    
    promedio = sum(resultados) / len(resultados)

    # Guardar el promedio en otro archivo CSV
    archivo_salida = input("Ingrese la ruta del archivo de salida: ")
    with open(archivo_salida, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Promedio"])
        writer.writerow([promedio])

def f():
    script = "example.py"
    output = "emissions_float32.csv"
    usecols = [3, 4, 31, 32]
    meanVals = {""}

    for _ in range(10):
        exec(open(script).read())
        df = pd.read_csv(PATH + output, usecols=usecols)


# Ejemplo de uso
n = int(input("Ingrese la cantidad de archivos CSV a procesar: "))
promedio_csv(n)