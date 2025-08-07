# Ubicaciones

- `analisis_dimensiones/` - Contiene los resultados del análisis de dimensiones reales.
- `resultados_eda/` - Contiene los resultados graficos y json del análisis exploratorio de datos (EDA).
- `datos_limpios/` - Contiene los datasets limpios y además uno consolidado que sirve para todo el análisis.
- `datos_raw/` - Contiene los datasets originales.

# Ejecucion

1. Crear entorno virtual venv
``` python -m venv env```
2. Activar entorno virtual
``` .\env\Scripts\activate```
3. Instalar dependencias
``` pip install -r requirements.txt```
4. Ejecutar script de limpieza
``` python scripts/limpieza_datos.py```
5. Ejecutar analisis de dimensiones
``` python scripts/analisis_dimensiones_reales.py```
6. Ejecutar EDA
``` python scripts/eda.py```