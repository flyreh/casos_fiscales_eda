"""
SCRIPTS PYTHON PARA LIMPIEZA DE DATOS - CASOS FISCALES PERÚ 2019-2023

Este archivo contiene todas las funciones necesarias para limpiar los 5 datasets
de casos fiscales con documentación completa y validaciones.

Autor: Sistema de Análisis Judicial
Fecha: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LimpiadorDatosFiscales:
    """
    Clase principal para limpiar datos de casos fiscales
    """
    
    def __init__(self, ruta_logs: str = "logs/"):
        """
        Inicializar el limpiador
        
        Args:
            ruta_logs: Directorio para guardar logs
        """
        self.ruta_logs = ruta_logs
        self.crear_directorio_logs()
        self.configurar_logging()
        self.resultados_limpieza = {}
        
    def crear_directorio_logs(self):
        """Crear directorio de logs si no existe"""
        if not os.path.exists(self.ruta_logs):
            os.makedirs(self.ruta_logs)
            
    def configurar_logging(self):
        """Configurar sistema de logging compatible con Windows"""
        # Crear formatter sin emojis para el archivo
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Crear formatter con emojis solo para consola
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Handler para archivo (sin emojis, con UTF-8)
        file_handler = logging.FileHandler(
            f'{self.ruta_logs}limpieza_datos.log', 
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        
        # Handler para consola (con emojis)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Configurar logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Evitar duplicación de logs
        self.logger.propagate = False
        
    def analizar_calidad_inicial(self, df: pd.DataFrame, año: int) -> Dict:
        """
        Analizar la calidad inicial del dataset
        
        Args:
            df: DataFrame a analizar
            año: Año del dataset
            
        Returns:
            Diccionario con métricas de calidad
        """
        self.logger.info(f"=== ANÁLISIS DATASET {año} ===")
        
        calidad = {
            'año': año,
            'filas_originales': len(df),
            'columnas': len(df.columns),
            'nulos_por_columna': df.isnull().sum().to_dict(),
            'tipos_datos': df.dtypes.to_dict(),
            'duplicados': df.duplicated().sum(),
            'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Análisis específico por tipo de columna
        calidad['analisis_categoricas'] = self._analizar_categoricas(df)
        calidad['analisis_numericas'] = self._analizar_numericas(df)
        calidad['analisis_fechas'] = self._analizar_fechas(df)
        
        # Mostrar resumen
        print(f"\nRESUMEN CALIDAD - {año}")
        print(f"   Filas: {calidad['filas_originales']:,}")
        print(f"   Columnas: {calidad['columnas']}")
        print(f"   Duplicados: {calidad['duplicados']}")
        print(f"   Memoria: {calidad['memoria_mb']:.2f} MB")
        
        # Mostrar nulos
        nulos_totales = sum(calidad['nulos_por_columna'].values())
        print(f"   Valores nulos: {nulos_totales:,}")
        
        for col, nulos in calidad['nulos_por_columna'].items():
            if nulos > 0:
                porcentaje = (nulos / calidad['filas_originales']) * 100
                print(f"     - {col}: {nulos:,} ({porcentaje:.1f}%)")
        
        return calidad
    
    def _analizar_categoricas(self, df: pd.DataFrame) -> Dict:
        """Analizar variables categóricas"""
        categoricas = ['distrito_fiscal', 'tipo_fiscalia', 'materia', 
                      'especialidad', 'tipo_caso', 'especializada']
        
        analisis = {}
        for col in categoricas:
            if col in df.columns:
                valores_unicos = df[col].nunique()
                valor_mas_frecuente = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                frecuencia_max = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                
                analisis[col] = {
                    'valores_unicos': valores_unicos,
                    'valor_mas_frecuente': valor_mas_frecuente,
                    'frecuencia_maxima': frecuencia_max,
                    'lista_valores': df[col].value_counts().head(10).to_dict()
                }
        
        return analisis
    
    def _analizar_numericas(self, df: pd.DataFrame) -> Dict:
        """Analizar variables numéricas"""
        numericas = ['anio', 'ingresado', 'atendido', 'ubigeo_pjfs']
        
        analisis = {}
        for col in numericas:
            if col in df.columns:
                serie = df[col].dropna()
                if len(serie) > 0:
                    analisis[col] = {
                        'min': serie.min(),
                        'max': serie.max(),
                        'media': serie.mean(),
                        'mediana': serie.median(),
                        'std': serie.std(),
                        'q25': serie.quantile(0.25),
                        'q75': serie.quantile(0.75),
                        'valores_cero': (serie == 0).sum(),
                        'valores_negativos': (serie < 0).sum()
                    }
        
        return analisis
    
    def _analizar_fechas(self, df: pd.DataFrame) -> Dict:
        """Analizar campos de fecha"""
        fechas = ['fecha_descarga', 'fecha_corte']
        
        analisis = {}
        for col in fechas:
            if col in df.columns:
                # Intentar parsear fechas
                try:
                    fechas_parseadas = pd.to_datetime(df[col], errors='coerce')
                    analisis[col] = {
                        'fecha_min': fechas_parseadas.min(),
                        'fecha_max': fechas_parseadas.max(),
                        'fechas_invalidas': fechas_parseadas.isnull().sum(),
                        'formato_detectado': 'DD/MM/YYYY'
                    }
                except:
                    analisis[col] = {
                        'error': 'No se pudo parsear el formato de fecha'
                    }
        
        return analisis
    
    def limpiar_dataset(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Aplicar todas las reglas de limpieza a un dataset
        
        Args:
            df: DataFrame original
            año: Año del dataset
            
        Returns:
            Tupla con (DataFrame limpio, diccionario de resultados)
        """
        self.logger.info(f"INICIANDO LIMPIEZA DATASET {año}")
        
        # 1. Análisis inicial
        calidad_inicial = self.analizar_calidad_inicial(df.copy(), año)
        
        # 2. Crear copia para trabajar
        df_limpio = df.copy()
        
        # 3. Aplicar reglas de limpieza
        df_limpio, resultado_nulos = self._limpiar_valores_nulos(df_limpio, año)
        df_limpio, resultado_duplicados = self._eliminar_duplicados(df_limpio, año)
        df_limpio, resultado_tipos = self._convertir_tipos_datos(df_limpio, año)
        df_limpio, resultado_categoricas = self._normalizar_categoricas(df_limpio, año)
        df_limpio, resultado_consistencia = self._validar_consistencia(df_limpio, año)
        df_limpio, resultado_outliers = self._identificar_outliers(df_limpio, año)
        
        # 4. Análisis final
        calidad_final = self.analizar_calidad_inicial(df_limpio, año)
        
        # 5. Compilar resultados
        resultado_completo = {
            'año': año,
            'calidad_inicial': calidad_inicial,
            'calidad_final': calidad_final,
            'transformaciones': {
                'nulos': resultado_nulos,
                'duplicados': resultado_duplicados,
                'tipos_datos': resultado_tipos,
                'categoricas': resultado_categoricas,
                'consistencia': resultado_consistencia,
                'outliers': resultado_outliers
            },
            'metricas_limpieza': self._calcular_metricas_limpieza(calidad_inicial, calidad_final)
        }
        
        # 6. Generar reporte
        self._generar_reporte_limpieza(resultado_completo)
        
        self.logger.info(f"LIMPIEZA COMPLETADA DATASET {año}")
        
        return df_limpio, resultado_completo
    
    def _limpiar_valores_nulos(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        """Limpiar valores nulos según reglas de negocio"""
        self.logger.info(f"Limpiando valores nulos - {año}")
        
        nulos_antes = df.isnull().sum().to_dict()
        columnas_numericas_cero = ['ingresado', 'atendido']
        
        for col in columnas_numericas_cero:
            if col in df.columns:
                nulos_originales = df[col].isnull().sum()
                valores_vacios = (df[col] == '').sum() if df[col].dtype == 'object' else 0
                if df[col].dtype == 'object':
                    df[col] = df[col].replace('', np.nan)
                    df[col] = df[col].replace(' ', np.nan)  # También espacios
                
                valores_convertidos = df[col].isnull().sum()
                df[col] = df[col].fillna(0)
                
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                print(f"variable > {col}: {valores_convertidos} valores nulos/vacíos convertidos a 0")
                self.logger.info(f"columna >{col}: nulos={nulos_originales}, vacíos={valores_vacios}, total convertidos={valores_convertidos}")
        
        columnas_criticas = ['periodo', 'anio', 'fecha_descarga', 'distrito_fiscal',
                        'tipo_fiscalia', 'materia', 'especialidad', 'tipo_caso',
                        'ubigeo_pjfs', 'dpto_pjfs', 'prov_pjfs', 'dist_pjfs', 'fecha_corte']
        
        filas_antes = len(df)
        for col in columnas_criticas:
            if col in df.columns:
                mask_nulos = df[col].isnull()
                if mask_nulos.any():
                    self.logger.warning(f"    Eliminando {mask_nulos.sum()} filas con nulos en '{col}'")
                    df = df[~mask_nulos]
        
        filas_despues = len(df)
        nulos_despues = df.isnull().sum().to_dict()
        
        resultado = {
            'filas_eliminadas': filas_antes - filas_despues,
            'nulos_antes': nulos_antes,
            'nulos_despues': nulos_despues,
            'columnas_criticas_procesadas': columnas_criticas,
            'columnas_numericas_convertidas': columnas_numericas_cero,
            'conversiones_a_cero': {
                col: nulos_antes.get(col, 0) for col in columnas_numericas_cero
            }
        }
        print(f" Filas eliminadas por nulos solo en campos clave: {resultado['filas_eliminadas']:,}")
        
        return df, resultado
    
    def _eliminar_duplicados(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        self.logger.info(f"Eliminando duplicados - {año}")
        
        filas_antes = len(df)
        duplicados_antes = df.duplicated().sum()
        df_sin_duplicados = df.drop_duplicates()
        
        filas_despues = len(df_sin_duplicados)
        duplicados_eliminados = filas_antes - filas_despues
        
        resultado = {
            'duplicados_detectados': duplicados_antes,
            'duplicados_eliminados': duplicados_eliminados,
            'filas_antes': filas_antes,
            'filas_despues': filas_despues
        }
        
        if duplicados_eliminados > 0:
            self.logger.warning(f"Duplicados eliminados: {duplicados_eliminados:,}")
        else:
            print(f"No se encontraron duplicados")
            
        return df_sin_duplicados, resultado
    
    def _convertir_tipos_datos(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        self.logger.info(f" Convirtiendo tipos de datos - {año}")
        tipos_antes = df.dtypes.to_dict()
        errores_conversion = {}
        
        try:
            df['fecha_descarga'] = pd.to_datetime(df['fecha_descarga'], errors='coerce')
            df['fecha_corte'] = pd.to_datetime(df['fecha_corte'], errors='coerce')
            
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce').fillna(0).astype('int32')
            
            for col in ['ingresado', 'atendido']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = df[col].clip(lower=0)
                    df[col] = df[col].astype('int32')
            
            df['ubigeo_pjfs'] = pd.to_numeric(df['ubigeo_pjfs'], errors='coerce').fillna(0).astype('int32')
            
            categoricas = ['distrito_fiscal', 'tipo_fiscalia', 'materia', 
                        'especialidad', 'tipo_caso', 'dpto_pjfs', 
                        'prov_pjfs', 'dist_pjfs']
            
            for col in categoricas:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        except Exception as e:
            errores_conversion[f'error_general'] = str(e)
            self.logger.error(f"Error en conversión: {e}")
        
        tipos_despues = df.dtypes.to_dict()
        
        resultado = {
            'tipos_antes': tipos_antes,
            'tipos_despues': tipos_despues,
            'errores': errores_conversion
        }
        
        print(f"Conversión de tipos completada")
        return df, resultado
    
    def _normalizar_categoricas(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        """Normalizar texto en variables categóricas"""
        self.logger.info(f"Normalizando variables categóricas - {año}")
        
        columnas_texto = ['distrito_fiscal', 'tipo_fiscalia', 'materia', 
                         'especialidad', 'tipo_caso', 'especializada',
                         'dpto_pjfs', 'prov_pjfs', 'dist_pjfs']
        
        cambios = {}
        
        for col in columnas_texto:
            if col in df.columns:
                valores_antes = df[col].unique()
                
                # Normalizar texto
                if col == 'especializada':
                    df[col] = df[col].astype(str).str.upper().str.strip()
                    
                    valores_a_reemplazar = ['NAN', 'NONE', 'NULL', '', 'NA', ',,']
                    
                    for valor in valores_a_reemplazar:
                        df[col] = df[col].replace(valor, 'NO_TIPO_ESPECIALIZADA')
                    
                    df[col] = df[col].replace(r'^\s*$', 'NO_TIPO_ESPECIALIZADA', regex=True)  # Solo espacios
                    df[col] = df[col].replace(r'^,+$', 'NO_TIPO_ESPECIALIZADA', regex=True)   # Solo comas
                    
                    df[col] = df[col].fillna('NO_TIPO_ESPECIALIZADA')                    
                    valores_no_especializada = (df[col] == 'NO_TIPO_ESPECIALIZADA').sum()
                    self.logger.info(f"{col}: {valores_no_especializada} valores convertidos a 'NO_TIPO_ESPECIALIZADA'")
                
                else:
                    df[col] = df[col].astype(str).str.upper().str.strip()
                
                valores_despues = df[col].unique()
                
                cambios[col] = {
                    'valores_antes': len(valores_antes),
                    'valores_despues': len(valores_despues),
                    'reduccion': len(valores_antes) - len(valores_despues)
                }
        
        resultado = {
            'columnas_procesadas': columnas_texto,
            'cambios_por_columna': cambios
        }
        
        print(f"Normalización categóricas completada")
        
        return df, resultado
    
    def _validar_consistencia(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        """Validar consistencias lógicas"""
        self.logger.info(f"Validando consistencias - {año}")
        
        # VALIDACIÓN 1: atendido <= ingresado
        casos_inconsistentes = df[df['atendido'] > df['ingresado']]
        
        if len(casos_inconsistentes) > 0:
            self.logger.warning(f"{len(casos_inconsistentes)} casos donde atendido > ingresado")
            # REGLA: Ajustar atendido = ingresado
            df.loc[df['atendido'] > df['ingresado'], 'atendido'] = \
                df.loc[df['atendido'] > df['ingresado'], 'ingresado']
        
        # VALIDACIÓN 2: Año vs fecha_descarga
        if 'fecha_descarga' in df.columns:
            df['año_descarga'] = df['fecha_descarga'].dt.year
            inconsistencias_fecha = df[df['año_descarga'] != df['anio'] + 1]
            
            if len(inconsistencias_fecha) > 0:
                self.logger.warning(f"{len(inconsistencias_fecha)} inconsistencias año vs fecha_descarga")
        
        # VALIDACIÓN 3: Valores negativos (después de la limpieza deberían ser 0)
        valores_negativos = {
            'ingresado': (df['ingresado'] < 0).sum(),
            'atendido': (df['atendido'] < 0).sum()
        }
        
        if any(valores_negativos.values()):
            self.logger.warning(f"Valores negativos detectados (convirtiendo a 0): {valores_negativos}")
            # Convertir valores negativos a 0
            df['ingresado'] = df['ingresado'].clip(lower=0)
            df['atendido'] = df['atendido'].clip(lower=0)
        
        # VALIDACIÓN 4: Casos donde ingresado=0 y atendido>0 (inconsistente)
        casos_atendido_sin_ingreso = df[(df['ingresado'] == 0) & (df['atendido'] > 0)]
        if len(casos_atendido_sin_ingreso) > 0:
            self.logger.warning(f"{len(casos_atendido_sin_ingreso)} casos con atendido>0 pero ingresado=0")
            # REGLA: Si no hay casos ingresados, no puede haber casos atendidos
            df.loc[(df['ingresado'] == 0) & (df['atendido'] > 0), 'atendido'] = 0
        
        resultado = {
            'casos_atendido_mayor_ingresado': len(casos_inconsistentes),
            'casos_ajustados': len(casos_inconsistentes),
            'valores_negativos': valores_negativos,
            'casos_atendido_sin_ingreso': len(casos_atendido_sin_ingreso)
        }
        
        print(f"Validación consistencias completada")
        
        return df, resultado
    
    def _identificar_outliers(self, df: pd.DataFrame, año: int) -> Tuple[pd.DataFrame, Dict]:
        """Identificar outliers en variables numéricas"""
        self.logger.info(f"Identificando outliers - {año}")
        
        columnas_numericas = ['ingresado', 'atendido']
        outliers_info = {}
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis de Outliers - {año}', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(columnas_numericas):
            # Método IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers_iqr = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
            
            # Método Z-score
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col]))
            outliers_zscore = df[z_scores > 3]
            
            outliers_info[col] = {
                'metodo_iqr': {
                    'count': len(outliers_iqr),
                    'percentage': (len(outliers_iqr) / len(df)) * 100,
                    'limite_inferior': limite_inferior,
                    'limite_superior': limite_superior
                },
                'metodo_zscore': {
                    'count': len(outliers_zscore),
                    'percentage': (len(outliers_zscore) / len(df)) * 100
                }
            }
            
            # Box Plot para método IQR
            box_data = df[col].values
            bp = axes[0, i].boxplot(box_data, patch_artist=True, 
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                flierprops=dict(marker='o', markerfacecolor='red', 
                                                markersize=4, alpha=0.6))
            
            axes[0, i].set_title(f'Box Plot IQR - {col.title()}')
            axes[0, i].set_ylabel('Valores')
            axes[0, i].grid(True, alpha=0.3)
            
            # Agregar líneas de límites IQR
            axes[0, i].axhline(y=limite_inferior, color='red', linestyle='--', 
                            alpha=0.7, label=f'Límite Inf: {limite_inferior:.0f}')
            axes[0, i].axhline(y=limite_superior, color='red', linestyle='--', 
                            alpha=0.7, label=f'Límite Sup: {limite_superior:.0f}')
            axes[0, i].legend(fontsize=8)
            
            # Añadir texto con estadísticas
            stats_text = f'Outliers IQR: {len(outliers_iqr)} ({(len(outliers_iqr)/len(df)*100):.1f}%)'
            axes[0, i].text(0.02, 0.98, stats_text, transform=axes[0, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        verticalalignment='top', fontsize=9)
            
            # Scatter Plot para método Z-Score
            x_values = range(len(df))
            
            mask_outliers_z = z_scores > 3
            
            axes[1, i].scatter(np.array(x_values)[~mask_outliers_z], 
                            df[col].values[~mask_outliers_z],
                            alpha=0.6, s=20, c='blue', label='Normal')
            
            # Plotear outliers
            if mask_outliers_z.any():
                axes[1, i].scatter(np.array(x_values)[mask_outliers_z], 
                                df[col].values[mask_outliers_z],
                                alpha=0.8, s=40, c='red', label='Outlier Z-Score')
            
            axes[1, i].set_title(f'Z-Score Outliers - {col.title()}')
            axes[1, i].set_xlabel('Índice de Registro')
            axes[1, i].set_ylabel('Valores')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
            
            # Añadir líneas de referencia (media ± 3σ)
            media = df[col].mean()
            std = df[col].std()
            axes[1, i].axhline(y=media + 3*std, color='red', linestyle=':', 
                            alpha=0.7, label=f'μ + 3σ')
            axes[1, i].axhline(y=media - 3*std, color='red', linestyle=':', 
                            alpha=0.7, label=f'μ - 3σ')
            
            # Añadir texto con estadísticas
            stats_text_z = f'Outliers Z-Score: {len(outliers_zscore)} ({(len(outliers_zscore)/len(df)*100):.1f}%)'
            axes[1, i].text(0.02, 0.98, stats_text_z, transform=axes[1, i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                        verticalalignment='top', fontsize=9)
            
            self.logger.info(f"variable> {col}:")
            self.logger.info(f"{len(outliers_iqr)} outliers IQR")
            self.logger.info(f"  Límites IQR: [{limite_inferior:.2f}, {limite_superior:.2f}]")
            self.logger.info(f"  Porcentaje de outliers IQR: {outliers_info[col]['metodo_iqr']['percentage']:.2f}%")
            self.logger.info(f"{len(outliers_zscore)} outliers z-score")
            self.logger.info(f"  Porcentaje de outliers z-score: {outliers_info[col]['metodo_zscore']['percentage']:.2f}%")
        
        plt.tight_layout()
        
        os.makedirs('resultados_limpieza/graficos', exist_ok=True)
        
        plt.savefig(f'resultados_limpieza/graficos/outliers_analisis_{año}.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        resultado = {
            'outliers_por_columna': outliers_info,
            'accion_tomada': 'Identificados pero no eliminados',
            'grafica_guardada': f'resultados_limpieza/graficos/outliers_analisis_{año}.png'
        }
        
        print(f"Identificación outliers completada")
        print(f"Gráfica guardada: resultados_limpieza/graficos/outliers_analisis_{año}.png")
        
        return df, resultado
    
    def _calcular_metricas_limpieza(self, calidad_inicial: Dict, calidad_final: Dict) -> Dict:
        """Calcular métricas de la limpieza"""
        return {
            'filas_eliminadas': calidad_inicial['filas_originales'] - calidad_final['filas_originales'],
            'porcentaje_retencion': (calidad_final['filas_originales'] / calidad_inicial['filas_originales']) * 100,
            'nulos_detectados': sum(calidad_inicial['nulos_por_columna'].values()) - sum(calidad_final['nulos_por_columna'].values()),
            'duplicados_eliminados': calidad_inicial['duplicados'] - calidad_final['duplicados'],
            'mejora_calidad': {
                'antes': calidad_inicial['filas_originales'] - sum(calidad_inicial['nulos_por_columna'].values()),
                'despues': calidad_final['filas_originales'] - sum(calidad_final['nulos_por_columna'].values())
            }
        }
    
    def _generar_reporte_limpieza(self, resultado: Dict):
        """Generar reporte detallado de limpieza"""
        año = resultado['año']
        
        print(f"\nREPORTE DE LIMPIEZA - {año}")
        print("="*50)
        
        inicial = resultado['calidad_inicial']
        final = resultado['calidad_final']
        metricas = resultado['metricas_limpieza']
        
        print(f"Filas originales: {inicial['filas_originales']:,}")
        print(f"Filas finales: {final['filas_originales']:,}")
        print(f"Filas eliminadas: {metricas['filas_eliminadas']:,}")
        print(f"Retención: {metricas['porcentaje_retencion']:.1f}%")
        print(f"Duplicados eliminados: {metricas['duplicados_eliminados']:,}")
        print(f"Nulos detectados: {metricas['nulos_detectados']:,}")
        
        # Guardar reporte en archivo
        with open(f"{self.ruta_logs}reporte_limpieza_{año}.json", 'w') as f:
            json.dump(resultado, f, indent=2, default=str)
    
    def procesar_todos_los_años(self, rutas_archivos: Dict[int, str]) -> Dict:
        """
        Procesar y limpiar todos los datasets de todos los años
        
        Args:
            rutas_archivos: Diccionario {año: ruta_archivo}
            
        Returns:
            Diccionario con datasets limpios y resultados
        """
        self.logger.info("INICIANDO PROCESAMIENTO COMPLETO")
        
        datasets_limpios = {}
        resultados_completos = {}
        
        for año, ruta in rutas_archivos.items():
            print(f"\n\nPROCESANDO AÑO {año}")
            
            try:
                # Cargar dataset
                df = pd.read_csv(ruta, encoding='utf-8')
                
                # Limpiar
                df_limpio, resultado = self.limpiar_dataset(df, año)
                
                # Guardar resultados
                datasets_limpios[año] = df_limpio
                resultados_completos[año] = resultado
                
                # Guardar dataset limpio
                ruta_salida = f"datos_limpios/casos_fiscales_{año}_limpio.csv"
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                df_limpio.to_csv(ruta_salida, index=False, encoding='utf-8')
                
                self.logger.info(f"Dataset {año} procesado y guardado en {ruta_salida}")
                
            except Exception as e:
                self.logger.error(f"❌ Error procesando año {año}: {e}")
                
        # Consolidar dataset
        if datasets_limpios:
            df_consolidado = pd.concat(datasets_limpios.values(), ignore_index=True)
            ruta_consolidado = "datos_limpios/casos_fiscales_2019_2023_consolidado.csv"
            df_consolidado.to_csv(ruta_consolidado, index=False, encoding='utf-8')
            
            self.logger.info(f"Dataset consolidado guardado en {ruta_consolidado}")
            
            # Generar reporte consolidado
            self._generar_reporte_consolidado(datasets_limpios, resultados_completos)
        
        return {
            'datasets_limpios': datasets_limpios,
            'resultados': resultados_completos,
            'dataset_consolidado': df_consolidado if datasets_limpios else None
        }
    
    def _generar_reporte_consolidado(self, datasets_limpios: Dict, resultados: Dict):
        """Generar reporte consolidado de todos los años"""
        print(f"\nREPORTE CONSOLIDADO 2019-2023")
        print("="*60)
        
        total_filas_originales = sum(r['calidad_inicial']['filas_originales'] for r in resultados.values())
        total_filas_finales = sum(r['calidad_final']['filas_originales'] for r in resultados.values())
        total_filas_eliminadas = total_filas_originales - total_filas_finales
        
        print(f"Total filas originales: {total_filas_originales:,}")
        print(f"Total filas finales: {total_filas_finales:,}")
        print(f"Total filas eliminadas: {total_filas_eliminadas:,}")
        print(f"Retención promedio: {(total_filas_finales/total_filas_originales)*100:.1f}%")
        
        print(f"\nResumen por año:")
        for año, resultado in resultados.items():
            inicial = resultado['calidad_inicial']['filas_originales']
            final = resultado['calidad_final']['filas_originales']
            retencion = (final/inicial)*100
            print(f"  {año}: {inicial:,} → {final:,} ({retencion:.1f}% retención)")
        
        # Guardar consolidado
        reporte_consolidado = {
            'resumen': {
                'total_filas_originales': total_filas_originales,
                'total_filas_finales': total_filas_finales,
                'retencion_promedio': (total_filas_finales/total_filas_originales)*100
            },
            'por_año': resultados
        }
        
        with open(f"{self.ruta_logs}reporte_consolidado.json", 'w') as f:
            json.dump(reporte_consolidado, f, indent=2, default=str)

# FUNCIÓN PRINCIPAL PARA EJECUTAR
def main():

    rutas_archivos = {
        2019: "datos_raw/BD-casos-fiscales-2019.csv",
        2020: "datos_raw/BD-casos-fiscales-2020.csv", 
        2021: "datos_raw/BD-casos-fiscales-2021.csv",
        2022: "datos_raw/BD-casos-fiscales-2022.csv",
        2023: "datos_raw/BD-casos-fiscales-2023.csv"
    }

    limpiador = LimpiadorDatosFiscales()
    
    resultado_completo = limpiador.procesar_todos_los_años(rutas_archivos)
    
    print("\n Procesamiento realizado")
    print("Archivos generados:")
    print(" datos_limpios/casos_fiscales_YYYY_limpio.csv (por año)")
    print(" datos_limpios/casos_fiscales_2019_2023_consolidado.csv")
    print(" logs/reporte_limpieza_YYYY.json (por año)")
    print(" logs/reporte_consolidado.json")
    print(" logs/limpieza_datos.log")
    
    return resultado_completo

if __name__ == "__main__":
    resultado = main()