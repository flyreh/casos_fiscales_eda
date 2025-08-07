"""

Este script realiza un análisis exploratorio exhaustivo y corregido de los datos de casos fiscales
enfocado en interpretación de hallazgos y corrección de errores del análisis original.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


from scipy import stats
from scipy.stats import chi2_contingency, normaltest, kstest, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

import warnings
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class EDAFiscaliaMejorado:
    
    def __init__(self, df: pd.DataFrame, nombre_dataset: str = "Casos Fiscales"):
        """
        Inicializar el analizador EDA mejorado
        
        Args:
            df: DataFrame con los datos
            nombre_dataset: Nombre descriptivo del dataset
        """
        self.df = df.copy()
        self.nombre_dataset = nombre_dataset
        self.resultados_eda = {}
        self.interpretaciones = {}

        self.ruta_imagenes = 'resultados_eda/graphics/'
        os.makedirs(self.ruta_imagenes, exist_ok=True)
        
        self._configurar_estilo()
        
        self._identificar_tipos_columnas_corregido()

        self._preprocesar_datos()
        
        print(f"EDA iniciado para: {nombre_dataset}")
        print(f"Dataset shape: {self.df.shape}")
        self._mostrar_resumen_inicial()
    
    def _configurar_estilo(self):
        plt.style.use('default')
        sns.set_palette("Set2")
        
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def _guardar_figura(self, fig, nombre_archivo):
        ruta_completa = os.path.join(self.ruta_imagenes, nombre_archivo)
        fig.savefig(f"{ruta_completa}.pdf", bbox_inches='tight')
        print(f" Gráfico guardado como: {ruta_completa}.png")
        
    def _identificar_tipos_columnas_corregido(self):
        
        # Variables numéricas reales
        self.columnas_numericas_reales = ['anio', 'ingresado', 'atendido']
        
        # Variables categóricas principales
        self.columnas_categoricas = [
            'distrito_fiscal', 'tipo_fiscalia', 'materia', 'especialidad', 
            'tipo_caso', 'especializada', 'dpto_pjfs', 'prov_pjfs'
        ]
        
        self.columnas_identificacion = ['ubigeo_pjfs', 'dist_pjfs']
        
        self.columnas_fechas = ['fecha_descarga', 'fecha_corte', 'periodo']
        
        self.columnas_redundantes = ['año_descarga']  # Redundante con anio
        
        # Métricas principales del negocio
        self.metricas_principales = ['ingresado', 'atendido']
        
        # Dimensiones geográficas
        self.dimensiones_geograficas = ['distrito_fiscal', 'dpto_pjfs', 'prov_pjfs']
        
        print(f"Variables numéricas reales: {len(self.columnas_numericas_reales)}")
        print(f"Variables categóricas: {len(self.columnas_categoricas)}")
        print(f"Variables de identificación: {len(self.columnas_identificacion)}")
    
    def _preprocesar_datos(self):
        """Preprocesar datos para análisis correcto"""
        
        # Tasa de atención (manejando división por cero)
        self.df['tasa_atencion'] = np.where(
            self.df['ingresado'] > 0,
            self.df['atendido'] / self.df['ingresado'],
            np.nan
        )
        
        # Casos pendientes
        self.df['casos_pendientes'] = self.df['ingresado'] - self.df['atendido']
        
        # Indicador de eficiencia perfecta
        self.df['eficiencia_perfecta'] = (self.df['atendido'] >= self.df['ingresado']).astype(int)
        
        # Categorizar volumen de casos
        self.df['volumen_ingresado'] = pd.cut(
            self.df['ingresado'],
            bins=[0, 10, 50, 200, 1000, float('inf')],
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        )
        
        # Limpiar especializada
        self.df['especializada_limpia'] = self.df['especializada'].replace(
            'NO_TIPO_ESPECIALIZADA', 'Sin Especialización'
        )
        
        # Variables booleanas útiles
        self.df['tiene_especializacion'] = (
            self.df['especializada'] != 'NO_TIPO_ESPECIALIZADA'
        ).astype(int)
        
        self.df['casos_sin_atencion'] = (self.df['atendido'] == 0).astype(int)
        self.df['casos_sin_ingreso'] = (self.df['ingresado'] == 0).astype(int)
        
        print("Variables derivadas creadas correctamente")
    
    def _mostrar_resumen_inicial(self):
        print("\nResumen inicial del dataset")
        print("-" * 50)
        
        # Información básica
        print(f"Período de análisis: {self.df['anio'].min()}-{self.df['anio'].max()}")
        print(f"Total de registros: {len(self.df):,}")
        print(f"Distritos fiscales: {self.df['distrito_fiscal'].nunique()}")
        print(f"Materias: {self.df['materia'].nunique()}")
        
        # Estadísticas de casos
        print(f"\nEstadísticas de casos:")
        print(f"  Total casos ingresados: {self.df['ingresado'].sum():,}")
        print(f"  Total casos atendidos: {self.df['atendido'].sum():,}")
        print(f"  Tasa global de atención: {self.df['atendido'].sum()/self.df['ingresado'].sum()*100:.2f}%")
        
        # Casos especiales
        casos_sin_ingreso = (self.df['ingresado'] == 0).sum()
        casos_sin_atencion = (self.df['atendido'] == 0).sum()
        casos_sobreatendidos = (self.df['atendido'] > self.df['ingresado']).sum()
        
        print(f"\nCasos especiales:")
        print(f"  Sin casos ingresados: {casos_sin_ingreso} ({casos_sin_ingreso/len(self.df)*100:.2f}%)")
        print(f"  Sin casos atendidos: {casos_sin_atencion} ({casos_sin_atencion/len(self.df)*100:.2f}%)")
        print(f"  Sobre-atendidos: {casos_sobreatendidos} ({casos_sobreatendidos/len(self.df)*100:.2f}%)")
    
    def estadisticas_descriptivas_mejoradas(self) -> Dict:
       
        print("\nCalculando estadísticas descriptivas...")
        
        estadisticas = {
            'resumen_general': self._calcular_resumen_general(),
            'variables_numericas': self._analizar_variables_numericas(),
            'variables_categoricas': self._analizar_variables_categoricas(),
            'metricas_negocio': self._analizar_metricas_negocio(),
            'correlaciones': self._analizar_correlaciones()
        }
        
        # Interpretar resultados
        self.interpretaciones['estadisticas'] = self._interpretar_estadisticas(estadisticas)
        
        # Guardar resultados
        self.resultados_eda['estadisticas_descriptivas'] = estadisticas
        
        self._mostrar_interpretaciones_estadisticas()
        
        return estadisticas
    
    def _calcular_resumen_general(self) -> Dict:
        return {
            'total_registros': len(self.df),
            'periodo_inicio': int(self.df['anio'].min()),
            'periodo_fin': int(self.df['anio'].max()),
            'total_distritos': int(self.df['distrito_fiscal'].nunique()),
            'total_departamentos': int(self.df['dpto_pjfs'].nunique()),
            'memoria_mb': float(self.df.memory_usage(deep=True).sum() / (1024**2)),
            'casos_totales_ingresados': int(self.df['ingresado'].sum()),
            'casos_totales_atendidos': int(self.df['atendido'].sum()),
            'tasa_atencion_global': float(self.df['atendido'].sum() / self.df['ingresado'].sum()),
            'completitud_datos': float((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100)
        }
    
    def _analizar_variables_numericas(self) -> Dict:
        analisis = {}
        
        for col in self.columnas_numericas_reales:
            serie = self.df[col].dropna()
            
            if len(serie) > 0:
                # Estadísticas básicas
                desc = serie.describe()
                
                # Estadísticas adicionales
                analisis[col] = {
                    # Estadísticas centrales
                    'count': int(desc['count']),
                    'mean': float(desc['mean']),
                    'median': float(desc['50%']),
                    'mode': float(serie.mode().iloc[0]) if not serie.mode().empty else None,
                    'std': float(desc['std']),
                    'min': float(desc['min']),
                    'max': float(desc['max']),
                    'q25': float(desc['25%']),
                    'q75': float(desc['75%']),
                    
                    # Medidas de variabilidad
                    'range': float(desc['max'] - desc['min']),
                    'iqr': float(desc['75%'] - desc['25%']),
                    'coefficient_variation': float(serie.std() / serie.mean()) if serie.mean() != 0 else 0,
                    
                    # Medidas de forma
                    'skewness': float(stats.skew(serie)),
                    'kurtosis': float(stats.kurtosis(serie)),
                    
                    # Valores especiales
                    'zeros_count': int((serie == 0).sum()),
                    'zeros_percentage': float((serie == 0).sum() / len(serie) * 100),
                    'negative_count': int((serie < 0).sum()),
                    'missing_count': int(self.df[col].isnull().sum()),
                    'missing_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                    
                    # Tests estadísticos
                    'normalidad_shapiro_p': float(stats.shapiro(serie.sample(min(5000, len(serie))))[1]) if len(serie) >= 3 else None,
                    'es_normal': bool(stats.shapiro(serie.sample(min(5000, len(serie))))[1] > 0.05) if len(serie) >= 3 else None,
                }
                
                # Interpretaciones específicas por variable
                if col in ['ingresado', 'atendido']:
                    # Para métricas de casos, añadir análisis específico
                    analisis[col].update({
                        'casos_extremos_altos': int((serie > serie.quantile(0.95)).sum()),
                        'casos_extremos_bajos': int((serie < serie.quantile(0.05)).sum()),
                        'concentracion_top10': float(serie.nlargest(10).sum() / serie.sum() * 100),
                    })
        
        return analisis
    
    def _analizar_variables_categoricas(self) -> Dict:
        analisis = {}
        
        for col in self.columnas_categoricas:
            if col in self.df.columns:
                serie = self.df[col].dropna()
                value_counts = self.df[col].value_counts()
                
                analisis[col] = {
                    'unique_values': int(self.df[col].nunique()),
                    'missing_count': int(self.df[col].isnull().sum()),
                    'missing_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                    
                    # Análisis de distribución
                    'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'mode_percentage': float(value_counts.iloc[0] / len(self.df) * 100) if len(value_counts) > 0 else 0,
                    
                    # Medidas de concentración
                    'entropy': float(stats.entropy(value_counts.values)) if len(value_counts) > 1 else 0,
                    'gini_concentration': self._calcular_gini(value_counts.values),
                    'top_3_values': dict(value_counts.head(3)),
                    'top_3_percentage': float(value_counts.head(3).sum() / len(self.df) * 100),
                    
                    # Análisis específico por tipo de variable
                    'interpretacion_concentracion': self._interpretar_concentracion(value_counts),
                }
        
        return analisis
    
    def _analizar_metricas_negocio(self) -> Dict:
        """Análisis específico de métricas del negocio judicial"""
        metricas = {}
        
        # Análisis de eficiencia
        tasa_atencion_valida = self.df['tasa_atencion'].dropna()
        
        metricas['eficiencia'] = {
            'tasa_atencion_promedio': float(tasa_atencion_valida.mean()),
            'tasa_atencion_mediana': float(tasa_atencion_valida.median()),
            'casos_eficiencia_perfecta': int((tasa_atencion_valida >= 1.0).sum()),
            'casos_eficiencia_perfecta_pct': float((tasa_atencion_valida >= 1.0).sum() / len(tasa_atencion_valida) * 100),
            'casos_baja_eficiencia': int((tasa_atencion_valida < 0.5).sum()),
            'casos_baja_eficiencia_pct': float((tasa_atencion_valida < 0.5).sum() / len(tasa_atencion_valida) * 100),
        }
        
        # Análisis de volumen
        metricas['volumen'] = {
            'distribucion_volumen': dict(self.df['volumen_ingresado'].value_counts()),
            'casos_alto_volumen': int((self.df['ingresado'] > 1000).sum()),
            'casos_bajo_volumen': int((self.df['ingresado'] <= 10).sum()),
            'concentracion_casos_top_10_distritos': float(
                self.df.groupby('distrito_fiscal')['ingresado'].sum().nlargest(10).sum() / 
                self.df['ingresado'].sum() * 100
            )
        }
        
        # Análisis temporal
        if self.df['anio'].nunique() > 1:
            evolucion_anual = self.df.groupby('anio')[['ingresado', 'atendido']].sum()
            metricas['temporal'] = {
                'tendencia_ingresados': self._calcular_tendencia(evolucion_anual['ingresado']),
                'tendencia_atendidos': self._calcular_tendencia(evolucion_anual['atendido']),
                'variabilidad_anual_ingresados': float(evolucion_anual['ingresado'].std()),
                'crecimiento_total_periodo': float(
                    (evolucion_anual['ingresado'].iloc[-1] - evolucion_anual['ingresado'].iloc[0]) /
                    evolucion_anual['ingresado'].iloc[0] * 100
                )
            }
        
        return metricas
    
    def _analizar_correlaciones(self) -> Dict:
        vars_correlacion = ['ingresado', 'atendido', 'tasa_atencion', 'casos_pendientes']
        
        # Filtrar variables que existen
        vars_disponibles = [var for var in vars_correlacion if var in self.df.columns]
        
        if len(vars_disponibles) < 2:
            return {}
        
        # Calcular correlaciones
        matriz_corr_pearson = self.df[vars_disponibles].corr()
        matriz_corr_spearman = self.df[vars_disponibles].corr(method='spearman')
        
        correlaciones = {
            'matriz_pearson': matriz_corr_pearson.to_dict(),
            'matriz_spearman': matriz_corr_spearman.to_dict(),
            'correlaciones_fuertes_pearson': self._encontrar_correlaciones_fuertes(matriz_corr_pearson),
            'correlaciones_fuertes_spearman': self._encontrar_correlaciones_fuertes(matriz_corr_spearman),
        }
        
        return correlaciones
    
    def _calcular_gini(self, valores):
        valores_sorted = np.sort(valores)
        n = len(valores)
        cumsum = np.cumsum(valores_sorted)
        return (2 * np.sum((np.arange(1, n+1)) * valores_sorted)) / (n * cumsum[-1]) - (n+1) / n
    
    def _interpretar_concentracion(self, value_counts):
        """Interpretar nivel de concentración en variable categórica"""
        total = value_counts.sum()
        top_3_pct = value_counts.head(3).sum() / total * 100
        
        if top_3_pct >= 80:
            return "Muy alta concentración - dominada por pocas categorías"
        elif top_3_pct >= 60:
            return "Alta concentración - algunas categorías dominantes"
        elif top_3_pct >= 40:
            return "Concentración moderada - distribución desbalanceada"
        else:
            return "Baja concentración - distribución relativamente balanceada"
    
    def _calcular_tendencia(self, serie_temporal):
        x = np.arange(len(serie_temporal))
        y = serie_temporal.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'pendiente': float(slope),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'es_significativa': bool(p_value < 0.05),
            'direccion': 'Creciente' if slope > 0 else 'Decreciente' if slope < 0 else 'Estable'
        }
    
    def _encontrar_correlaciones_fuertes(self, matriz_corr: pd.DataFrame, umbral: float = 0.5) -> List[Dict]:
        # correlaciones fuertes entre variables
        correlaciones_fuertes = []
        
        for i in range(len(matriz_corr.columns)):
            for j in range(i+1, len(matriz_corr.columns)):
                col1 = matriz_corr.columns[i]
                col2 = matriz_corr.columns[j]
                correlacion = matriz_corr.iloc[i, j]
                
                if abs(correlacion) >= umbral and not np.isnan(correlacion):
                    correlaciones_fuertes.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlacion': float(correlacion),
                        'fuerza': self._interpretar_fuerza_correlacion(abs(correlacion)),
                        'direccion': 'Positiva' if correlacion > 0 else 'Negativa'
                    })
        
        return sorted(correlaciones_fuertes, key=lambda x: abs(x['correlacion']), reverse=True)
    
    def _interpretar_fuerza_correlacion(self, abs_corr):
        """Interpretar la fuerza de una correlación"""
        if abs_corr >= 0.9:
            return "Muy fuerte"
        elif abs_corr >= 0.7:
            return "Fuerte"
        elif abs_corr >= 0.5:
            return "Moderada"
        elif abs_corr >= 0.3:
            return "Débil"
        else:
            return "Muy débil"
    
    def _interpretar_estadisticas(self, estadisticas: Dict) -> Dict:
        """Generar interpretaciones de los hallazgos estadísticos"""
        interpretaciones = []
        
        # Interpretar resumen general
        resumen = estadisticas['resumen_general']
        interpretaciones.append(f"El dataset abarca {resumen['periodo_fin'] - resumen['periodo_inicio'] + 1} años con {resumen['total_registros']:,} registros")
        interpretaciones.append(f"Tasa de atención global del {resumen['tasa_atencion_global']*100:.1f}% indica {'alta' if resumen['tasa_atencion_global'] > 0.8 else 'baja'} eficiencia del sistema")
        
        # Interpretar variables numéricas
        if 'variables_numericas' in estadisticas:
            for var, stats in estadisticas['variables_numericas'].items():
                if var in ['ingresado', 'atendido']:
                   
                    if abs(stats['skewness']) > 1:
                        interpretaciones.append(f"{var.title()} muestra distribución muy asimétrica ({'derecha' if stats['skewness'] > 0 else 'izquierda'}), indicando casos extremos")
                    
                    if stats['coefficient_variation'] > 1:
                        interpretaciones.append(f"{var.title()} presenta alta variabilidad (CV={stats['coefficient_variation']:.2f}), sugiriendo gran heterogeneidad entre registros")
                    
                    if 'concentracion_top10' in stats and stats['concentracion_top10'] > 50:
                        interpretaciones.append(f"Top 10 casos de {var} concentran {stats['concentracion_top10']:.1f}% del total, indicando alta concentración")
                    
                    if stats['zeros_percentage'] > 5:
                        interpretaciones.append(f"{stats['zeros_percentage']:.1f}% de registros tienen {var}=0, requiere investigación de causas")
        
        if 'metricas_negocio' in estadisticas:
            eficiencia = estadisticas['metricas_negocio']['eficiencia']
            if eficiencia['casos_eficiencia_perfecta_pct'] > 20:
                interpretaciones.append(f"{eficiencia['casos_eficiencia_perfecta_pct']:.1f}% de casos con eficiencia perfecta sugiere posible sobre-registro o casos trasladados")
            
            if eficiencia['casos_baja_eficiencia_pct'] > 30:
                interpretaciones.append(f"{eficiencia['casos_baja_eficiencia_pct']:.1f}% de casos con baja eficiencia (<50%) indica problemas de capacidad")
            
            # Interpretación temporal
            if 'temporal' in estadisticas['metricas_negocio']:
                temporal = estadisticas['metricas_negocio']['temporal']
                if temporal['tendencia_ingresados']['es_significativa']:
                    interpretaciones.append(f"Tendencia {temporal['tendencia_ingresados']['direccion'].lower()} significativa en casos ingresados (R²={temporal['tendencia_ingresados']['r_squared']:.3f})")
        
        # Interpretar correlaciones
        if 'correlaciones' in estadisticas and estadisticas['correlaciones']:
            corr_fuertes = estadisticas['correlaciones'].get('correlaciones_fuertes_pearson', [])
            for corr in corr_fuertes[:3]:  # Top 3
                interpretaciones.append(f"Correlación {corr['fuerza'].lower()} {corr['direccion'].lower()} entre {corr['variable1']} y {corr['variable2']} (r={corr['correlacion']:.3f})")
        
        return interpretaciones
    
    def _mostrar_interpretaciones_estadisticas(self):
        """Mostrar interpretaciones de forma organizada"""
        if 'estadisticas' not in self.interpretaciones:
            return
            
        print(f"\nPrincipales interpretaciones")
        print("-" * 60)
        
        for i, interpretacion in enumerate(self.interpretaciones['estadisticas'], 1):
            print(f"{i:2d}. {interpretacion}")
    
    def detectar_patrones_y_sesgos(self) -> Dict:
        """
        Identificar patrones notables y posibles sesgos en los datos
        """
        print("\nDetectando patrones, sesgos y agrupaciones...")
        
        patrones = {}
        
        patrones['sesgos_geograficos'] = self._analizar_sesgos_geograficos()

        patrones['sesgos_temporales'] = self._analizar_sesgos_temporales()

        patrones['sesgos_materia'] = self._analizar_sesgos_materia()

        patrones['patrones_eficiencia'] = self._analizar_patrones_eficiencia()

        patrones['agrupaciones'] = self._detectar_agrupaciones()
        
        self.resultados_eda['patrones_sesgos'] = patrones
        
        self._mostrar_interpretaciones_patrones(patrones)
        
        return patrones
    
    def _analizar_sesgos_geograficos(self) -> Dict:
        """Analizar sesgos en distribución geográfica"""
        sesgos_geo = {}
 
        casos_por_depto = self.df.groupby('dpto_pjfs')['ingresado'].sum().sort_values(ascending=False)
        total_casos = casos_por_depto.sum()
        
        sesgos_geo['concentracion_departamental'] = {
            'top_3_departamentos': dict(casos_por_depto.head(3)),
            'concentracion_top_3': float(casos_por_depto.head(3).sum() / total_casos * 100),
            'coeficiente_gini': self._calcular_gini(casos_por_depto.values),
            'departamentos_bajo_volumen': int((casos_por_depto < casos_por_depto.median() * 0.1).sum())
        }
        
        # Eficiencia por región
        eficiencia_por_depto = self.df.groupby('dpto_pjfs').agg({
            'ingresado': 'sum',
            'atendido': 'sum',
            'tasa_atencion': 'mean'
        }).sort_values('tasa_atencion', ascending=False)
        
        eficiencia_por_depto['tasa_calculada'] = eficiencia_por_depto['atendido'] / eficiencia_por_depto['ingresado']
        
        sesgos_geo['disparidad_eficiencia'] = {
            'departamento_mas_eficiente': eficiencia_por_depto.index[0],
            'tasa_mas_alta': float(eficiencia_por_depto['tasa_calculada'].iloc[0]),
            'departamento_menos_eficiente': eficiencia_por_depto.index[-1],
            'tasa_mas_baja': float(eficiencia_por_depto['tasa_calculada'].iloc[-1]),
            'brecha_eficiencia': float(eficiencia_por_depto['tasa_calculada'].iloc[0] - eficiencia_por_depto['tasa_calculada'].iloc[-1])
        }
        
        return sesgos_geo
    
    def _analizar_sesgos_temporales(self) -> Dict:
        """Analizar sesgos y patrones temporales"""
        sesgos_temp = {}
        
        if self.df['anio'].nunique() <= 1:
            return {'mensaje': 'Insuficientes años para análisis temporal'}
        
        # Evolución anual
        evolucion = self.df.groupby('anio').agg({
            'ingresado': ['sum', 'mean', 'std'],
            'atendido': ['sum', 'mean', 'std'],
            'tasa_atencion': 'mean'
        }).round(3)
        
        # Calcular tendencias
        años = self.df['anio'].unique()
        casos_anuales = [self.df[self.df['anio'] == año]['ingresado'].sum() for año in sorted(años)]
        
        # Detectar años atípicos
        casos_mean = np.mean(casos_anuales)
        casos_std = np.std(casos_anuales)
        años_atipicos = []
        
        for i, (año, casos) in enumerate(zip(sorted(años), casos_anuales)):
            if abs(casos - casos_mean) > 2 * casos_std:
                años_atipicos.append({
                    'año': int(año),
                    'casos': int(casos),
                    'desviacion_estandar': float((casos - casos_mean) / casos_std),
                    'tipo': 'Alto' if casos > casos_mean else 'Bajo'
                })
        
        sesgos_temp['años_atipicos'] = años_atipicos
        sesgos_temp['coeficiente_variacion_temporal'] = float(np.std(casos_anuales) / np.mean(casos_anuales))
        sesgos_temp['tendencia_general'] = self._calcular_tendencia(pd.Series(casos_anuales))
        
        return sesgos_temp
    
    def _analizar_sesgos_materia(self) -> Dict:
        """Analizar sesgos por tipo de materia"""
        sesgos_materia = {}
        
        # Distribución por materia
        dist_materia = self.df.groupby('materia').agg({
            'ingresado': ['sum', 'count'],
            'atendido': 'sum',
            'tasa_atencion': 'mean'
        }).round(3)
        
        dist_materia.columns = ['casos_totales', 'registros', 'atendidos', 'eficiencia_promedio']
        dist_materia['participacion_pct'] = dist_materia['casos_totales'] / dist_materia['casos_totales'].sum() * 100
        
        # Identificar materias dominantes
        materia_dominante = dist_materia['participacion_pct'].idxmax()
        participacion_dominante = dist_materia['participacion_pct'].max()
        
        sesgos_materia['dominancia_materia'] = {
            'materia_dominante': materia_dominante,
            'participacion': float(participacion_dominante),
            'es_dominante': bool(participacion_dominante > 40),  # Más del 40% se considera dominante
        }
        
        # Eficiencia por materia
        eficiencias = dist_materia['eficiencia_promedio'].sort_values(ascending=False)
        sesgos_materia['disparidad_eficiencia_materia'] = {
            'materia_mas_eficiente': eficiencias.index[0],
            'eficiencia_maxima': float(eficiencias.iloc[0]),
            'materia_menos_eficiente': eficiencias.index[-1],
            'eficiencia_minima': float(eficiencias.iloc[-1]),
            'brecha_eficiencia': float(eficiencias.iloc[0] - eficiencias.iloc[-1])
        }
        
        return sesgos_materia
    
    def _analizar_patrones_eficiencia(self) -> Dict:
        """Analizar patrones de eficiencia por diferentes dimensiones"""
        patrones_ef = {}
        
        # Eficiencia por tipo de fiscalía
        ef_tipo = self.df.groupby('tipo_fiscalia').agg({
            'ingresado': 'sum',
            'atendido': 'sum',
            'tasa_atencion': ['mean', 'std', 'count']
        })
        
        ef_tipo.columns = ['ing_total', 'at_total', 'ef_promedio', 'ef_std', 'registros']
        ef_tipo['ef_calculada'] = ef_tipo['at_total'] / ef_tipo['ing_total']
        
        patrones_ef['por_tipo_fiscalia'] = ef_tipo.round(3).to_dict('index')
        
        # Eficiencia por especialización
        tiene_esp = self.df[self.df['tiene_especializacion'] == 1]['tasa_atencion'].mean()
        sin_esp = self.df[self.df['tiene_especializacion'] == 0]['tasa_atencion'].mean()
        
        patrones_ef['efecto_especializacion'] = {
            'eficiencia_con_especializacion': float(tiene_esp),
            'eficiencia_sin_especializacion': float(sin_esp),
            'diferencia': float(tiene_esp - sin_esp),
            'especializacion_mejora_eficiencia': bool(tiene_esp > sin_esp)
        }
        
        # Identificar registros con patrones anómalos
        anomalias = self.df[
            (self.df['atendido'] > self.df['ingresado'] * 1.5) |  # Sobre-atención extrema
            ((self.df['ingresado'] > 0) & (self.df['atendido'] == 0))  # Sin atención con casos ingresados
        ]
        
        patrones_ef['anomalias_identificadas'] = {
            'total_anomalias': len(anomalias),
            'porcentaje_dataset': float(len(anomalias) / len(self.df) * 100),
            'tipos_anomalia': {
                'sobre_atencion_extrema': int(((anomalias['atendido'] > anomalias['ingresado'] * 1.5)).sum()),
                'sin_atencion_con_casos': int(((anomalias['ingresado'] > 0) & (anomalias['atendido'] == 0)).sum())
            }
        }
        
        return patrones_ef
    
    def _detectar_agrupaciones(self) -> Dict:
        """Detectar agrupaciones naturales en los datos"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Preparar datos para clustering
        variables_clustering = ['ingresado', 'atendido', 'tasa_atencion']
        datos_clustering = self.df[variables_clustering].dropna()
        
        if len(datos_clustering) < 10:
            return {'mensaje': 'Insuficientes datos para clustering'}
        
        scaler = StandardScaler()
        datos_norm = scaler.fit_transform(datos_clustering)
        
        from sklearn.metrics import silhouette_score
        
        mejor_k = 3
        mejor_score = -1
        
        for k in range(2, min(8, len(datos_clustering)//10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(datos_norm)
            score = silhouette_score(datos_norm, labels)
            
            if score > mejor_score:
                mejor_score = score
                mejor_k = k
        
        # Aplicar clustering óptimo
        kmeans_final = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
        clusters = kmeans_final.fit_predict(datos_norm)
        
        # Analizar características de cada cluster
        datos_clustering['cluster'] = clusters
        caracteristicas_clusters = datos_clustering.groupby('cluster')[variables_clustering].agg(['mean', 'std', 'count']).round(3)
        
        # Interpretar clusters
        interpretacion_clusters = {}
        for cluster_id in range(mejor_k):
            cluster_data = datos_clustering[datos_clustering['cluster'] == cluster_id]
            
            if cluster_data['ingresado'].mean() > datos_clustering['ingresado'].quantile(0.75):
                tipo_volumen = "Alto volumen"
            elif cluster_data['ingresado'].mean() < datos_clustering['ingresado'].quantile(0.25):
                tipo_volumen = "Bajo volumen"
            else:
                tipo_volumen = "Volumen medio"
            
            if cluster_data['tasa_atencion'].mean() > 0.9:
                tipo_eficiencia = "Alta eficiencia"
            elif cluster_data['tasa_atencion'].mean() > 0.7:
                tipo_eficiencia = "Eficiencia media"
            else:
                tipo_eficiencia = "Baja eficiencia"
            
            interpretacion_clusters[f'Cluster_{cluster_id}'] = {
                'tamaño': len(cluster_data),
                'porcentaje': float(len(cluster_data) / len(datos_clustering) * 100),
                'caracteristica_volumen': tipo_volumen,
                'caracteristica_eficiencia': tipo_eficiencia,
                'casos_promedio_ingresado': float(cluster_data['ingresado'].mean()),
                'eficiencia_promedio': float(cluster_data['tasa_atencion'].mean())
            }
        
        return {
            'numero_clusters_optimo': mejor_k,
            'silhouette_score': float(mejor_score),
            'caracteristicas_clusters': caracteristicas_clusters.to_dict(),
            'interpretacion_clusters': interpretacion_clusters
        }
    
    def _mostrar_interpretaciones_patrones(self, patrones: Dict):
        print(f"\nPrincipales patrones y sesgos detectados")
        print("-" * 60)
        
        # Sesgos geográficos
        if 'sesgos_geograficos' in patrones:
            geo = patrones['sesgos_geograficos']
            conc = geo['concentracion_departamental']
            print(f"\nSesgos Geográficos:")
            print(f"  -Top 3 departamentos concentran {conc['concentracion_top_3']:.1f}% de casos")
            print(f"  -Coeficiente de Gini: {conc['coeficiente_gini']:.3f} ({'Alta' if conc['coeficiente_gini'] > 0.5 else 'Moderada'} concentración)")
            
            if 'disparidad_eficiencia' in geo:
                disp = geo['disparidad_eficiencia']
                print(f"  -Brecha de eficiencia entre departamentos: {disp['brecha_eficiencia']:.3f}")
                print(f"  -{disp['departamento_mas_eficiente']} (más eficiente) vs {disp['departamento_menos_eficiente']} (menos eficiente)")
        
        # Sesgos por materia
        if 'sesgos_materia' in patrones:
            materia = patrones['sesgos_materia']
            if 'dominancia_materia' in materia:
                dom = materia['dominancia_materia']
                print(f"\nSesgos por Materia:")
                print(f"  -{dom['materia_dominante']} domina con {dom['participacion']:.1f}% de casos")
                print(f"  -{'Existe' if dom['es_dominante'] else 'No existe'} dominancia clara de una materia")
    
        if 'patrones_eficiencia' in patrones:
            ef = patrones['patrones_eficiencia']
            if 'efecto_especializacion' in ef:
                esp = ef['efecto_especializacion']
                print(f"\nPatrones de Eficiencia:")
                print(f"  -Especialización {'mejora' if esp['especializacion_mejora_eficiencia'] else 'no mejora'} eficiencia")
                print(f"  -Diferencia: {esp['diferencia']:+.3f} puntos de eficiencia")
            
            if 'anomalias_identificadas' in ef:
                anom = ef['anomalias_identificadas']
                print(f"  -{anom['total_anomalias']} registros anómalos ({anom['porcentaje_dataset']:.2f}% del dataset)")
        
        # Agrupaciones
        if 'agrupaciones' in patrones and 'numero_clusters_optimo' in patrones['agrupaciones']:
            agrup = patrones['agrupaciones']
            print(f"\nAgrupaciones Naturales:")
            print(f"  -{agrup['numero_clusters_optimo']} grupos naturales identificados")
            print(f"  -Calidad de agrupación (Silhouette): {agrup['silhouette_score']:.3f}")
            
            if 'interpretacion_clusters' in agrup:
                for cluster_name, cluster_info in agrup['interpretacion_clusters'].items():
                    print(f"  -{cluster_name}: {cluster_info['caracteristica_volumen']}, {cluster_info['caracteristica_eficiencia']} ({cluster_info['porcentaje']:.1f}% casos)")
    
    def detectar_outliers_mejorado(self) -> Dict:
        
        print("\nDetectando outliers con análisis contextual...")
        
        outliers = {}
        
        # Solo analizar variables numéricas reales
        for col in self.columnas_numericas_reales:
            outliers[col] = self._detectar_outliers_variable(col)
        
        # Outliers multivariados
        outliers['multivariado'] = self._detectar_outliers_multivariados()
        
        # Outliers contextales (específicos del dominio)
        outliers['contextuales'] = self._detectar_outliers_contextuales()
        
        self.resultados_eda['outliers'] = outliers
        self._mostrar_interpretaciones_outliers(outliers)
        
        return outliers
    
    def _detectar_outliers_variable(self, columna: str) -> Dict:
        serie = self.df[columna].dropna()
        
        if len(serie) < 10:
            return {'error': 'Insuficientes datos'}
        
        resultado = {'total_valores': len(serie), 'metodos': {}}
        
        # Método IQR
        Q1, Q3 = serie.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        
        outliers_iqr = serie[(serie < limite_inf) | (serie > limite_sup)]
        
        resultado['metodos']['IQR'] = {
            'count': len(outliers_iqr),
            'percentage': len(outliers_iqr) / len(serie) * 100,
            'valores_extremos': outliers_iqr.nlargest(5).tolist() if len(outliers_iqr) > 0 else [],
            'interpretacion': self._interpretar_outliers_iqr(columna, len(outliers_iqr), len(serie))
        }
        
        # Método Z-Score 
        median = serie.median()
        mad = np.median(np.abs(serie - median))
        if mad > 0:
            modified_z = 0.6745 * (serie - median) / mad
            outliers_zscore = serie[np.abs(modified_z) > 3.5]
            
            resultado['metodos']['Z_Score_Modificado'] = {
                'count': len(outliers_zscore),
                'percentage': len(outliers_zscore) / len(serie) * 100,
                'valores_extremos': outliers_zscore.nlargest(5).tolist() if len(outliers_zscore) > 0 else [],
                'interpretacion': self._interpretar_outliers_zscore(columna, len(outliers_zscore), len(serie))
            }
        
        return resultado
    
    def _detectar_outliers_multivariados(self) -> Dict:
        from sklearn.ensemble import IsolationForest
        
        # Usar variables principales
        variables = ['ingresado', 'atendido', 'tasa_atencion']
        datos = self.df[variables].dropna()
        
        if len(datos) < 50:
            return {'error': 'Insuficientes datos para análisis multivariado'}
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% de outliers esperados
        outliers_pred = iso_forest.fit_predict(datos)
        
        outliers_indices = datos.index[outliers_pred == -1]
        outliers_data = self.df.loc[outliers_indices]
        
        return {
            'total_outliers': len(outliers_indices),
            'porcentaje': len(outliers_indices) / len(datos) * 100,
            'caracteristicas_outliers': {
                'ingresado_promedio': float(outliers_data['ingresado'].mean()),
                'atendido_promedio': float(outliers_data['atendido'].mean()),
                'tasa_atencion_promedio': float(outliers_data['tasa_atencion'].mean()),
                'casos_extremos': outliers_data.nlargest(3, 'ingresado')[['distrito_fiscal', 'materia', 'ingresado', 'atendido']].to_dict('records')
            },
            'interpretacion': 'Casos con combinaciones atípicas de volumen y eficiencia que requieren revisión individual'
        }
    
    def _detectar_outliers_contextuales(self) -> Dict:
        """Detectar outliers específicos del contexto judicial"""
        outliers_contextuales = {}
        
        # Casos con eficiencia imposible (más atendidos que ingresados de forma extrema)
        sobre_atencion_extrema = self.df[self.df['atendido'] > self.df['ingresado'] * 2]
        
        outliers_contextuales['sobre_atencion_extrema'] = {
            'count': len(sobre_atencion_extrema),
            'casos': sobre_atencion_extrema[['distrito_fiscal', 'materia', 'ingresado', 'atendido', 'tasa_atencion']].nlargest(5, 'tasa_atencion').to_dict('records'),
            'interpretacion': 'Posibles errores de registro o casos trasladados de períodos anteriores'
        }
        
        # Casos con volumen extremadamente alto
        percentil_99 = self.df['ingresado'].quantile(0.99)
        volumen_extremo = self.df[self.df['ingresado'] > percentil_99]
        
        outliers_contextuales['volumen_extremo'] = {
            'count': len(volumen_extremo),
            'umbral': float(percentil_99),
            'casos': volumen_extremo[['distrito_fiscal', 'materia', 'ingresado', 'atendido']].nlargest(5, 'ingresado').to_dict('records'),
            'interpretacion': 'Distritos o materias con carga de trabajo excepcional que pueden requerir recursos adicionales'
        }
        
        # 3. Casos sin atención con ingreso alto
        sin_atencion_alto_ingreso = self.df[(self.df['atendido'] == 0) & (self.df['ingresado'] > self.df['ingresado'].median())]
        
        outliers_contextuales['sin_atencion_alto_volumen'] = {
            'count': len(sin_atencion_alto_ingreso),
            'casos': sin_atencion_alto_ingreso[['distrito_fiscal', 'materia', 'ingresado', 'especializada']].nlargest(5, 'ingresado').to_dict('records'),
            'interpretacion': 'Posibles problemas operativos o administrativos que impiden el procesamiento de casos'
        }
        
        return outliers_contextuales
    
    def _interpretar_outliers_iqr(self, columna: str, num_outliers: int, total: int) -> str:
        """Interpretar outliers detectados por IQR según la variable"""
        porcentaje = num_outliers / total * 100
        
        if columna in ['ingresado', 'atendido']:
            if porcentaje > 10:
                return f"Alta presencia de casos extremos ({porcentaje:.1f}%) sugiere gran heterogeneidad en carga de trabajo"
            elif porcentaje > 5:
                return f"Presencia moderada de casos extremos ({porcentaje:.1f}%) indica algunos distritos/materias con volumen excepcional"
            else:
                return f"Pocos casos extremos ({porcentaje:.1f}%) indica distribución relativamente homogénea"
        else:
            return f"{porcentaje:.1f}% de valores atípicos detectados"
    
    def _interpretar_outliers_zscore(self, columna: str, num_outliers: int, total: int) -> str:
        """Interpretar outliers detectados por Z-Score modificado"""
        porcentaje = num_outliers / total * 100
        
        if porcentaje > 15:
            return "Muchos valores extremos - posible problema de calidad de datos o gran variabilidad natural"
        elif porcentaje > 5:
            return "Algunos valores extremos - revisar casos para identificar patrones o errores"
        else:
            return "Pocos valores extremos - dentro del rango esperado"
    
    def _mostrar_interpretaciones_outliers(self, outliers: Dict):
        """Mostrar interpretaciones de outliers encontrados"""
        print(f"\nPrincipales hallazgos de outliers")
        print("=" * 60)
        
        for variable, resultado in outliers.items():
            if variable == 'multivariado':
                if 'error' not in resultado:
                    print(f"\nOutliers Multivariados:")
                    print(f"  -{resultado['total_outliers']} casos ({resultado['porcentaje']:.2f}%) con patrones atípicos")
                    print(f"  -Interpretación: {resultado['interpretacion']}")
                continue
                
            if variable == 'contextuales':
                print(f"\nOutliers Contextuales del Sistema Judicial:")
                for tipo, datos in resultado.items():
                    print(f"  -{tipo.replace('_', ' ').title()}: {datos['count']} casos")
                    print(f"    --{datos['interpretacion']}")
                continue
            
            if 'error' in resultado:
                continue
                
            print(f"\n{variable.title()}:")
            print(f"  Total valores analizados: {resultado['total_valores']:,}")
            
            for metodo, datos in resultado['metodos'].items():
                print(f"  -{metodo}: {datos['count']} outliers ({datos['percentage']:.2f}%)")
                print(f"    --{datos['interpretacion']}")
                if datos['valores_extremos']:
                    print(f"    --Valores extremos: {datos['valores_extremos']}")
    
    def generar_visualizaciones_principales(self):
        """
        Generar visualizaciones principales con interpretaciones
        """
        print("\nGenerando visualizaciones principales con interpretaciones")
        
        self._crear_dashboard_metricas()
        
        self._crear_visualizaciones_distribuciones()
        
        self._crear_visualizaciones_correlaciones()
        
        self._crear_visualizaciones_geograficas()
        
        self._crear_visualizaciones_temporales()
        
        self._crear_visualizaciones_eficiencia()
        
        self._crear_visualizaciones_outliers()
    
    def _crear_dashboard_metricas(self):
        """Crear dashboard con métricas principales"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dashboard de Métricas Principales - Sistema Judicial', fontsize=16, fontweight='bold')
        
        # Distribución de casos ingresados
        axes[0,0].hist(self.df['ingresado'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribución de Casos Ingresados')
        axes[0,0].set_xlabel('Casos Ingresados')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].axvline(self.df['ingresado'].median(), color='red', linestyle='--', label=f'Mediana: {self.df["ingresado"].median():.0f}')
        axes[0,0].legend()
        
        # Distribución de casos atendidos
        axes[0,1].hist(self.df['atendido'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Distribución de Casos Atendidos')
        axes[0,1].set_xlabel('Casos Atendidos')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].axvline(self.df['atendido'].median(), color='red', linestyle='--', label=f'Mediana: {self.df["atendido"].median():.0f}')
        axes[0,1].legend()
        
        # Distribución de tasa de atención
        tasa_limpia = self.df['tasa_atencion'].dropna()
        tasa_limitada = tasa_limpia[tasa_limpia <= 3]  # Limitar para visualización
        
        axes[0,2].hist(tasa_limitada, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0,2].set_title('Distribución de Tasa de Atención')
        axes[0,2].set_xlabel('Tasa de Atención')
        axes[0,2].set_ylabel('Frecuencia')
        axes[0,2].axvline(1.0, color='red', linestyle='-', linewidth=2, label='Eficiencia Perfecta (1.0)')
        axes[0,2].axvline(tasa_limitada.mean(), color='blue', linestyle='--', label=f'Media: {tasa_limitada.mean():.2f}')
        axes[0,2].legend()
        
        # Casos por materia
        materia_counts = self.df['materia'].value_counts()
        axes[1,0].barh(materia_counts.index, materia_counts.values, color='lightcoral')
        axes[1,0].set_title('Distribución por Materia')
        axes[1,0].set_xlabel('Número de Registros')
        
        # Añadir porcentajes
        total = materia_counts.sum()
        for i, v in enumerate(materia_counts.values):
            axes[1,0].text(v + total*0.01, i, f'{v/total*100:.1f}%', va='center')
        
        # Casos por tipo de fiscalía
        tipo_counts = self.df['tipo_fiscalia'].value_counts()
        wedges, texts, autotexts = axes[1,1].pie(tipo_counts.values, labels=tipo_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Distribución por Tipo de Fiscalía')
        
        # Top 10 distritos por volumen
        top_distritos = self.df.groupby('distrito_fiscal')['ingresado'].sum().nlargest(10)
        axes[1,2].barh(range(len(top_distritos)), top_distritos.values, color='gold')
        axes[1,2].set_yticks(range(len(top_distritos)))
        axes[1,2].set_yticklabels(top_distritos.index, fontsize=9)
        axes[1,2].set_title('Top 10 Distritos por Casos Ingresados')
        axes[1,2].set_xlabel('Casos Ingresados Totales')
        
        plt.tight_layout()

        self._guardar_figura(fig, "dashboard_metricas_principales")

        plt.show()
        
        # Interpretaciones
        print("\nInterpretaciones de las graficas:")
        print(f"1. Mediana de casos ingresados: {self.df['ingresado'].median():.0f} - indica volumen típico por registro")
        print(f"2. Tasa de atención promedio: {tasa_limitada.mean():.2f} - {'por encima' if tasa_limitada.mean() > 1 else 'por debajo'} de la eficiencia perfecta")
        print(f"3. Materia dominante: {materia_counts.index[0]} ({materia_counts.iloc[0]/total*100:.1f}% de registros)")
        print(f"4. Concentración geográfica: Top 10 distritos concentran {top_distritos.sum()/self.df['ingresado'].sum()*100:.1f}% de casos")
    
    def _crear_visualizaciones_distribuciones(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis de Distribuciones y Detección de Outliers', fontsize=16, fontweight='bold')
        
        # Variables numéricas para analizar
        vars_numericas = ['ingresado', 'atendido', 'tasa_atencion']
        
        for i, var in enumerate(vars_numericas):
            # Histograma con KDE
            if var == 'tasa_atencion':
                data_plot = self.df[var].dropna()
                data_plot = data_plot[data_plot <= 3]  # Limitar para visualización
            else:
                data_plot = self.df[var]
            
            axes[0, i].hist(data_plot.dropna(), bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black')
            
            # Añadir KDE
            from scipy.stats import gaussian_kde
            if len(data_plot.dropna()) > 1:
                kde_data = data_plot.dropna()
                kde = gaussian_kde(kde_data)
                x_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                axes[0, i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            axes[0, i].set_title(f'Distribución de {var.replace("_", " ").title()}')
            axes[0, i].set_xlabel(var.replace("_", " ").title())
            axes[0, i].set_ylabel('Densidad')
            axes[0, i].legend()
            
            # Boxplot
            bp = axes[1, i].boxplot(data_plot.dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][0].set_alpha(0.7)
            
            axes[1, i].set_title(f'Boxplot - {var.replace("_", " ").title()}')
            axes[1, i].set_ylabel(var.replace("_", " ").title())
            
            # Añadir estadísticas al boxplot
            Q1 = data_plot.quantile(0.25)
            Q3 = data_plot.quantile(0.75)
            IQR = Q3 - Q1
            outliers_count = len(data_plot[(data_plot < Q1 - 1.5*IQR) | (data_plot > Q3 + 1.5*IQR)])
            
            axes[1, i].text(1.1, data_plot.median(), f'Outliers: {outliers_count}\n({outliers_count/len(data_plot)*100:.1f}%)', 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        self._guardar_figura(fig, "Distribuciones_boxplot_KDE")
        plt.show()
        
        # Interpretaciones de distribuciones
        print("\nInterpretaciones de Distribuciones:")
        for var in vars_numericas:
            if var == 'tasa_atencion':
                data_analysis = self.df[var].dropna()
            else:
                data_analysis = self.df[var]
                
            skewness = stats.skew(data_analysis.dropna())
            kurtosis = stats.kurtosis(data_analysis.dropna())
            
            print(f"\n{var.replace('_', ' ').title()}:")
            print(f"  -Asimetría: {skewness:.3f} ({'Derecha' if skewness > 0.5 else 'Izquierda' if skewness < -0.5 else 'Simétrica'})")
            print(f"  -Curtosis: {kurtosis:.3f} ({'Leptocúrtica' if kurtosis > 3 else 'Platicúrtica' if kurtosis < -3 else 'Mesocúrtica'})")
            
            # Interpretación específica
            if var == 'ingresado':
                if skewness > 1:
                    print(f"  -Interpretación: Pocos distritos/materias manejan volúmenes muy altos")
            elif var == 'atendido':
                if skewness > 1:
                    print(f"  -Interpretación: Concentración en pocos casos de alta productividad")
            elif var == 'tasa_atencion':
                casos_sobreeficientes = (data_analysis > 1.2).sum()
                print(f"  -{casos_sobreeficientes} casos ({casos_sobreeficientes/len(data_analysis)*100:.1f}%) con sobre-eficiencia >20%")
    
    def _crear_visualizaciones_correlaciones(self):
        # Variables para correlación
        vars_corr = ['ingresado', 'atendido', 'tasa_atencion', 'casos_pendientes']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Matriz de correlación Pearson
        corr_pearson = self.df[vars_corr].corr()
        
        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
        
        # Heatmap Pearson
        sns.heatmap(corr_pearson, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=axes[0])
        axes[0].set_title('Matriz de Correlación Pearson')
        
        # Matriz de correlación Spearman (más robusta a outliers)
        corr_spearman = self.df[vars_corr].corr(method='spearman')
        
        sns.heatmap(corr_spearman, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=axes[1])
        axes[1].set_title('Matriz de Correlación Spearman')
        
        plt.tight_layout()
        self._guardar_figura(fig, "Matriz_Correlación ")
        plt.show()
        
        print("\nInterpretaciones de Correlaciones:")
        corr_ingresado_atendido = corr_pearson.loc['ingresado', 'atendido']
        print(f"-Ingresado vs Atendido (Pearson): {corr_ingresado_atendido:.3f}")
        
        if corr_ingresado_atendido > 0.800:
            print("  --Correlación muy fuerte: el sistema mantiene proporcionalidad entre casos ingresados y atendidos")
        elif corr_ingresado_atendido > 0.600:
            print("  --Correlación fuerte: existe buena capacidad de respuesta del sistema")
        else:
            print("  --Correlación moderada/débil: posibles problemas de capacidad o eficiencia variable")
        
        # Analizar correlación con tasa de atención
        corr_tasa_ingresado = corr_pearson.loc['tasa_atencion', 'ingresado']
        print(f"- Tasa Atención vs Ingresado: {corr_tasa_ingresado:.3f}")
        
        if corr_tasa_ingresado < -0.3:
            print("  --Correlación negativa: a mayor volumen, menor eficiencia (saturación del sistema)")
        elif corr_tasa_ingresado > 0.3:
            print("  --Correlación positiva: mayor volumen se asocia con mayor eficiencia (economías de escala)")
        else:
            print("  --Sin correlación clara: la eficiencia no depende linealmente del volumen")
    
    def _crear_visualizaciones_geograficas(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Geográfico del Sistema Judicial', fontsize=16, fontweight='bold')
        
        # Top 15 distritos por casos ingresados
        top_distritos = self.df.groupby('distrito_fiscal')['ingresado'].sum().nlargest(15)
        
        axes[0,0].barh(range(len(top_distritos)), top_distritos.values, color='steelblue')
        axes[0,0].set_yticks(range(len(top_distritos)))
        axes[0,0].set_yticklabels(top_distritos.index, fontsize=9)
        axes[0,0].set_title('Top 15 Distritos - Casos Ingresados')
        axes[0,0].set_xlabel('Total Casos Ingresados')
        
        # Eficiencia por departamento
        eficiencia_depto = self.df.groupby('dpto_pjfs').agg({
            'ingresado': 'sum',
            'atendido': 'sum'
        })
        eficiencia_depto['tasa_atencion'] = eficiencia_depto['atendido'] / eficiencia_depto['ingresado']
        eficiencia_depto = eficiencia_depto.sort_values('tasa_atencion', ascending=True)
        
        colors = ['red' if x < 0.8 else 'yellow' if x < 0.95 else 'green' for x in eficiencia_depto['tasa_atencion']]
        
        axes[0,1].barh(range(len(eficiencia_depto)), eficiencia_depto['tasa_atencion'], color=colors)
        axes[0,1].set_yticks(range(len(eficiencia_depto)))
        axes[0,1].set_yticklabels(eficiencia_depto.index, fontsize=9)
        axes[0,1].set_title('Eficiencia por Departamento')
        axes[0,1].set_xlabel('Tasa de Atención')
        axes[0,1].axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[0,1].legend()
        
        # 3. Distribución de casos por departamento
        casos_depto = self.df.groupby('dpto_pjfs')['ingresado'].sum().sort_values(ascending=False)
        
        axes[1,0].pie(casos_depto.head(10).values, 
                     labels=casos_depto.head(10).index, 
                     autopct='%1.1f%%', 
                     startangle=90)
        axes[1,0].set_title('Top 10 Departamentos - Distribución de Casos')
        
        # 4. Scatter: Volumen vs Eficiencia por distrito
        distritos_stats = self.df.groupby('distrito_fiscal').agg({
            'ingresado': 'sum',
            'atendido': 'sum'
        })
        distritos_stats['tasa_atencion'] = distritos_stats['atendido'] / distritos_stats['ingresado']
        
        scatter = axes[1,1].scatter(distritos_stats['ingresado'], 
                                   distritos_stats['tasa_atencion'],
                                   alpha=0.6, s=60, c='purple')
        
        axes[1,1].set_xlabel('Total Casos Ingresados')
        axes[1,1].set_ylabel('Tasa de Atención')
        axes[1,1].set_title('Volumen vs Eficiencia por Distrito')
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[1,1].legend()
        
        # Línea de tendencia
        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(distritos_stats['ingresado'], distritos_stats['tasa_atencion'])
        line = slope * distritos_stats['ingresado'] + intercept
        axes[1,1].plot(distritos_stats['ingresado'], line, 'r--', alpha=0.8, label=f'Tendencia (R²={r_value**2:.3f})')
        axes[1,1].legend()
        
        plt.tight_layout()
        self._guardar_figura(fig, "Análisis_Geográfico")
        plt.show()
        
        print("\nInterpretaciones Geográficas:")
        
        # Concentración geográfica
        concentracion_top5 = top_distritos.head(5).sum() / self.df['ingresado'].sum() * 100
        print(f"-Top 5 distritos concentran {concentracion_top5:.1f}% de todos los casos")
        
        if concentracion_top5 > 50:
            print("  --Alta concentración geográfica sugiere necesidad de redistribución de recursos")
        
        # Disparidad de eficiencia
        ef_max = eficiencia_depto['tasa_atencion'].max()
        ef_min = eficiencia_depto['tasa_atencion'].min()
        print(f"-Brecha de eficiencia entre departamentos: {ef_max - ef_min:.3f}")
        print(f"-Departamento más eficiente: {eficiencia_depto.index[-1]} ({ef_max:.3f})")
        print(f"-Departamento menos eficiente: {eficiencia_depto.index[0]} ({ef_min:.3f})")
        
        # Correlación volumen-eficiencia
        if abs(r_value) > 0.3:
            direccion = "positiva" if r_value > 0 else "negativa"
            print(f"- Correlación {direccion} significativa entre volumen y eficiencia (R²={r_value**2:.3f})")
            
            if r_value < -0.3:
                print("  --A mayor volumen, menor eficiencia: indicios de saturación del sistema")
            elif r_value > 0.3:
                print("  --A mayor volumen, mayor eficiencia: posibles economías de escala")
    
    def _crear_visualizaciones_temporales(self):
        if self.df['anio'].nunique() <= 1:
            print("Insuficientes años para análisis temporal")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Temporal del Sistema Judicial', fontsize=16, fontweight='bold')
        
        # 1. Evolución temporal de casos totales
        evolucion = self.df.groupby('anio').agg({
            'ingresado': 'sum',
            'atendido': 'sum'
        })
        
        axes[0,0].plot(evolucion.index, evolucion['ingresado'], 'b-o', linewidth=2, markersize=8, label='Ingresados')
        axes[0,0].plot(evolucion.index, evolucion['atendido'], 'r-s', linewidth=2, markersize=8, label='Atendidos')
        axes[0,0].set_title('Evolución Temporal - Casos Totales')
        axes[0,0].set_xlabel('Año')
        axes[0,0].set_ylabel('Número de Casos')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Evolución de la tasa de atención
        evolucion['tasa_global'] = evolucion['atendido'] / evolucion['ingresado']
        
        axes[0,1].plot(evolucion.index, evolucion['tasa_global'], 'g-^', linewidth=3, markersize=8)
        axes[0,1].set_title('Evolución de la Tasa de Atención Global')
        axes[0,1].set_xlabel('Año')
        axes[0,1].set_ylabel('Tasa de Atención')
        axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Evolución por materia
        evolucion_materia = self.df.groupby(['anio', 'materia'])['ingresado'].sum().unstack(fill_value=0)
        
        for materia in evolucion_materia.columns:
            axes[1,0].plot(evolucion_materia.index, evolucion_materia[materia], 'o-', linewidth=2, label=materia)
        
        axes[1,0].set_title('Evolución por Materia')
        axes[1,0].set_xlabel('Año')
        axes[1,0].set_ylabel('Casos Ingresados')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Variación año a año
        variacion = evolucion[['ingresado', 'atendido']].pct_change() * 100
        variacion = variacion.dropna()
        
        x = np.arange(len(variacion))
        width = 0.35
        
        axes[1,1].bar(x - width/2, variacion['ingresado'], width, label='Ingresados', alpha=0.8)
        axes[1,1].bar(x + width/2, variacion['atendido'], width, label='Atendidos', alpha=0.8)
        
        axes[1,1].set_title('Variación Porcentual Anual')
        axes[1,1].set_xlabel('Año')
        axes[1,1].set_ylabel('Variación (%)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(variacion.index)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._guardar_figura(fig, "Análisis_Temporal")
        plt.show()
        
        # Interpretaciones temporales
        print("\nInterpretaciones Temporales:")
        
        # Tendencia general
        años = sorted(evolucion.index)
        casos_inicial = evolucion.loc[años[0], 'ingresado']
        casos_final = evolucion.loc[años[-1], 'ingresado']
        crecimiento_total = (casos_final - casos_inicial) / casos_inicial * 100
        
        print(f"- Crecimiento total del período: {crecimiento_total:+.1f}%")
        
        if abs(crecimiento_total) > 20:
            direccion = "fuerte crecimiento" if crecimiento_total > 0 else "fuerte reducción"
            print(f"  --{direccion} en la carga de trabajo del sistema judicial")
        
        # Análisis de la eficiencia temporal
        ef_inicial = evolucion.loc[años[0], 'tasa_global']
        ef_final = evolucion.loc[años[-1], 'tasa_global']
        cambio_eficiencia = ef_final - ef_inicial
        
        print(f"- Cambio en eficiencia: {cambio_eficiencia:+.3f} puntos")
        
        if abs(cambio_eficiencia) > 0.1:
            direccion = "mejora" if cambio_eficiencia > 0 else "deterioro"
            print(f"  --{direccion} significativa en la eficiencia del sistema")
        
        # Variabilidad temporal
        cv_temporal = evolucion['ingresado'].std() / evolucion['ingresado'].mean()
        print(f"- Variabilidad temporal (CV): {cv_temporal:.3f}")
        
        if cv_temporal > 0.15:
            print("  --Alta variabilidad entre años sugiere factores externos o cambios estructurales")
    
    def _crear_visualizaciones_eficiencia(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Detallado de Eficiencia del Sistema', fontsize=16, fontweight='bold')
        
        # Distribución de eficiencia por categorías
        categorias_eficiencia = pd.cut(self.df['tasa_atencion'].dropna(), 
                                      bins=[0, 0.5, 0.8, 1.0, 1.2, float('inf')],
                                      labels=['Muy Baja (<0.5)', 'Baja (0.5-0.8)', 'Buena (0.8-1.0)', 'Excelente (1.0-1.2)', 'Excepcional (>1.2)'])
        
        cat_counts = categorias_eficiencia.value_counts()
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        axes[0,0].pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Distribución de Categorías de Eficiencia')
        
        # Eficiencia por materia
        ef_materia = self.df.groupby('materia').agg({
            'ingresado': 'sum',
            'atendido': 'sum',
            'tasa_atencion': ['mean', 'std', 'count']
        })
        
        ef_materia.columns = ['ing_total', 'at_total', 'ef_media', 'ef_std', 'registros']
        ef_materia['ef_calculada'] = ef_materia['at_total'] / ef_materia['ing_total']
        
        # Ordenar por eficiencia calculada
        ef_materia = ef_materia.sort_values('ef_calculada')
        
        # Usar colores según nivel de eficiencia
        colors = ['red' if x < 0.8 else 'yellow' if x < 1 else 'green' for x in ef_materia['ef_calculada']]
        
        axes[0,1].barh(ef_materia.index, ef_materia['ef_calculada'], color=colors)
        axes[0,1].set_title('Eficiencia por Materia')
        axes[0,1].set_xlabel('Tasa de Atención Calculada')
        axes[0,1].axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[0,1].legend()
        
        # Scatter: Relación entre volumen y eficiencia
        axes[1,0].scatter(self.df['ingresado'], self.df['tasa_atencion'], 
                         alpha=0.3, s=30, c='blue')
        
        # Añadir línea de tendencia
        from scipy import stats as stats_scipy
        mask = ~np.isnan(self.df['tasa_atencion'])
        if mask.sum() > 1:  # Asegurar que hay al menos dos puntos para la regresión
            slope, intercept, r_value, p_value, std_err = stats_scipy.linregress(
                self.df.loc[mask, 'ingresado'], 
                self.df.loc[mask, 'tasa_atencion']
            )
            
            x_range = np.linspace(0, self.df['ingresado'].max(), 100)
            y_pred = slope * x_range + intercept
            
            axes[1,0].plot(x_range, y_pred, 'r--', linewidth=2, 
                          label=f'Tendencia (R²={r_value**2:.3f})')
        
        axes[1,0].set_title('Relación entre Volumen y Eficiencia')
        axes[1,0].set_xlabel('Casos Ingresados')
        axes[1,0].set_ylabel('Tasa de Atención')
        axes[1,0].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[1,0].legend()
        
        # 4. Comparación de eficiencia por tipo de fiscalía
        ef_tipo = self.df.groupby('tipo_fiscalia').agg({
            'ingresado': 'sum',
            'atendido': 'sum'
        })
        
        ef_tipo['tasa_atencion'] = ef_tipo['atendido'] / ef_tipo['ingresado']
        
        # Crear barras con error para mostrar variabilidad
        error_tipo = self.df.groupby('tipo_fiscalia')['tasa_atencion'].std()
        
        axes[1,1].bar(ef_tipo.index, ef_tipo['tasa_atencion'], 
                     yerr=error_tipo, capsize=10, alpha=0.7,
                     color=['skyblue', 'orange', 'lightgreen'])
        
        axes[1,1].set_title('Eficiencia por Tipo de Fiscalía')
        axes[1,1].set_ylabel('Tasa de Atención')
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Eficiencia Perfecta')
        axes[1,1].legend()
        
        plt.tight_layout()
        self._guardar_figura(fig, "Análisis_Eficiencia")
        plt.show()
        
        # Interpretaciones de eficiencia
        print("\nInterpretaciones de Eficiencia:")
        
        # Categorías de eficiencia
        pct_baja_ef = (cat_counts.get('Muy Baja (<0.5)', 0) + cat_counts.get('Baja (0.5-0.8)', 0)) / cat_counts.sum() * 100
        pct_buena = cat_counts.get('Buena (0.8-1.0)', 0) / cat_counts.sum() * 100
        pct_perfecta = (cat_counts.get('Excelente (1.0-1.2)', 0) + cat_counts.get('Excepcional (>1.2)', 0)) / cat_counts.sum() * 100
    
        print(f"- {pct_baja_ef:.1f}% de casos tienen eficiencia baja o muy baja (<0.8)")
        print(f"- {pct_buena:.1f}% de casos tienen eficiencia buena (0.8-1.0)")
        print(f"- {pct_perfecta:.1f}% de casos tienen eficiencia excelente o excepcional (>1.0)")
        
        # Eficiencia por materia
        materia_max_ef = ef_materia['ef_calculada'].idxmax()
        materia_min_ef = ef_materia['ef_calculada'].idxmin()
        
        print(f"- Materia más eficiente: {materia_max_ef} ({ef_materia.loc[materia_max_ef, 'ef_calculada']:.3f})")
        print(f"- Materia menos eficiente: {materia_min_ef} ({ef_materia.loc[materia_min_ef, 'ef_calculada']:.3f})")
        
        # Relación volumen-eficiencia
        if mask.sum() > 1:
            if abs(r_value) > 0.2:
                tendencia = "positiva" if slope > 0 else "negativa"
                print(f"- Tendencia {tendencia} entre volumen y eficiencia (R²={r_value**2:.3f})")
                if slope < 0:
                    print("  --A mayor volumen de casos, menor eficiencia → posible sobrecarga del sistema")
                else:
                    print("  --A mayor volumen de casos, mayor eficiencia → posibles economías de escala")
            else:
                print("- No hay relación clara entre volumen y eficiencia (R² muy bajo)")

    def _crear_visualizaciones_outliers(self):
        if 'outliers' not in self.resultados_eda:
            print("Debe ejecutar detectar_outliers_mejorado() primero")
            return
        
        outliers = self.resultados_eda['outliers']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Outliers y Casos Extremos', fontsize=16, fontweight='bold')
        
        # Box plots para variables principales
        variables_box = ['ingresado', 'atendido']
        
        box_data = []
        for var in variables_box:
            data = self.df[var].dropna()
            limite_sup = data.quantile(0.99)  # Usar el percentil 99 como límite
            box_data.append(data[data <= limite_sup])
        
        axes[0,0].boxplot(box_data, labels=variables_box, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.8),
                         flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
        
        axes[0,0].set_title('Box Plots - Detección de Outliers')
        axes[0,0].set_ylabel('Valor')
        axes[0,0].grid(True, alpha=0.3)

        if 'multivariado' in outliers and 'error' not in outliers['multivariado']:
            # Crear datos de base
            scatter_data = self.df[['ingresado', 'atendido', 'tasa_atencion']].copy()
            
            # Identificar outliers multivariados
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outliers_pred = iso_forest.fit_predict(
                scatter_data.dropna().reset_index(drop=True)
            )
            
            scatter_df = scatter_data.dropna().reset_index(drop=True).copy()
            scatter_df['outlier'] = outliers_pred == -1
            
            # Dibujar el scatter plot
            axes[0,1].scatter(scatter_df[~scatter_df['outlier']]['ingresado'], 
                            scatter_df[~scatter_df['outlier']]['atendido'],
                            alpha=0.3, s=30, c='blue', label='Normal')
            
            axes[0,1].scatter(scatter_df[scatter_df['outlier']]['ingresado'], 
                            scatter_df[scatter_df['outlier']]['atendido'],
                            alpha=0.8, s=60, c='red', label='Outlier')
            
            axes[0,1].set_title('Outliers Multivariados')
            axes[0,1].set_xlabel('Casos Ingresados')
            axes[0,1].set_ylabel('Casos Atendidos')
            axes[0,1].legend()
            
            max_val = max(scatter_df['ingresado'].max(), scatter_df['atendido'].max()) * 1.1
            axes[0,1].plot([0, max_val], [0, max_val], 'g--', alpha=0.7, label='Atención Perfecta')
            axes[0,1].legend()
        else:
            axes[0,1].text(0.5, 0.5, 'Datos insuficientes para análisis multivariado',
                         ha='center', va='center', fontsize=12)
            axes[0,1].axis('off')
        
        if 'contextuales' in outliers:
            ctx_outliers = outliers['contextuales']
            
            # Extraer tipos y conteos
            tipos = list(ctx_outliers.keys())
            counts = [ctx_outliers[k]['count'] for k in tipos]
            
            tipos_labels = [t.replace('_', ' ').title() for t in tipos]
            
            # Gráfico de barras
            axes[1,0].bar(tipos_labels, counts, color=['crimson', 'darkred', 'maroon'])
            axes[1,0].set_title('Outliers Contextuales por Tipo')
            axes[1,0].set_ylabel('Número de Casos')
            axes[1,0].tick_params(axis='x', rotation=20)
            
            # Añadir etiquetas
            for i, v in enumerate(counts):
                axes[1,0].text(i, v + 0.1, str(v), ha='center')
        else:
            axes[1,0].text(0.5, 0.5, 'No se detectaron outliers contextuales',
                         ha='center', va='center', fontsize=12)
            axes[1,0].axis('off')
        
        if 'ingresado' in outliers:
            outliers_info = []
            
            for var in ['ingresado', 'atendido']:
                if var in outliers:
                    for metodo, datos in outliers[var]['metodos'].items():
                        outliers_info.append(f"{var.title()} - {metodo}: {datos['count']} outliers ({datos['percentage']:.2f}%)")
                        if 'interpretacion' in datos:
                            outliers_info.append(f"  • {datos['interpretacion']}")
            
            # Crear tabla de texto
            if outliers_info:
                axes[1,1].axis('off')
                axes[1,1].set_title('Resumen de Outliers Univariados')
                axes[1,1].text(0, 0.9, '\n'.join(outliers_info), va='top', fontsize=10)
            else:
                axes[1,1].text(0.5, 0.5, 'No se detectaron outliers univariados',
                             ha='center', va='center', fontsize=12)
                axes[1,1].axis('off')
        else:
            axes[1,1].text(0.5, 0.5, 'No se detectaron outliers univariados',
                         ha='center', va='center', fontsize=12)
            axes[1,1].axis('off')
        
        plt.tight_layout()
        self._guardar_figura(fig, "Análisis_Outliers")
        plt.show()
        
        # Interpretar outliers
        print("\nInterpretaciones de Outliers:")
        
        # Outliers multivariados
        if 'multivariado' in outliers and 'error' not in outliers['multivariado']:
            multi_out = outliers['multivariado']
            print(f"- Se detectaron {multi_out['total_outliers']} outliers multivariados ({multi_out['porcentaje']:.1f}% de los datos)")
            
            if 'caracteristicas_outliers' in multi_out:
                chars = multi_out['caracteristicas_outliers']
                print(f"  --Promedio de casos ingresados en outliers: {chars['ingresado_promedio']:.1f}")
                print(f"  --Promedio de casos atendidos en outliers: {chars['atendido_promedio']:.1f}")
                
                if 'casos_extremos' in chars and chars['casos_extremos']:
                    print(f"  --Caso más extremo: {chars['casos_extremos'][0]['distrito_fiscal']} - {chars['casos_extremos'][0]['materia']} ({chars['casos_extremos'][0]['ingresado']} ingresados)")
        
        # Outliers contextuales
        if 'contextuales' in outliers:
            ctx_out = outliers['contextuales']
            print("\n- Outliers contextuales detectados:")
            
            for tipo, datos in ctx_out.items():
                print(f"  --{tipo.replace('_', ' ').title()}: {datos['count']} casos")
                print(f"    {datos['interpretacion']}")

                
    def guardar_resultados(self, ruta_carpeta='resultados_eda'):
        import os
        import json
        from datetime import datetime
        
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar resultados como JSON
        try:
            with open(f"{ruta_carpeta}/resultados_eda_{timestamp}.json", 'w', encoding='utf-8') as f:
                resultados_serializables = {}
                
                for categoria, datos in self.resultados_eda.items():
                    # Convertir DataFrames, Series, arrays, etc.
                    if isinstance(datos, dict):
                        resultados_serializables[categoria] = self._hacer_serializable(datos)
                    else:
                        resultados_serializables[categoria] = str(datos)
                
                json.dump(resultados_serializables, f, ensure_ascii=False, indent=2)
                
            print(f"\nResultados guardados en: {ruta_carpeta}/resultados_eda_{timestamp}.json")
        except Exception as e:
            print(f"\nError al guardar resultados: {e}")
    
    def _hacer_serializable(self, obj):
        """Convertir objeto a formato serializable para JSON"""
        if isinstance(obj, dict):
            # Convertir claves de tupla a cadenas
            result = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    new_key = "_".join(str(x) for x in k)
                else:
                    new_key = k
                result[new_key] = self._hacer_serializable(v)
            return result
        elif isinstance(obj, list):
            return [self._hacer_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            if isinstance(obj, pd.DataFrame) and isinstance(obj.columns, pd.MultiIndex):
                obj.columns = ["_".join(str(col) for col in col_tuple) 
                            if isinstance(col_tuple, tuple) else str(col_tuple) 
                            for col_tuple in obj.columns]
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    def ejecutar_eda_completo(self):
        print("\nIniciando analisis EDA\n")
        
        self.estadisticas_descriptivas_mejoradas()
        
        self.detectar_patrones_y_sesgos()
        
        self.detectar_outliers_mejorado()
        
        self.generar_visualizaciones_principales()
        
        print("\nAnalisis completado\n")
        
        return self.resultados_eda

def ejecutar_analisis(ruta_archivo):
    try:
        print(f"Cargando datos desde: {ruta_archivo}")
        df = pd.read_csv(ruta_archivo)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas\n")

        eda = EDAFiscaliaMejorado(df, nombre_dataset="Casos Fiscales Perú 2019-2023")
        resultados = eda.ejecutar_eda_completo()
        
        eda.guardar_resultados()
        
        return resultados
        
    except Exception as e:
        print(f"Error en el análisis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    ruta_datos = "datos_limpios/casos_fiscales_2019_2023_consolidado.csv"
    if len(sys.argv) > 1:
        ruta_archivo = sys.argv[1]
    else:
        ruta_archivo = ruta_datos
        
    ejecutar_analisis(ruta_archivo)
