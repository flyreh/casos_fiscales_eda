"""

1. EXTRAER TODO de los datos reales
2. NO ASUMIR complejidades, niveles, o clasificaciones arbitrarias  
3. SOLO usar patrones observables en los datos mismos
4. Si no hay información suficiente para clasificar, dejarlo genérico

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import os
from collections import Counter
import re

class AnalizadorDimensionesRealesPuro:
    
    def __init__(self, ruta_datos_limpios: str = "datos_limpios"):
        self.ruta_datos = Path(ruta_datos_limpios)
        self.resultados = {
            'timestamp': datetime.now().isoformat(),
            'archivos_analizados': [],
            'total_registros': 0,
            'dimensiones': {},
            'metadatos': {
                'enfoque': '100% Data-Driven Sin Supuestos',
                'principio': 'Solo extraer lo que existe, no asumir nada'
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("analisis de dimensiones reales")
    
    def cargar_todos_los_datasets(self) -> pd.DataFrame:
        """Cargar y consolidar todos los datasets limpios"""
        print("Cargando todos los datasets...")
        
        consolidado = self.ruta_datos / "casos_fiscales_2019_2023_consolidado.csv"
        archivos_csv = list(self.ruta_datos.glob("*limpio.csv"))
        
        if consolidado.exists():
            print(f"Usando dataset consolidado: {consolidado.name}")
            df = pd.read_csv(consolidado)
            self.resultados['archivos_analizados'] = [consolidado.name]
        elif archivos_csv:
            print(f" Consolidando {len(archivos_csv)} archivos individuales...")
            dfs = []
            for archivo in sorted(archivos_csv):
                print(f"- Cargando {archivo.name}...")
                df_temp = pd.read_csv(archivo)
                dfs.append(df_temp)
                self.resultados['archivos_analizados'].append(archivo.name)
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No se encontraron archivos CSV limpios")
        
        self.resultados['total_registros'] = len(df)
        print(f"Total registros: {len(df):,}")
        print(f"Columnas: {len(df.columns)}")
        
        return df
    
    def analizar_tipos_caso_puro(self, df: pd.DataFrame) -> Dict:
        
        tipos_unicos = df['tipo_caso'].dropna().unique()
        tipos_counts = df['tipo_caso'].value_counts().to_dict()
        
        tipos_data = []
        for tipo in sorted(tipos_unicos):
            tipo_clean = tipo.upper().strip()
            
            categoria = self._inferir_categoria_desde_nombre(tipo_clean)
            
            tipos_data.append({
                'tipo_caso': tipo_clean,
                'categoria': categoria,  # Solo patrones observables
                'frecuencia': tipos_counts[tipo],
                'porcentaje': (tipos_counts[tipo] / len(df)) * 100
            })
        
        analisis = {
            'total_tipos': len(tipos_unicos),
            'tipos_encontrados': [t['tipo_caso'] for t in tipos_data],
            'por_categoria': self._agrupar_por_categoria(tipos_data),
            'mas_frecuentes': sorted(tipos_data, key=lambda x: x['frecuencia'], reverse=True)[:10],
            'datos_sql': tipos_data,
            'metadatos': {
                'clasificacion_metodo': 'Inferencia automática desde nombres reales',
                'supuestos_aplicados': 'ninguno',
                'valores_externos': 'no se usaron valores precargados'
            }
        }
        
        print(f" {analisis['total_tipos']} tipos de caso encontrados en los datos")
        print(" Por categoría inferida:")
        for cat, count in analisis['por_categoria'].items():
            print(f" --{cat}: {count} tipos")
        
        print(" Los 5 más frecuentes:")
        for tipo_data in analisis['mas_frecuentes'][:5]:
            freq = tipo_data['frecuencia']
            pct = tipo_data['porcentaje']
            print(f"--{tipo_data['tipo_caso']}: {freq:,} casos ({pct:.1f}%)")
        
        return analisis
    
    def _inferir_categoria_desde_nombre(self, tipo_caso: str) -> str:
        tipo_upper = tipo_caso.upper()
        
        if 'DENUNCIA' in tipo_upper:
            return 'INGRESO'
        elif any(palabra in tipo_upper for palabra in ['APELACION', 'QUEJA']):
            return 'RECURSO'  
        elif 'PROCESO' in tipo_upper:
            return 'PROCESO_FORMAL'
        else:
            return 'PROCESO_GENERAL'  # Categoría general
    
    def _agrupar_por_categoria(self, tipos_data: List[Dict]) -> Dict[str, int]:
        categorias = {}
        for tipo in tipos_data:
            cat = tipo['categoria']
            categorias[cat] = categorias.get(cat, 0) + 1
        return categorias
    
    def analizar_fiscalias_especializadas_puro(self, df: pd.DataFrame) -> Dict:
        print("\nAnalizando fiscalías especializadas...")
        
        # extraemos casos con especialización real
        df_especializada = df[df['especializada'].notna()].copy()
        
        if len(df_especializada) == 0:
            print(" No se encontraron fiscalías especializadas en los datos")
            return {
                'total_fiscalias_especializadas': 0,
                'datos_sql': [],
                'metadatos': {'casos_especializados': 0, 'casos_generales': len(df)}
            }
        
        # extraemos valores únicos reales
        especializadas_unicas = df_especializada['especializada'].unique()
        especializadas_counts = df_especializada['especializada'].value_counts().to_dict()
        
        # Creamos datos básicos sin clasificaciones arbitrarias
        fiscalias_data = []
        for especializada in sorted(especializadas_unicas):
            esp_clean = especializada.upper().strip()
            
            # ÚNICA clasificación permitida: inferir área desde el nombre real
            area_inferida = self._inferir_area_desde_nombre(esp_clean)
            
            fiscalias_data.append({
                'especializada': esp_clean,
                'area_especializacion': area_inferida,  # Solo inferida del nombre
                'frecuencia': especializadas_counts[especializada],
                'porcentaje': (especializadas_counts[especializada] / len(df_especializada)) * 100
            })
        
        analisis = {
            'total_fiscalias_especializadas': len(fiscalias_data),
            'especializadas_encontradas': [f['especializada'] for f in fiscalias_data],
            'por_area_inferida': self._agrupar_por_area(fiscalias_data),
            'casos_especializados': len(df_especializada),
            'casos_generales': len(df) - len(df_especializada),
            'porcentaje_especializacion': (len(df_especializada) / len(df)) * 100,
            'mas_frecuentes': sorted(fiscalias_data, key=lambda x: x['frecuencia'], reverse=True),
            'datos_sql': fiscalias_data,
            'metadatos': {
                'clasificacion_metodo': 'Inferencia automática desde nombres reales',
                'supuestos_aplicados': 'NINGUNO',
                'areas_predefinidas': 'NO se usaron áreas precargadas'
            }
        }
        
        print(f"{analisis['total_fiscalias_especializadas']} fiscalías especializadas encontradas")
        print(f"Casos especializados: {analisis['casos_especializados']:,} ({analisis['porcentaje_especializacion']:.1f}%)")
        print(f"Casos generales: {analisis['casos_generales']:,}")
        
        print("Top 5 especializaciones por frecuencia:")
        for fisc_data in analisis['mas_frecuentes'][:5]:
            freq = fisc_data['frecuencia']
            print(f"      -{fisc_data['area_especializacion']}: {freq:,} casos")
        
        return analisis
    
    def _inferir_area_desde_nombre(self, especializada: str) -> str:
       
        esp_upper = especializada.upper()
        
        if 'CORRUPCION' in esp_upper:
            return 'Anticorrupción'
        elif 'AMBIENTAL' in esp_upper or 'AMBIENTE' in esp_upper:
            return 'Medio Ambiente'  
        elif 'CRIMINALIDAD' in esp_upper:
            return 'Crimen Organizado'
        elif 'DROGAS' in esp_upper or 'TRAFICO' in esp_upper:
            return 'Trafico ilicito de drogas'
        elif 'VIOLENCIA' in esp_upper and 'MUJER' in esp_upper:
            return 'Violencia de Género'
        elif 'LAVADO' in esp_upper and 'ACTIVOS' in esp_upper:
            return 'Lavado de Activos'
        elif 'ADUANEROS' in esp_upper or 'PROPIEDAD INTELECTUAL' in esp_upper:
            return 'Propiedad Intelectual'
        elif 'TRATA' in esp_upper and 'PERSONAS' in esp_upper:
            return 'Trata de Personas'
        elif 'TRANSITO' in esp_upper or 'VIAL' in esp_upper:
            return 'Tránsito y Seguridad Vial'
        elif 'TRIBUTARIOS' in esp_upper:
            return 'Tributario'
        elif 'TURISMO' in esp_upper:
            return 'Turismo'
        else:
            return 'General'  # Para casos no clasificables
    
    def _agrupar_por_area(self, fiscalias_data: List[Dict]) -> Dict[str, int]:
        """Agrupar fiscalías por área inferida"""
        areas = {}
        for fisc in fiscalias_data:
            area = fisc['area_especializacion']
            areas[area] = areas.get(area, 0) + 1
        return areas
    
    def analizar_materias_puro(self, df: pd.DataFrame) -> Dict:
        """
        Analizar materias REALES - Solo extraer lo que existe
        NO asumir prioridades ni áreas del derecho
        """
        print("\nAnalizando materias...")
        
        # 1. EXTRAER valores únicos reales
        materias_unicas = df['materia'].dropna().unique()
        materias_counts = df['materia'].value_counts().to_dict()
        
        # 2. Crear datos básicos SIN clasificaciones arbitrarias
        materias_data = []
        for materia in sorted(materias_unicas):
            materia_clean = materia.upper().strip()
            
            materias_data.append({
                'materia': materia_clean,
                'frecuencia': materias_counts[materia],
                'porcentaje': (materias_counts[materia] / len(df)) * 100,
                'es_principal': materias_counts[materia] >= len(df) * 0.05  # >5% se considera principal
            })
        
        analisis = {
            'total_materias': len(materias_unicas),
            'materias_encontradas': [m['materia'] for m in materias_data],
            'materias_principales': [m for m in materias_data if m['es_principal']],
            'materias_secundarias': [m for m in materias_data if not m['es_principal']],
            'distribucion': sorted(materias_data, key=lambda x: x['frecuencia'], reverse=True),
            'datos_sql': materias_data,
            'metadatos': {
                'clasificacion_metodo': 'Solo frecuencia de aparición',
                'supuestos_aplicados': 'NINGUNO',
                'prioridades_predefinidas': 'NO se asignaron prioridades arbitrarias'
            }
        }
        
        print(f"   {analisis['total_materias']} materias encontradas en los datos")
        print(f"   Materias principales (>5% casos): {len(analisis['materias_principales'])}")
        print(f"   Materias secundarias: {len(analisis['materias_secundarias'])}")
        
        print("   Distribución real por frecuencia:")
        for materia_data in analisis['distribucion']:
            freq = materia_data['frecuencia']
            pct = materia_data['porcentaje']
            tipo = "Principal" if materia_data['es_principal'] else "Secundaria"
            print(f"      • {materia_data['materia']}: {freq:,} casos ({pct:.1f}%) - {tipo}")
        
        return analisis
    
    def analizar_especialidades_puro(self, df: pd.DataFrame) -> Dict:
        print("\nAnalizando especialidades...")
        
        #EXTRAER especialidades únicas con sus materias reales
        esp_df = df[['especialidad', 'materia']].dropna().drop_duplicates()
        especialidades_counts = df['especialidad'].value_counts().to_dict()
        
        #Crear datos básicos SIN clasificaciones arbitrarias
        especialidades_data = []
        for _, row in esp_df.iterrows():
            especialidad = row['especialidad'].upper().strip()
            materia_asociada = row['materia'].upper().strip()
            
            especialidades_data.append({
                'especialidad': especialidad,
                'materia_asociada': materia_asociada,
                'frecuencia': especialidades_counts.get(row['especialidad'], 0),
                'porcentaje': (especialidades_counts.get(row['especialidad'], 0) / len(df)) * 100
            })
        
        # Eliminar duplicados
        especialidades_unique = {}
        for esp in especialidades_data:
            key = esp['especialidad']
            if key not in especialidades_unique or esp['frecuencia'] > especialidades_unique[key]['frecuencia']:
                especialidades_unique[key] = esp
        
        especialidades_final = list(especialidades_unique.values())
        
        analisis = {
            'total_especialidades': len(especialidades_final),
            'especialidades_encontradas': [e['especialidad'] for e in especialidades_final],
            'por_materia': self._agrupar_especialidades_por_materia(especialidades_final),
            'mas_frecuentes': sorted(especialidades_final, key=lambda x: x['frecuencia'], reverse=True)[:10],
            'datos_sql': especialidades_final,
            'metadatos': {
                'clasificacion_metodo': 'Solo asociación real especialidad-materia',
                'supuestos_aplicados': 'NINGUNO',
                'gravedad_predefinida': 'NO se asignó gravedad arbitraria',
                'especializacion_predefinida': 'NO se clasificó como especializada/general'
            }
        }
        
        print(f"   {analisis['total_especialidades']} especialidades únicas encontradas")
        
        print("   Distribución por materia:")
        for materia, count in analisis['por_materia'].items():
            print(f"      -{materia}: {count} especialidades")
        
        print("   Los 5 especialidades más frecuentes:")
        for esp_data in analisis['mas_frecuentes'][:5]:
            freq = esp_data['frecuencia']
            pct = esp_data['porcentaje']
            print(f"      • {esp_data['especialidad']}: {freq:,} casos ({pct:.1f}%)")
        
        return analisis
    
    def _agrupar_especialidades_por_materia(self, especialidades_data: List[Dict]) -> Dict[str, int]:
        por_materia = {}
        for esp in especialidades_data:
            materia = esp['materia_asociada']
            por_materia[materia] = por_materia.get(materia, 0) + 1
        return por_materia
    
    def analizar_ubicaciones_geograficas_puro(self, df: pd.DataFrame) -> Dict:
        print("\nAnalizando ubicaciones geográficas...")
        
        # Extraer ubicaciones únicas
        ubicaciones_cols = ['ubigeo_pjfs', 'dpto_pjfs', 'prov_pjfs', 'dist_pjfs']
        ubicaciones_df = df[ubicaciones_cols].drop_duplicates().dropna()
        
        # Limpiar y normalizar
        ubicaciones_df = ubicaciones_df.copy()
        for col in ['dpto_pjfs', 'prov_pjfs', 'dist_pjfs']:
            ubicaciones_df[col] = ubicaciones_df[col].astype(str).str.upper().str.strip()
        
        ubicaciones_df['region_geografica'] = ubicaciones_df['dpto_pjfs'].apply(
            self._inferir_region_desde_departamento_real
        )
        
        analisis = {
            'total_ubicaciones': len(ubicaciones_df),
            'departamentos_unicos': ubicaciones_df['dpto_pjfs'].nunique(),
            'provincias_unicas': ubicaciones_df['prov_pjfs'].nunique(),
            'distritos_unicos': ubicaciones_df['dist_pjfs'].nunique(),
            'departamentos_encontrados': sorted(ubicaciones_df['dpto_pjfs'].unique().tolist()),
            'regiones_inferidas': ubicaciones_df['region_geografica'].value_counts().to_dict(),
            'datos_sql': ubicaciones_df.to_dict('records'),
            'metadatos': {
                'clasificacion_metodo': 'Inferencia geográfica desde departamentos reales',
                'supuestos_aplicados': 'NINGUNO sobre ubicaciones específicas'
            }
        }
        
        print(f"   {analisis['total_ubicaciones']} ubicaciones únicas")
        print(f"   {analisis['departamentos_unicos']} departamentos encontrados")
        
        print("   Distribución regional inferida:")
        for region, count in analisis['regiones_inferidas'].items():
            print(f"      -{region}: {count} ubicaciones")
        
        return analisis
    
    def _inferir_region_desde_departamento_real(self, departamento: str) -> str:
        depto_clean = departamento.upper().strip()
        
        if any(d in depto_clean for d in ['TUMBES', 'PIURA', 'LAMBAYEQUE', 'LA LIBERTAD', 'CAJAMARCA', 'AMAZONAS']):
            return 'NORTE'
        elif any(d in depto_clean for d in ['ANCASH', 'LIMA', 'CALLAO', 'ICA', 'HUANCAVELICA', 'JUNIN', 'PASCO']):
            return 'CENTRO'
        elif any(d in depto_clean for d in ['AREQUIPA', 'MOQUEGUA', 'TACNA', 'CUSCO', 'APURIMAC', 'AYACUCHO', 'PUNO']):
            return 'SUR'
        elif any(d in depto_clean for d in ['LORETO', 'UCAYALI', 'MADRE DE DIOS', 'SAN MARTIN', 'HUANUCO']):
            return 'SELVA'
        else:
            return 'REGION_NO_CLASIFICADA'  # Para departamentos no reconocidos
    
    def analizar_distritos_fiscales_puro(self, df: pd.DataFrame) -> Dict:
        print("\nAnalizando distritos fiscales...")
        
        # Extraer distritos únicos con sus ubicaciones
        distritos_cols = ['distrito_fiscal', 'ubigeo_pjfs', 'dpto_pjfs']
        distritos_df = df[distritos_cols].drop_duplicates().dropna()
        
        # Limpiar nombres
        distritos_df = distritos_df.copy()
        distritos_df['distrito_fiscal'] = distritos_df['distrito_fiscal'].astype(str).str.upper().str.strip()
        
        analisis = {
            'total_distritos': len(distritos_df),
            'distritos_encontrados': sorted(distritos_df['distrito_fiscal'].unique().tolist()),
            'distritos_por_departamento': distritos_df.groupby('dpto_pjfs')['distrito_fiscal'].nunique().to_dict(),
            'datos_sql': distritos_df.to_dict('records'),
            'metadatos': {
                'clasificacion_metodo': 'Solo extracción de datos reales',
                'supuestos_aplicados': 'NINGUNO'
            }
        }
        
        print(f"   {analisis['total_distritos']} distritos fiscales únicos encontrados")
        print("   Top 5 departamentos por número de distritos:")
        top_deptos = sorted(analisis['distritos_por_departamento'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        for depto, count in top_deptos:
            print(f"      -{depto}: {count} distritos")
        
        return analisis
    
    def analizar_tipos_fiscalia_puro(self, df: pd.DataFrame) -> Dict:
        """Analizar tipos de fiscalía reales"""
        print("\nAnalizando tipos de fiscalía (PURO - sin supuestos)...")
        
        tipos_unicos = df['tipo_fiscalia'].dropna().unique()
        tipos_counts = df['tipo_fiscalia'].value_counts().to_dict()
        
        # Solo inferir nivel jerárquico desde nombres (conocimiento básico del sistema judicial)
        tipos_data = []
        for tipo in sorted(tipos_unicos):
            tipo_clean = tipo.upper().strip()
            nivel = self._inferir_nivel_jerarquico_desde_nombre(tipo_clean)
            
            tipos_data.append({
                'tipo_fiscalia': tipo_clean,
                'nivel_jerarquico': nivel,
                'frecuencia': tipos_counts[tipo],
                'porcentaje': (tipos_counts[tipo] / len(df)) * 100
            })
        
        analisis = {
            'total_tipos': len(tipos_unicos),
            'tipos_encontrados': [t['tipo_fiscalia'] for t in tipos_data],
            'distribucion': sorted(tipos_data, key=lambda x: x['frecuencia'], reverse=True),
            'datos_sql': tipos_data,
            'metadatos': {
                'clasificacion_metodo': 'Inferencia jerárquica desde nombres (conocimiento judicial básico)',
                'supuestos_aplicados': 'Solo jerarquía del sistema judicial peruano'
            }
        }
        
        print(f"{len(tipos_unicos)} tipos de fiscalía encontrados:")
        for tipo_data in tipos_data:
            freq = tipo_data['frecuencia']
            pct = tipo_data['porcentaje']
            print(f"      -{tipo_data['tipo_fiscalia']}: {freq:,} casos ({pct:.1f}%)")
        
        return analisis
    
    def _inferir_nivel_jerarquico_desde_nombre(self, tipo: str) -> int:
        tipo_clean = tipo.upper().strip()
        if 'SUPREMA' in tipo_clean:
            return 1
        elif 'SUPERIOR' in tipo_clean:
            return 2
        elif 'PROVINCIAL' in tipo_clean:
            return 3
        else:
            return 4  # Otros tipos
    
    def analizar_dimensiones_temporales_puro(self, df: pd.DataFrame) -> Dict:
        """Analizar dimensión temporal real"""
        print("\nAnalizando dimensión temporal...")
        
        # Años únicos
        años_unicos = sorted(df['anio'].dropna().unique().astype(int))
        periodos_unicos = df['periodo'].dropna().unique()
        
        # Crear datos temporales REALES
        tiempo_data = []
        for año in años_unicos:
            # Buscar período correspondiente a este año en los datos
            periodo_año = df[df['anio'] == año]['periodo'].iloc[0]
            
            tiempo_data.append({
                'anio': int(año),
                'periodo': periodo_año.strip(),
                'total_dias': 366 if año % 4 == 0 else 365  # Solo cálculo de días del año
            })
        
        analisis = {
            'años_encontrados': años_unicos,
            'periodos_encontrados': list(periodos_unicos),
            'rango_años': f"{min(años_unicos)}-{max(años_unicos)}",
            'total_años': len(años_unicos),
            'datos_sql': tiempo_data,
            'metadatos': {
                'clasificacion_metodo': 'Solo extracción de años y períodos reales',
                'supuestos_aplicados': 'NINGUNO sobre estructura temporal'
            }
        }
        
        print(f"   Período de datos: {analisis['rango_años']}")
        print(f"   {analisis['total_años']} años de datos")
        print(f"   Períodos encontrados: {', '.join(periodos_unicos)}")
        
        return analisis
    
    def generar_sql_puro(self) -> str:

        print("\nGenerando SQL data-driven...")
        
        sql_parts = []
        
        # Header
        sql_parts.append("""

SET client_encoding = 'UTF8';

""")
        
        # Estructura de tablas
        sql_parts.append(self._generar_estructura_tablas())
        
        # Ubicaciones 
        sql_parts.append("-- Ubicaciones Geográficas\n")
        sql_parts.append("INSERT INTO dim_ubicaciones (ubigeo_pjfs, departamento, provincia, distrito, region_geografica) VALUES\n")
        ubicaciones_data = self.resultados['dimensiones']['ubicaciones']['datos_sql']
        ubicaciones_sql = []
        for ubi in ubicaciones_data:
            ubicaciones_sql.append(f"({ubi['ubigeo_pjfs']}, '{ubi['dpto_pjfs']}', '{ubi['prov_pjfs']}', '{ubi['dist_pjfs']}', '{ubi['region_geografica']}')")
        sql_parts.append(",\n".join(ubicaciones_sql) + ";\n\n")
        
        # Tipos de fiscalía
        sql_parts.append("--Tipos de Fiscalía\n")
        sql_parts.append("INSERT INTO dim_tipos_fiscalia (tipo_fiscalia, nivel_jerarquico) VALUES\n")
        tipos_fiscalia_data = self.resultados['dimensiones']['tipos_fiscalia']['datos_sql']
        tipos_sql = []
        for tipo in tipos_fiscalia_data:
            tipos_sql.append(f"('{tipo['tipo_fiscalia']}', {tipo['nivel_jerarquico']})")
        sql_parts.append(",\n".join(tipos_sql) + ";\n\n")
        
        # Materias 
        sql_parts.append("--Materias (sin prioridades arbitrarias)\n")
        sql_parts.append("INSERT INTO dim_materias (materia, area_derecho, prioridad_nacional) VALUES\n")
        materias_data = self.resultados['dimensiones']['materias']['datos_sql']
        materias_sql = []
        for materia in materias_data:
            # Solo usar NULL para prioridad ya que no tenemos datos reales para esto
            materias_sql.append(f"('{materia['materia']}', NULL, NULL)")
        sql_parts.append(",\n".join(materias_sql) + ";\n\n")
        
        # Tipos de caso 
        sql_parts.append("--Tipos de Caso (sin complejidades arbitrarias)\n")
        sql_parts.append("INSERT INTO dim_tipos_caso (tipo_caso, categoria, complejidad_promedio) VALUES\n")
        tipos_caso_data = self.resultados['dimensiones']['tipos_caso']['datos_sql']
        tipos_caso_sql = []
        for tipo in tipos_caso_data:
            tipos_caso_sql.append(f"('{tipo['tipo_caso']}', '{tipo['categoria']}', NULL)")
        sql_parts.append(",\n".join(tipos_caso_sql) + ";\n\n")
        
        # Fiscalías especializadas
        if self.resultados['dimensiones']['fiscalias_especializadas']['total_fiscalias_especializadas'] > 0:
            sql_parts.append("--Fiscalías Especializadas\n")
            sql_parts.append("INSERT INTO dim_fiscalias_especializadas (especializada, area_especializacion, nivel_especialidad, requiere_capacitacion_especial) VALUES\n")
            fiscalias_data = self.resultados['dimensiones']['fiscalias_especializadas']['datos_sql']
            fiscalias_sql = []
            for fisc in fiscalias_data:
                # Solo uso área inferida, no niveles arbitrarios
                fiscalias_sql.append(f"('{fisc['especializada']}', '{fisc['area_especializacion']}', NULL, TRUE)")
            sql_parts.append(",\n".join(fiscalias_sql) + ";\n\n")
        
        # Tiempo
        sql_parts.append("--Dimensión Temporal\n")
        sql_parts.append("INSERT INTO dim_tiempo (anio, periodo, mes_inicio, mes_fin, total_dias) VALUES\n")
        tiempo_data = self.resultados['dimensiones']['tiempo']['datos_sql']
        tiempo_sql = []
        for tiempo in tiempo_data:
            # Solo datos reales de años y períodos
            tiempo_sql.append(f"({tiempo['anio']}, '{tiempo['periodo']}', 1, 12, {tiempo['total_dias']})")
        sql_parts.append(",\n".join(tiempo_sql) + ";\n\n")
        
        # Especialidades y distritos
        sql_parts.append(self._generar_inserts_relacionales())
        
        # Índices y vistas
        sql_parts.append(self._generar_indices_y_vistas())
        
        return "".join(sql_parts)
    
    def _generar_estructura_tablas(self) -> str:
        return """-- Estructura de tablas

DROP TABLE IF EXISTS dim_ubicaciones CASCADE;
CREATE TABLE dim_ubicaciones (
    id_ubicacion SERIAL PRIMARY KEY,
    ubigeo_pjfs INTEGER UNIQUE NOT NULL,
    departamento VARCHAR(70) NOT NULL,
    provincia VARCHAR(70) NOT NULL,
    distrito VARCHAR(70) NOT NULL,
    region_geografica VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_distritos_fiscales CASCADE;
CREATE TABLE dim_distritos_fiscales (
    id_distrito_fiscal SERIAL PRIMARY KEY,
    distrito_fiscal VARCHAR(50) UNIQUE NOT NULL,
    id_ubicacion INTEGER REFERENCES dim_ubicaciones(id_ubicacion),
    estado_activo BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_tipos_fiscalia CASCADE;
CREATE TABLE dim_tipos_fiscalia (
    id_tipo_fiscalia SERIAL PRIMARY KEY,
    tipo_fiscalia VARCHAR(20) UNIQUE NOT NULL,
    nivel_jerarquico INTEGER,
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_materias CASCADE;
CREATE TABLE dim_materias (
    id_materia SERIAL PRIMARY KEY,
    materia VARCHAR(30) UNIQUE NOT NULL,
    area_derecho VARCHAR(50),
    prioridad_nacional INTEGER,
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_especialidades CASCADE;
CREATE TABLE dim_especialidades (
    id_especialidad SERIAL PRIMARY KEY,
    especialidad VARCHAR(50) UNIQUE NOT NULL,
    id_materia INTEGER REFERENCES dim_materias(id_materia),
    es_especializada BOOLEAN DEFAULT NULL,  -- NULL = no clasificado
    gravedad_casos INTEGER DEFAULT NULL,     -- NULL = no clasificado
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_tipos_caso CASCADE;
CREATE TABLE dim_tipos_caso (
    id_tipo_caso SERIAL PRIMARY KEY,
    tipo_caso VARCHAR(50) UNIQUE NOT NULL,
    categoria VARCHAR(30),
    complejidad_promedio INTEGER DEFAULT NULL,  -- NULL = no clasificado
    tiempo_promedio_dias INTEGER,
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_fiscalias_especializadas CASCADE;
CREATE TABLE dim_fiscalias_especializadas (
    id_fiscalia_especializada SERIAL PRIMARY KEY,
    especializada VARCHAR(100) UNIQUE,
    area_especializacion VARCHAR(50),
    nivel_especialidad INTEGER DEFAULT NULL,  -- NULL = no clasificado
    requiere_capacitacion_especial BOOLEAN DEFAULT TRUE,
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

DROP TABLE IF EXISTS dim_tiempo CASCADE;
CREATE TABLE dim_tiempo (
    id_tiempo SERIAL PRIMARY KEY,
    anio INTEGER NOT NULL,
    periodo VARCHAR(50) NOT NULL,
    trimestre INTEGER,
    mes_inicio INTEGER,
    mes_fin INTEGER,
    total_dias INTEGER,
    es_periodo_completo BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(anio, periodo)
);

DROP TABLE IF EXISTS fact_casos_fiscales CASCADE;
CREATE TABLE fact_casos_fiscales (
    id_caso SERIAL PRIMARY KEY,
    id_tiempo INTEGER NOT NULL REFERENCES dim_tiempo(id_tiempo),
    id_distrito_fiscal INTEGER NOT NULL REFERENCES dim_distritos_fiscales(id_distrito_fiscal),
    id_tipo_fiscalia INTEGER NOT NULL REFERENCES dim_tipos_fiscalia(id_tipo_fiscalia),
    id_materia INTEGER NOT NULL REFERENCES dim_materias(id_materia),
    id_especialidad INTEGER NOT NULL REFERENCES dim_especialidades(id_especialidad),
    id_tipo_caso INTEGER NOT NULL REFERENCES dim_tipos_caso(id_tipo_caso),
    id_fiscalia_especializada INTEGER REFERENCES dim_fiscalias_especializadas(id_fiscalia_especializada),
    
    casos_ingresados INTEGER NOT NULL DEFAULT 0,
    casos_atendidos INTEGER NOT NULL DEFAULT 0,
    casos_pendientes INTEGER GENERATED ALWAYS AS (casos_ingresados - casos_atendidos) STORED,
    tasa_atencion DECIMAL(5,4) GENERATED ALWAYS AS (
        CASE 
            WHEN casos_ingresados > 0 THEN CAST(casos_atendidos AS DECIMAL) / casos_ingresados
            ELSE 0
        END
    ) STORED,
    
    fecha_descarga DATE NOT NULL,
    fecha_corte DATE NOT NULL,
    periodo_original VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT chk_casos_positivos CHECK (casos_ingresados >= 0 AND casos_atendidos >= 0),
    CONSTRAINT chk_casos_logicos CHECK (casos_atendidos <= casos_ingresados),
    CONSTRAINT chk_fechas_logicas CHECK (fecha_corte >= fecha_descarga)
);

"""
    
    def _generar_inserts_relacionales(self) -> str:
        """Generar inserts para tablas que requieren referencias"""
        sql = """-- DATOS REALES: Especialidades (requiere IDs de materias)
INSERT INTO dim_especialidades (especialidad, id_materia)
SELECT 
    esp.especialidad,
    m.id_materia
FROM (VALUES
"""
        
        especialidades_data = self.resultados['dimensiones']['especialidades']['datos_sql']
        esp_sql = []
        for esp in especialidades_data:
            esp_sql.append(f"('{esp['especialidad']}', '{esp['materia_asociada']}')")
        
        sql += ",\n".join(esp_sql)
        sql += """\n) AS esp(especialidad, materia_asociada)
JOIN dim_materias m ON m.materia = esp.materia_asociada;

-- DATOS REALES: Distritos Fiscales (requiere IDs de ubicaciones)
INSERT INTO dim_distritos_fiscales (distrito_fiscal, id_ubicacion)
SELECT DISTINCT
    df.distrito_fiscal,
    u.id_ubicacion
FROM (VALUES
"""
        
        distritos_data = self.resultados['dimensiones']['distritos_fiscales']['datos_sql']
        distritos_sql = []
        for dist in distritos_data:
            distritos_sql.append(f"('{dist['distrito_fiscal']}', {dist['ubigeo_pjfs']})")
        
        sql += ",\n".join(distritos_sql)
        sql += """\n) AS df(distrito_fiscal, ubigeo_pjfs)
JOIN dim_ubicaciones u ON u.ubigeo_pjfs = df.ubigeo_pjfs;

"""
        
        return sql
    
    def _generar_indices_y_vistas(self) -> str:
        return """
-- ÍNDICES PARA PERFORMANCE

CREATE INDEX idx_fact_casos_tiempo ON fact_casos_fiscales(id_tiempo);
CREATE INDEX idx_fact_casos_distrito ON fact_casos_fiscales(id_distrito_fiscal);
CREATE INDEX idx_fact_casos_materia ON fact_casos_fiscales(id_materia);


CREATE OR REPLACE VIEW v_casos_fiscales_completo AS
SELECT 
    fc.id_caso,
    dt.anio,
    dt.periodo,
    df.distrito_fiscal,
    u.departamento,
    u.provincia,
    u.distrito,
    u.ubigeo_pjfs,
    u.region_geografica,
    tif.tipo_fiscalia,
    tif.nivel_jerarquico,
    m.materia,
    e.especialidad,
    tc.tipo_caso,
    tc.categoria as categoria_caso,
    fe.especializada as fiscalia_especializada,
    fe.area_especializacion,
    fc.casos_ingresados,
    fc.casos_atendidos,
    fc.casos_pendientes,
    fc.tasa_atencion,
    fc.fecha_descarga,
    fc.fecha_corte
FROM fact_casos_fiscales fc
INNER JOIN dim_tiempo dt ON fc.id_tiempo = dt.id_tiempo
INNER JOIN dim_distritos_fiscales df ON fc.id_distrito_fiscal = df.id_distrito_fiscal
INNER JOIN dim_ubicaciones u ON df.id_ubicacion = u.id_ubicacion
INNER JOIN dim_tipos_fiscalia tif ON fc.id_tipo_fiscalia = tif.id_tipo_fiscalia
INNER JOIN dim_materias m ON fc.id_materia = m.id_materia
INNER JOIN dim_especialidades e ON fc.id_especialidad = e.id_especialidad
INNER JOIN dim_tipos_caso tc ON fc.id_tipo_caso = tc.id_tipo_caso
LEFT JOIN dim_fiscalias_especializadas fe ON fc.id_fiscalia_especializada = fe.id_fiscalia_especializada;

SELECT 'Esquema data-driven creado exitosamente ' AS resultado;

"""
    
    def ejecutar_analisis_completo_puro(self) -> Dict[str, str]:
        # Cargar datasets
        df_completo = self.cargar_todos_los_datasets()
        
        # Analizar cada dimensión
        
        self.resultados['dimensiones']['ubicaciones'] = self.analizar_ubicaciones_geograficas_puro(df_completo)
        self.resultados['dimensiones']['distritos_fiscales'] = self.analizar_distritos_fiscales_puro(df_completo)
        self.resultados['dimensiones']['tipos_fiscalia'] = self.analizar_tipos_fiscalia_puro(df_completo)
        self.resultados['dimensiones']['materias'] = self.analizar_materias_puro(df_completo)
        self.resultados['dimensiones']['especialidades'] = self.analizar_especialidades_puro(df_completo)
        self.resultados['dimensiones']['tipos_caso'] = self.analizar_tipos_caso_puro(df_completo)
        self.resultados['dimensiones']['fiscalias_especializadas'] = self.analizar_fiscalias_especializadas_puro(df_completo)
        self.resultados['dimensiones']['tiempo'] = self.analizar_dimensiones_temporales_puro(df_completo)
        
        # Guardar resultados
        archivos_generados = self._guardar_resultados_puro()
        
        print(f"Datos analizados: {self.resultados['total_registros']:,} registros")
        print(f"Archivos generados: {len(archivos_generados)}")
        
        return archivos_generados
    
    def _guardar_resultados_puro(self) -> Dict[str, str]:
        salida_path = Path("analisis_dimensiones")
        sql_path = Path("sql")
        salida_path.mkdir(exist_ok=True)
        
        archivos_generados = {}
        
        # JSON completo
        json_path = salida_path / "analisis_.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False, default=str)
        archivos_generados['json_completo'] = str(json_path)
        
        # SQL puro sin supuestos
        with open(sql_path / "esquema.sql", 'w', encoding='utf-8') as f:
            f.write(self.generar_sql_puro())
        archivos_generados['sql_puro'] = str(sql_path / "esquema.sql")
        
        return archivos_generados
    

def main():
    print("analisis data-driven")
    print("Casos Fiscales Perú 2019-2023")
    print("-" * 70)
    
    try:
        analizador = AnalizadorDimensionesRealesPuro()
        archivos_generados = analizador.ejecutar_analisis_completo_puro()
        
        print(f"\nANÁLISIS EXITOSO")
        print("Archivos generados en: analisis_dimensiones_reales_puro/")
        print("Usar: slq/esquema.sql")
        
        return archivos_generados
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultado = main()