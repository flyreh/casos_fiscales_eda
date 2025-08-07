--Resumen Ejecutivo por Año
CREATE OR REPLACE VIEW v_resumen_ejecutivo AS
SELECT 
    anio,
    COUNT(*) as total_registros,
    SUM(casos_ingresados) as total_ingresados,
    SUM(casos_atendidos) as total_atendidos,
    SUM(casos_pendientes) as total_pendientes,
    ROUND(AVG(tasa_atencion), 4) as tasa_atencion_promedio,
    ROUND(STDDEV(tasa_atencion), 4) as desviacion_tasa_atencion,
    COUNT(DISTINCT distrito_fiscal) as distritos_activos,
    COUNT(DISTINCT materia) as materias_activas,
    
    -- Métricas de calidad (definiendo clasificación en línea)
    COUNT(CASE WHEN 
        CASE 
            WHEN tasa_atencion >= 0.95 THEN 'Excelente'
            WHEN tasa_atencion >= 0.85 THEN 'Bueno'
            WHEN tasa_atencion >= 0.70 THEN 'Regular'
            ELSE 'Deficiente'
        END = 'Excelente' THEN 1 END) as registros_excelentes,
    
    COUNT(CASE WHEN 
        CASE 
            WHEN tasa_atencion >= 0.95 THEN 'Excelente'
            WHEN tasa_atencion >= 0.85 THEN 'Bueno'
            WHEN tasa_atencion >= 0.70 THEN 'Regular'
            ELSE 'Deficiente'
        END = 'Deficiente' THEN 1 END) as registros_deficientes,
    
    -- Distribución por volumen (definiendo clasificación en línea)
    COUNT(CASE WHEN 
        CASE 
            WHEN casos_ingresados >= 10000 THEN 'Muy Alto'
            WHEN casos_ingresados >= 5000 THEN 'Alto'
            WHEN casos_ingresados >= 1000 THEN 'Medio'
            ELSE 'Bajo'
        END = 'Muy Alto' THEN 1 END) as registros_volumen_muy_alto,
    
    COUNT(CASE WHEN 
        CASE 
            WHEN casos_ingresados >= 10000 THEN 'Muy Alto'
            WHEN casos_ingresados >= 5000 THEN 'Alto'
            WHEN casos_ingresados >= 1000 THEN 'Medio'
            ELSE 'Bajo'
        END = 'Alto' THEN 1 END) as registros_volumen_alto

FROM v_casos_fiscales_completo
GROUP BY anio
ORDER BY anio;

--Performance por Distrito Fiscal
CREATE OR REPLACE VIEW v_performance_distrito AS
SELECT 
    distrito_fiscal,
    departamento,
    region_geografica,
    anio,
    
    -- Métricas agregadas
    SUM(casos_ingresados) as total_ingresados,
    SUM(casos_atendidos) as total_atendidos,
    SUM(casos_pendientes) as total_pendientes,
    
    -- Tasas calculadas
    ROUND(
        CASE 
            WHEN SUM(casos_ingresados) > 0 
            THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
            ELSE 0
        END, 4
    ) as tasa_atencion_distrito,
    
    -- Conteos
    COUNT(*) as registros_distrito,
    COUNT(DISTINCT materia) as materias_manejadas,
    COUNT(DISTINCT tipo_fiscalia) as tipos_fiscalia_activos,
    
    -- Rankings
    DENSE_RANK() OVER (
        PARTITION BY anio 
        ORDER BY SUM(casos_ingresados) DESC
    ) as ranking_volumen,
    
    DENSE_RANK() OVER (
        PARTITION BY anio 
        ORDER BY 
            CASE 
                WHEN SUM(casos_ingresados) > 0 
                THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
                ELSE 0
            END DESC
    ) as ranking_eficiencia

FROM v_casos_fiscales_completo
GROUP BY distrito_fiscal, departamento, region_geografica, anio
ORDER BY anio DESC, tasa_atencion_distrito DESC;

--Análisis por Materia y Especialidad
CREATE OR REPLACE VIEW v_analisis_materia AS
SELECT 
    materia,
    especialidad,
    anio,
    
    -- Agregaciones básicas
    SUM(casos_ingresados) as casos_ingresados,
    SUM(casos_atendidos) as casos_atendidos,
    SUM(casos_pendientes) as casos_pendientes,
    
    -- Estadísticas descriptivas
    ROUND(AVG(casos_ingresados), 2) as promedio_ingresados_por_registro,
    ROUND(STDDEV(casos_ingresados), 2) as desviacion_ingresados,
    MAX(casos_ingresados) as max_ingresados_por_registro,
    MIN(casos_ingresados) as min_ingresados_por_registro,
    
    -- Tasa de atención agregada
    ROUND(
        CASE 
            WHEN SUM(casos_ingresados) > 0 
            THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
            ELSE 0
        END, 4
    ) as tasa_atencion,
    
    -- Conteos
    COUNT(*) as registros,
    COUNT(DISTINCT distrito_fiscal) as distritos_con_materia,
    
    -- Porcentajes
    ROUND(
        100.0 * SUM(casos_ingresados) / 
        SUM(SUM(casos_ingresados)) OVER (PARTITION BY anio), 2
    ) as porcentaje_casos_del_año

FROM v_casos_fiscales_completo
GROUP BY materia, especialidad, anio
ORDER BY anio DESC, casos_ingresados DESC;

--Top Performers (Distritos más eficientes)
CREATE OR REPLACE VIEW v_top_performers AS
WITH performance_calcs AS (
    SELECT 
        distrito_fiscal,
        departamento,
        region_geografica,
        
        -- Agregaciones totales (todos los años)
        SUM(casos_ingresados) as total_ingresados,
        SUM(casos_atendidos) as total_atendidos,
        SUM(casos_pendientes) as total_pendientes,
        
        -- Tasa de atención general
        ROUND(
            CASE 
                WHEN SUM(casos_ingresados) > 0 
                THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
                ELSE 0
            END, 4
        ) as tasa_atencion,
        
        -- Consistencia año a año
        COUNT(DISTINCT anio) as años_activos,
        ROUND(STDDEV(tasa_atencion), 4) as consistencia_tasa,
        
        -- Diversidad de casos
        COUNT(DISTINCT materia) as materias_manejadas,
        COUNT(DISTINCT especialidad) as especialidades_manejadas,
        COUNT(*) as total_registros

    FROM v_casos_fiscales_completo
    GROUP BY distrito_fiscal, departamento, region_geografica
    HAVING SUM(casos_ingresados) >= 500  -- Filtro de volumen mínimo
)
SELECT 
    *,
    
    -- Rankings
    DENSE_RANK() OVER (ORDER BY tasa_atencion DESC) as ranking_eficiencia,
    DENSE_RANK() OVER (ORDER BY total_ingresados DESC) as ranking_volumen,
    DENSE_RANK() OVER (ORDER BY consistencia_tasa ASC) as ranking_consistencia,
    
    -- Clasificaciones
    CASE 
        WHEN tasa_atencion >= 0.95 THEN 'Excelente'
        WHEN tasa_atencion >= 0.85 THEN 'Bueno'
        WHEN tasa_atencion >= 0.70 THEN 'Regular'
        ELSE 'Deficiente'
    END as clasificacion_eficiencia,
    
    CASE 
        WHEN total_ingresados >= 10000 THEN 'Muy Alto Volumen'
        WHEN total_ingresados >= 5000 THEN 'Alto Volumen'
        WHEN total_ingresados >= 1000 THEN 'Medio Volumen'
        ELSE 'Bajo Volumen'
    END as clasificacion_volumen

FROM performance_calcs
ORDER BY tasa_atencion DESC;

--Evolución Temporal Detallada
CREATE OR REPLACE VIEW v_evolucion_temporal AS
SELECT 
    anio,
    materia,
    departamento,
    
    -- Agregaciones por período
    SUM(casos_ingresados) as ingresados,
    SUM(casos_atendidos) as atendidos,
    SUM(casos_pendientes) as pendientes,
    
    -- Tasa de atención
    ROUND(
        CASE 
            WHEN SUM(casos_ingresados) > 0 
            THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
            ELSE 0
        END, 4
    ) as tasa_atencion,
    
    -- Conteos
    COUNT(DISTINCT distrito_fiscal) as distritos_reportando,
    COUNT(*) as registros,
    
    -- Variaciones año anterior
    LAG(SUM(casos_ingresados)) OVER (
        PARTITION BY materia, departamento 
        ORDER BY anio
    ) as ingresados_año_anterior,
    
    ROUND(
        100.0 * (
            SUM(casos_ingresados) - 
            LAG(SUM(casos_ingresados)) OVER (
                PARTITION BY materia, departamento 
                ORDER BY anio
            )
        ) / NULLIF(
            LAG(SUM(casos_ingresados)) OVER (
                PARTITION BY materia, departamento 
                ORDER BY anio
            ), 0
        ), 2
    ) as variacion_porcentual_ingresados

FROM v_casos_fiscales_completo
GROUP BY anio, materia, departamento
ORDER BY anio, materia, departamento;

--Análisis de Fiscalías Especializadas
CREATE OR REPLACE VIEW v_fiscalias_especializadas AS
SELECT 
    COALESCE(fiscalia_especializada, 'NO ESPECIALIZADA') as tipo_fiscalia_especializada,
    COALESCE(area_especializacion, 'GENERAL') as area,
    anio,
    materia,
    
    -- Métricas
    SUM(casos_ingresados) as casos_ingresados,
    SUM(casos_atendidos) as casos_atendidos,
    COUNT(*) as registros,
    COUNT(DISTINCT distrito_fiscal) as distritos,
    
    -- Comparación especializada vs no especializada
    ROUND(
        100.0 * SUM(casos_ingresados) / 
        SUM(SUM(casos_ingresados)) OVER (PARTITION BY anio, materia), 2
    ) as porcentaje_casos_materia,
    
    -- Eficiencia
    ROUND(
        CASE 
            WHEN SUM(casos_ingresados) > 0 
            THEN CAST(SUM(casos_atendidos) AS DECIMAL) / SUM(casos_ingresados)
            ELSE 0
        END, 4
    ) as tasa_atencion

FROM v_casos_fiscales_completo
GROUP BY 
    COALESCE(fiscalia_especializada, 'NO ESPECIALIZADA'),
    COALESCE(area_especializacion, 'GENERAL'),
    anio, 
    materia
ORDER BY anio DESC, casos_ingresados DESC;

--Métricas por Región Geográfica
CREATE OR REPLACE VIEW v_metricas_regionales AS
SELECT 
    region_geografica,
    departamento,
    anio,
    
    -- Agregaciones
    SUM(casos_ingresados) as total_ingresados,
    SUM(casos_atendidos) as total_atendidos,
    COUNT(DISTINCT distrito_fiscal) as distritos_region,
    COUNT(*) as registros_region,
    
    -- Promedios
    ROUND(AVG(casos_ingresados), 2) as promedio_ingresados_por_registro,
    ROUND(AVG(tasa_atencion), 4) as tasa_atencion_promedio,
    
    -- Distribución por materias
    COUNT(DISTINCT materia) as materias_activas,
    
    -- Casos especializados vs generales
    COUNT(CASE WHEN fiscalia_especializada IS NOT NULL THEN 1 END) as registros_especializados,
    COUNT(CASE WHEN fiscalia_especializada IS NULL THEN 1 END) as registros_generales,
    
    -- Porcentaje especialización
    ROUND(
        100.0 * COUNT(CASE WHEN fiscalia_especializada IS NOT NULL THEN 1 END) / 
        COUNT(*), 2
    ) as porcentaje_especializacion

FROM v_casos_fiscales_completo
WHERE region_geografica IS NOT NULL
GROUP BY region_geografica, departamento, anio
ORDER BY anio DESC, total_ingresados DESC;