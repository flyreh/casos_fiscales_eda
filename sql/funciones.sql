-- Función para calcular percentiles de eficiencia
CREATE OR REPLACE FUNCTION calcular_percentil_eficiencia(
    p_anio INTEGER DEFAULT NULL,
    p_materia VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    distrito_fiscal VARCHAR,
    tasa_atencion DECIMAL,
    percentil INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH datos AS (
        SELECT 
            v.distrito_fiscal,
            ROUND(
                CASE 
                    WHEN SUM(v.casos_ingresados) > 0 
                    THEN CAST(SUM(v.casos_atendidos) AS DECIMAL) / SUM(v.casos_ingresados)
                    ELSE 0
                END, 4
            ) as tasa_atencion
        FROM v_casos_fiscales_completo v
        WHERE 
            (p_anio IS NULL OR v.anio = p_anio)
            AND (p_materia IS NULL OR v.materia = p_materia)
        GROUP BY v.distrito_fiscal
        HAVING SUM(v.casos_ingresados) > 0
    )
    SELECT 
        d.distrito_fiscal,
        d.tasa_atencion,
        CAST(PERCENT_RANK() OVER (ORDER BY d.tasa_atencion) * 100 AS INTEGER) as percentil
    FROM datos d
    ORDER BY d.tasa_atencion DESC;
END;
$$ LANGUAGE plpgsql;

-- Trigger para actualizar timestamp en updates
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Aplicar trigger a tabla de hechos
CREATE TRIGGER tr_fact_casos_fiscales_updated_at
    BEFORE UPDATE ON fact_casos_fiscales
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();