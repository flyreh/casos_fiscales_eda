

SET client_encoding = 'UTF8';

-- Estructura de tablas

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

-- Ubicaciones Geográficas
INSERT INTO dim_ubicaciones (ubigeo_pjfs, departamento, provincia, distrito, region_geografica) VALUES
(10101, 'AMAZONAS', 'CHACHAPOYAS', 'CHACHAPOYAS', 'NORTE'),
(20101, 'ANCASH', 'HUARAZ', 'HUARAZ', 'CENTRO'),
(30101, 'APURIMAC', 'ABANCAY', 'ABANCAY', 'SUR'),
(40101, 'AREQUIPA', 'AREQUIPA', 'AREQUIPA', 'SUR'),
(50110, 'AYACUCHO', 'HUAMANGA', 'SAN JUAN BAUTISTA', 'SUR'),
(60101, 'CAJAMARCA', 'CAJAMARCA', 'CAJAMARCA', 'NORTE'),
(70101, 'CALLAO', 'CALLAO', 'CALLAO', 'CENTRO'),
(150501, 'LIMA', 'CAÑETE', 'SAN VICENTE DE CAÑETE', 'CENTRO'),
(80108, 'CUSCO', 'CUSCO', 'WANCHAQ', 'SUR'),
(90101, 'HUANCAVELICA', 'HUANCAVELICA', 'HUANCAVELICA', 'CENTRO'),
(100101, 'HUANUCO', 'HUANUCO', 'HUANUCO', 'SELVA'),
(150801, 'LIMA', 'HUAURA', 'HUACHO', 'CENTRO'),
(110101, 'ICA', 'ICA', 'ICA', 'CENTRO'),
(120114, 'JUNIN', 'HUANCAYO', 'EL TAMBO', 'CENTRO'),
(130101, 'LA LIBERTAD', 'TRUJILLO', 'TRUJILLO', 'NORTE'),
(140101, 'LAMBAYEQUE', 'CHICLAYO', 'CHICLAYO', 'NORTE'),
(150101, 'LIMA', 'LIMA', 'LIMA', 'CENTRO'),
(150137, 'LIMA', 'LIMA', 'SANTA ANITA', 'CENTRO'),
(70106, 'CALLAO', 'CALLAO', 'VENTANILLA', 'CENTRO'),
(150112, 'LIMA', 'LIMA', 'INDEPENDENCIA', 'CENTRO'),
(150142, 'LIMA', 'LIMA', 'VILLA EL SALVADOR', 'CENTRO'),
(160101, 'LORETO', 'MAYNAS', 'IQUITOS', 'SELVA'),
(170101, 'MADRE DE DIOS', 'TAMBOPATA', 'TAMBOPATA', 'SELVA'),
(180101, 'MOQUEGUA', 'MARISCAL NIETO', 'MOQUEGUA', 'SUR'),
(190113, 'PASCO', 'PASCO', 'YANACANCHA', 'CENTRO'),
(200101, 'PIURA', 'PIURA', 'PIURA', 'NORTE'),
(210101, 'PUNO', 'PUNO', 'PUNO', 'SUR'),
(220101, 'SAN MARTIN', 'MOYOBAMBA', 'MOYOBAMBA', 'SELVA'),
(21801, 'ANCASH', 'SANTA', 'CHIMBOTE', 'CENTRO'),
(120301, 'JUNIN', 'CHANCHAMAYO', 'CHANCHAMAYO', 'CENTRO'),
(200601, 'PIURA', 'SULLANA', 'SULLANA', 'NORTE'),
(230101, 'TACNA', 'TACNA', 'TACNA', 'SUR'),
(240101, 'TUMBES', 'TUMBES', 'TUMBES', 'NORTE'),
(250101, 'UCAYALI', 'CORONEL PORTILLO', 'CALLERIA', 'SELVA');

--Tipos de Fiscalía
INSERT INTO dim_tipos_fiscalia (tipo_fiscalia, nivel_jerarquico) VALUES
('PROVINCIAL', 3),
('SUPERIOR', 2),
('SUPREMA', 1);

--Materias (sin prioridades arbitrarias)
INSERT INTO dim_materias (materia, area_derecho, prioridad_nacional) VALUES
('CIVIL', NULL, NULL),
('CONTENCIOSO ADMINISTRATIVO', NULL, NULL),
('CONTROL INTERNO', NULL, NULL),
('FAMILIA', NULL, NULL),
('PENAL', NULL, NULL);

--Tipos de Caso (sin complejidades arbitrarias)
INSERT INTO dim_tipos_caso (tipo_caso, categoria, complejidad_promedio) VALUES
('APELACION DENUNCIA', 'INGRESO', NULL),
('APELACION EXPEDIENTE', 'RECURSO', NULL),
('CONSULTAS', 'PROCESO_GENERAL', NULL),
('CONTIENDA', 'PROCESO_GENERAL', NULL),
('DENUNCIA', 'INGRESO', NULL),
('EXCLUSION FISCAL', 'PROCESO_GENERAL', NULL),
('EXPEDIENTE', 'PROCESO_GENERAL', NULL),
('INCIDENTE', 'PROCESO_GENERAL', NULL),
('INCIDENTE-EXPEDIENTE', 'PROCESO_GENERAL', NULL),
('INVESTIGACION PREVENTIVA', 'PROCESO_GENERAL', NULL),
('PROCESO CIVIL', 'PROCESO_FORMAL', NULL),
('PROCESO CONTENCIOSO', 'PROCESO_FORMAL', NULL),
('PROCESO FAMILIA', 'PROCESO_FORMAL', NULL),
('PROCESO PENAL', 'PROCESO_FORMAL', NULL),
('QUEJAS', 'RECURSO', NULL);

--Fiscalías Especializadas
INSERT INTO dim_fiscalias_especializadas (especializada, area_especializacion, nivel_especialidad, requiere_capacitacion_especial) VALUES
('CONTRA LA CRIMINALIDAD ORGANIZADA', 'Crimen Organizado', NULL, TRUE),
('CRIMINALIDAD ORGANIZADA', 'Crimen Organizado', NULL, TRUE),
('DELITOS ADUANEROS Y CONTRA LA PROPIEDAD INTELECTUAL', 'Propiedad Intelectual', NULL, TRUE),
('DELITOS DE CIBERDELINCUENCIA', 'General', NULL, TRUE),
('DELITOS DE CORRUPCION DE FUNCIONARIOS', 'Anticorrupción', NULL, TRUE),
('DELITOS DE TRAFICO ILICITO DE DROGAS', 'Trafico ilicito de drogas', NULL, TRUE),
('DELITOS DE TRATA DE PERSONAS', 'Trata de Personas', NULL, TRUE),
('DELITOS TRIBUTARIOS', 'Tributario', NULL, TRUE),
('LAVADO DE ACTIVOS Y PERDIDA DE DOMINIO', 'Lavado de Activos', NULL, TRUE),
('MATERIA AMBIENTAL', 'Medio Ambiente', NULL, TRUE),
('NO_TIPO_ESPECIALIZADA', 'General', NULL, TRUE),
('TRANSITO Y SEGURIDAD VIAL', 'Tránsito y Seguridad Vial', NULL, TRUE),
('TURISMO', 'Turismo', NULL, TRUE),
('VIOLENCIA CONTRA LA MUJER Y LOS INTEGRANTES DEL GRUPO FAMILIAR', 'Violencia de Género', NULL, TRUE);

--Dimensión Temporal
INSERT INTO dim_tiempo (anio, periodo, mes_inicio, mes_fin, total_dias) VALUES
(2019, 'ENERO - DICIEMBRE', 1, 12, 365),
(2020, 'ENERO - DICIEMBRE', 1, 12, 366),
(2021, 'ENERO - DICIEMBRE', 1, 12, 365),
(2022, 'ENERO - DICIEMBRE', 1, 12, 365),
(2023, 'ENERO - DICIEMBRE', 1, 12, 365);

-- DATOS REALES: Especialidades (requiere IDs de materias)
INSERT INTO dim_especialidades (especialidad, id_materia)
SELECT 
    esp.especialidad,
    m.id_materia
FROM (VALUES
('CIVIL', 'CIVIL'),
('ACTOS CONTRA LA LIBERTAD SEXUAL', 'FAMILIA'),
('FAMILIA CIVIL', 'FAMILIA'),
('FAMILIA PENAL', 'FAMILIA'),
('FAMILIA TUTELAR', 'FAMILIA'),
('VIOLENCIA FAMILIAR', 'FAMILIA'),
('PENAL', 'PENAL'),
('CONTENCIOSO ADMINISTRATIVO', 'CONTENCIOSO ADMINISTRATIVO'),
('CONTROL INTERNO', 'CONTROL INTERNO'),
('FAMILIA', 'FAMILIA')
) AS esp(especialidad, materia_asociada)
JOIN dim_materias m ON m.materia = esp.materia_asociada;

-- DATOS REALES: Distritos Fiscales (requiere IDs de ubicaciones)
INSERT INTO dim_distritos_fiscales (distrito_fiscal, id_ubicacion)
SELECT DISTINCT
    df.distrito_fiscal,
    u.id_ubicacion
FROM (VALUES
('AMAZONAS', 10101),
('ANCASH', 20101),
('APURIMAC', 30101),
('AREQUIPA', 40101),
('AYACUCHO', 50110),
('CAJAMARCA', 60101),
('CALLAO', 70101),
('CAÑETE', 150501),
('CUSCO', 80108),
('HUANCAVELICA', 90101),
('HUANUCO', 100101),
('HUAURA', 150801),
('ICA', 110101),
('JUNIN', 120114),
('LA LIBERTAD', 130101),
('LAMBAYEQUE', 140101),
('LIMA CENTRO', 150101),
('LIMA ESTE', 150137),
('LIMA NOROESTE', 70106),
('LIMA NORTE', 150112),
('LIMA SUR', 150142),
('LORETO', 160101),
('MADRE DE DIOS', 170101),
('MOQUEGUA', 180101),
('PASCO', 190113),
('PIURA', 200101),
('PUNO', 210101),
('SAN MARTIN', 220101),
('SANTA', 21801),
('SELVA CENTRAL', 120301),
('SULLANA', 200601),
('TACNA', 230101),
('TUMBES', 240101),
('UCAYALI', 250101)
) AS df(distrito_fiscal, ubigeo_pjfs)
JOIN dim_ubicaciones u ON u.ubigeo_pjfs = df.ubigeo_pjfs;


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

