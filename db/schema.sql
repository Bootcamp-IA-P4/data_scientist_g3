-- Crear secuencia para ID autoincremental
CREATE SEQUENCE stroke_predictions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

-- Tabla para predicciones de stroke
CREATE TABLE stroke_predictions (
    id INTEGER DEFAULT nextval('stroke_predictions_id_seq'::regclass) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    gender VARCHAR(50) NOT NULL CHECK (gender IN ('Masculino', 'Femenino', 'Otro')),
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 120),
    hypertension VARCHAR(50) NOT NULL CHECK (hypertension IN ('Sí', 'No')),
    heart_disease VARCHAR(50) NOT NULL CHECK (heart_disease IN ('Sí', 'No')),
    ever_married VARCHAR(50) NOT NULL CHECK (ever_married IN ('Sí', 'No')),
    work_type VARCHAR(100) NOT NULL CHECK (work_type IN ('Empleado Público', 'Privado', 'Autónomo', 'Niño', 'Nunca trabajó')),
    residence_type VARCHAR(50) NOT NULL CHECK (residence_type IN ('Urbano', 'Rural')),
    avg_glucose_level DECIMAL(6,2) NOT NULL CHECK (avg_glucose_level >= 50 AND avg_glucose_level <= 500),
    bmi DECIMAL(5,2) CHECK (bmi >= 10 AND bmi <= 60),
    smoking_status VARCHAR(50) NOT NULL CHECK (smoking_status IN ('Nunca fumó', 'Fuma', 'Fumó antes', 'NS/NC')),
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability DECIMAL(5,4) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('Bajo', 'Medio', 'Alto', 'Crítico'))
);

-- Secuencia para ID autoincremental de imágenes
CREATE SEQUENCE image_predictions_id_seq
START WITH 1
INCREMENT BY 1
NO MINVALUE
NO MAXVALUE
CACHE 1;

-- Tabla para predicciones de imágenes
CREATE TABLE image_predictions (
    id INTEGER DEFAULT nextval('image_predictions_id_seq'::regclass) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Información de la imagen
    image_filename VARCHAR(255) NOT NULL,
    image_url TEXT NOT NULL,
    image_size INTEGER,
    image_format VARCHAR(10),
    
    -- Resultado del modelo de red neuronal
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability DECIMAL(5,4) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('Bajo', 'Medio', 'Alto', 'Crítico')),
    
    -- Metadatos adicionales
    model_confidence DECIMAL(5,4),
    processing_time_ms INTEGER,
    
    -- Relación opcional con predicciones de datos clínicos
    stroke_prediction_id INTEGER REFERENCES stroke_predictions(id)
);

-- Índices para optimizar consultas de stroke
CREATE INDEX idx_stroke_predictions_created_at ON stroke_predictions(created_at DESC);
CREATE INDEX idx_stroke_predictions_prediction ON stroke_predictions(prediction);
CREATE INDEX idx_stroke_predictions_risk_level ON stroke_predictions(risk_level);
CREATE INDEX idx_stroke_predictions_age ON stroke_predictions(age);
CREATE INDEX idx_stroke_predictions_hypertension ON stroke_predictions(hypertension);
CREATE INDEX idx_stroke_predictions_heart_disease ON stroke_predictions(heart_disease);

-- Índices para optimizar consultas de imágenes
CREATE INDEX idx_image_predictions_created_at ON image_predictions(created_at DESC);
CREATE INDEX idx_image_predictions_prediction ON image_predictions(prediction);
CREATE INDEX idx_image_predictions_risk_level ON image_predictions(risk_level);
CREATE INDEX idx_image_predictions_stroke_id ON image_predictions(stroke_prediction_id);

-- Crear función para formatear fecha en español
CREATE OR REPLACE FUNCTION format_date_spanish(timestamp_input TIMESTAMP WITH TIME ZONE)
RETURNS TEXT AS $$
BEGIN
    RETURN to_char(timestamp_input AT TIME ZONE 'Europe/Madrid', 'DD/MM/YYYY HH24:MI');
END;
$$ LANGUAGE plpgsql;

-- Crear vista con fecha formateada para stroke
CREATE VIEW stroke_predictions_formatted AS
SELECT 
    id,
    format_date_spanish(created_at) AS fecha_creacion,
    gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose_level,
    bmi,
    smoking_status,
    prediction,
    probability,
    risk_level,
    created_at
FROM stroke_predictions;

-- Crear vista con fecha formateada para imágenes
CREATE VIEW image_predictions_formatted AS
SELECT
    id,
    format_date_spanish(created_at) AS fecha_creacion,
    image_filename,
    image_url,
    image_size,
    image_format,
    prediction,
    probability,
    risk_level,
    model_confidence,
    processing_time_ms,
    stroke_prediction_id,
    created_at
FROM image_predictions;

-- Comentarios para tabla de stroke
COMMENT ON TABLE stroke_predictions IS 'Predicciones de riesgo de stroke realizadas por el modelo XGBoost';
COMMENT ON COLUMN stroke_predictions.gender IS 'Género del paciente: Masculino, Femenino, Otro';
COMMENT ON COLUMN stroke_predictions.hypertension IS 'Presencia de hipertensión: Sí, No';
COMMENT ON COLUMN stroke_predictions.heart_disease IS 'Presencia de enfermedad cardíaca: Sí, No';
COMMENT ON COLUMN stroke_predictions.ever_married IS 'Estado civil (alguna vez casado): Sí, No';
COMMENT ON COLUMN stroke_predictions.work_type IS 'Tipo de trabajo: Empleado Público, Privado, Autónomo, Niño, Nunca trabajó';
COMMENT ON COLUMN stroke_predictions.residence_type IS 'Tipo de residencia: Urbano, Rural';
COMMENT ON COLUMN stroke_predictions.smoking_status IS 'Estado de fumador: Nunca fumó, Fuma, Fumó antes, NS/NC';
COMMENT ON COLUMN stroke_predictions.prediction IS '0=Sin riesgo de stroke, 1=Riesgo de stroke';
COMMENT ON COLUMN stroke_predictions.probability IS 'Probabilidad de stroke (0.0 a 1.0)';
COMMENT ON COLUMN stroke_predictions.risk_level IS 'Nivel de riesgo interpretado: Bajo, Medio, Alto, Crítico';

-- Comentarios para tabla de imágenes
COMMENT ON TABLE image_predictions IS 'Predicciones de riesgo de stroke basadas en análisis de imágenes con redes neuronales';
COMMENT ON COLUMN image_predictions.image_filename IS 'Nombre original del archivo de imagen';
COMMENT ON COLUMN image_predictions.image_url IS 'URL de la imagen almacenada en Supabase Storage';
COMMENT ON COLUMN image_predictions.prediction IS '0=Sin riesgo de stroke, 1=Riesgo de stroke detectado en imagen';
COMMENT ON COLUMN image_predictions.probability IS 'Probabilidad de stroke detectada por red neuronal (0.0 a 1.0)';
COMMENT ON COLUMN image_predictions.risk_level IS 'Nivel de riesgo interpretado: Bajo, Medio, Alto, Crítico';
COMMENT ON COLUMN image_predictions.stroke_prediction_id IS 'Referencia opcional a predicción clínica del mismo paciente';