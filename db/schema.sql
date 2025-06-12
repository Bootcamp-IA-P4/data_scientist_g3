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

-- Índices para optimizar consultas
CREATE INDEX idx_stroke_predictions_created_at ON stroke_predictions(created_at DESC);
CREATE INDEX idx_stroke_predictions_prediction ON stroke_predictions(prediction);
CREATE INDEX idx_stroke_predictions_risk_level ON stroke_predictions(risk_level);
CREATE INDEX idx_stroke_predictions_age ON stroke_predictions(age);
CREATE INDEX idx_stroke_predictions_hypertension ON stroke_predictions(hypertension);
CREATE INDEX idx_stroke_predictions_heart_disease ON stroke_predictions(heart_disease);

-- Crear función para formatear fecha en español
CREATE OR REPLACE FUNCTION format_date_spanish(timestamp_input TIMESTAMP WITH TIME ZONE)
RETURNS TEXT AS $$
BEGIN
    RETURN to_char(timestamp_input AT TIME ZONE 'Europe/Madrid', 'DD/MM/YYYY HH24:MI');
END;
$$ LANGUAGE plpgsql;

-- Crear vista con fecha formateada en español
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