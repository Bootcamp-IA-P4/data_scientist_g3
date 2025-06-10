-- Tabla para predicciones de stroke
CREATE TABLE stroke_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    gender VARCHAR(50) NOT NULL,
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 120),
    hypertension BOOLEAN NOT NULL,
    heart_disease BOOLEAN NOT NULL,
    ever_married VARCHAR(50) NOT NULL,
    work_type VARCHAR(100) NOT NULL,
    residence_type VARCHAR(50) NOT NULL,
    avg_glucose_level DECIMAL(6,2) NOT NULL CHECK (avg_glucose_level >= 50 AND avg_glucose_level <= 500),
    bmi DECIMAL(5,2) CHECK (bmi >= 10 AND bmi <= 60),
    smoking_status VARCHAR(50) NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    probability DECIMAL(5,4) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('Bajo', 'Medio', 'Alto', 'Crítico'))
);

-- Índices para optimizar consultas
CREATE INDEX idx_stroke_predictions_created_at ON stroke_predictions(created_at DESC);
CREATE INDEX idx_stroke_predictions_prediction ON stroke_predictions(prediction);
CREATE INDEX idx_stroke_predictions_risk_level ON stroke_predictions(risk_level);
CREATE INDEX idx_stroke_predictions_age ON stroke_predictions(age);

-- Comentarios para documentación
COMMENT ON TABLE stroke_predictions IS 'Predicciones de riesgo de stroke realizadas por el modelo XGBoost';
COMMENT ON COLUMN stroke_predictions.prediction IS '0=Sin riesgo de stroke, 1=Riesgo de stroke';
COMMENT ON COLUMN stroke_predictions.probability IS 'Probabilidad de stroke (0.0 a 1.0)';
COMMENT ON COLUMN stroke_predictions.risk_level IS 'Nivel de riesgo interpretado: Bajo, Medio, Alto, Crítico';-- Tabla para predicciones de stroke
CREATE TABLE stroke_predictions (
    -- Identificador y metadatos
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Patient input data (original format from frontend)
    gender VARCHAR(50) NOT NULL, -- "Masculino", "Femenino"
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 120),
    hypertension BOOLEAN NOT NULL, -- true/false
    heart_disease BOOLEAN NOT NULL, -- true/false
    ever_married VARCHAR(50) NOT NULL, -- "Sí", "No"
    work_type VARCHAR(100) NOT NULL, -- "Empleado Público", "Privado", etc.
    residence_type VARCHAR(50) NOT NULL, -- "Urbano", "Rural"
    avg_glucose_level DECIMAL(6,2) NOT NULL CHECK (avg_glucose_level >= 50 AND avg_glucose_level <= 500),
    bmi DECIMAL(5,2) CHECK (bmi >= 10 AND bmi <= 60), -- NULL permitido
    smoking_status VARCHAR(50) NOT NULL, -- "Nunca fumó", "Fuma", "Fumó antes", "NS/NC"
    
    -- Resultado del modelo
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)), -- 0: No stroke, 1: Stroke
    probability DECIMAL(5,4) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('Bajo', 'Medio', 'Alto', 'Crítico'))
);

-- Índices para optimizar consultas
CREATE INDEX idx_stroke_predictions_created_at ON stroke_predictions(created_at DESC);
CREATE INDEX idx_stroke_predictions_prediction ON stroke_predictions(prediction);
CREATE INDEX idx_stroke_predictions_risk_level ON stroke_predictions(risk_level);
CREATE INDEX idx_stroke_predictions_age ON stroke_predictions(age);

-- Comentarios para documentación
COMMENT ON TABLE stroke_predictions IS 'Predicciones de riesgo de stroke realizadas por el modelo XGBoost';
COMMENT ON COLUMN stroke_predictions.prediction IS '0=Sin riesgo de stroke, 1=Riesgo de stroke';
COMMENT ON COLUMN stroke_predictions.probability IS 'Probabilidad de stroke (0.0 a 1.0)';
COMMENT ON COLUMN stroke_predictions.risk_level IS 'Nivel de riesgo interpretado: Bajo, Medio, Alto, Crítico';