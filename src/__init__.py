"""
Pipeline de predicción de stroke
"""

from .pipeline.stroke_pipeline import predict_stroke_risk, get_pipeline_status

__all__ = ['predict_stroke_risk', 'get_pipeline_status']
