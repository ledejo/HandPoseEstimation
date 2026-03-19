"""
Feature extraction for GMM clustering.
Reuses the same feature extraction logic as DBSCAN for consistency.
"""

from ..DBSCAN.features_dbscan import DbscanFeatureExtractor

# GMM uses identical feature extraction as DBSCAN
GmmFeatureExtractor = DbscanFeatureExtractor

__all__ = ["GmmFeatureExtractor"]
