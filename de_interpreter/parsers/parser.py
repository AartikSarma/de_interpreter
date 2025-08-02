"""Simplified omics data parser with auto-detection."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import yaml

from .omics_data import OmicsFeature, OmicsExperimentContext, OmicsType, OmicsColumnMapping


class OmicsParser:
    """Parser that automatically detects and handles all omics types."""
    
    def __init__(self, omics_type: Optional[OmicsType] = None):
        self.omics_type = omics_type
        self._column_mappings = {}
        
    def parse_data(self, file_path: Path) -> List[OmicsFeature]:
        """Parse omics data file and return list of OmicsFeatures."""
        
        # Read the file
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.tsv', '.txt']:
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Auto-detect omics type if not specified
        detected_omics_type = self._detect_omics_type(df)
        omics_type = self.omics_type or detected_omics_type
        
        # Get column mappings for this omics type
        mappings = OmicsColumnMapping.get_mappings_for_omics(omics_type)
        self._column_mappings = self._map_columns(df, mappings)
        
        # Parse data into OmicsFeature objects
        features = []
        for _, row in df.iterrows():
            feature = self._parse_row_to_feature(row, omics_type)
            if feature:
                features.append(feature)
                
        return features
    
    def parse_metadata(self, file_path: Path) -> OmicsExperimentContext:
        """Parse metadata file and return OmicsExperimentContext."""
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                metadata = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported metadata format: {file_path.suffix}")
            
        return self._metadata_to_context(metadata)
    
    def _detect_omics_type(self, df: pd.DataFrame) -> OmicsType:
        """Auto-detect omics type from dataframe columns."""
        columns = [col.lower() for col in df.columns]
        
        # Check for transcriptomics indicators
        if any(term in ' '.join(columns) for term in ['gene', 'transcript', 'ensembl']):
            return OmicsType.TRANSCRIPTOMICS
            
        # Check for proteomics indicators  
        if any(term in ' '.join(columns) for term in ['protein', 'uniprot', 'peptide']):
            return OmicsType.PROTEOMICS
            
        # Check for metabolomics indicators
        if any(term in ' '.join(columns) for term in ['metabolite', 'compound', 'chemical', 'kegg', 'hmdb']):
            return OmicsType.METABOLOMICS
            
        # Check for genomics indicators
        if any(term in ' '.join(columns) for term in ['variant', 'snp', 'mutation', 'rsid']):
            return OmicsType.GENOMICS
            
        # Check for metagenomics indicators
        if any(term in ' '.join(columns) for term in ['species', 'taxon', 'otu', 'asv', 'microbe']):
            return OmicsType.METAGENOMICS
            
        # Check for epigenomics indicators
        if any(term in ' '.join(columns) for term in ['region', 'peak', 'dmr', 'cpg', 'chromatin']):
            return OmicsType.EPIGENOMICS
            
        # Check for lipidomics indicators
        if any(term in ' '.join(columns) for term in ['lipid', 'fatty_acid']):
            return OmicsType.LIPIDOMICS
            
        # Default to transcriptomics
        return OmicsType.TRANSCRIPTOMICS
    
    def _map_columns(self, df: pd.DataFrame, mappings: Dict[str, List[str]]) -> Dict[str, str]:
        """Map dataframe columns to standard names."""
        column_map = {}
        df_columns = df.columns.tolist()
        
        for standard_name, possible_names in mappings.items():
            # First try exact match (case-sensitive)
            for col in df_columns:
                if col in possible_names:
                    column_map[standard_name] = col
                    break
            
            # If no exact match, try case-insensitive
            if standard_name not in column_map:
                for col in df_columns:
                    if col.lower() in [name.lower() for name in possible_names]:
                        column_map[standard_name] = col
                        break
                        
        return column_map
    
    def _parse_row_to_feature(self, row: pd.Series, omics_type: OmicsType) -> Optional[OmicsFeature]:
        """Parse a dataframe row into an OmicsFeature."""
        try:
            # Extract required fields
            feature_id = self._get_value(row, 'feature_id', required=True)
            log2fc = float(self._get_value(row, 'log2FoldChange', required=True))
            pvalue = float(self._get_value(row, 'pvalue', required=True))
            padj = float(self._get_value(row, 'padj', required=True))
            
            # Extract optional fields
            feature_symbol = self._get_value(row, 'feature_symbol')
            base_mean = self._get_value(row, 'baseMean')
            if base_mean is not None:
                base_mean = float(base_mean)
                
            return OmicsFeature(
                feature_id=str(feature_id),
                feature_symbol=feature_symbol,
                feature_name=None,
                omics_type=omics_type,
                log2_fold_change=log2fc,
                p_value=pvalue,
                padj=padj,
                base_mean=base_mean
            )
            
        except (ValueError, TypeError, KeyError):
            # Skip rows with missing/invalid required data
            return None
    
    def _get_value(self, row: pd.Series, standard_name: str, required: bool = False) -> Any:
        """Get value from row using column mapping."""
        if standard_name in self._column_mappings:
            col_name = self._column_mappings[standard_name]
            value = row.get(col_name)
            if pd.isna(value):
                value = None
            return value
        elif required:
            raise KeyError(f"Required column '{standard_name}' not found")
        return None
    
    def _metadata_to_context(self, metadata: Dict[str, Any]) -> OmicsExperimentContext:
        """Convert metadata dict to OmicsExperimentContext."""
        # Auto-detect omics type from metadata if not specified
        omics_type_str = metadata.get('omics_type', 'transcriptomics')
        try:
            omics_type = OmicsType(omics_type_str.lower())
        except ValueError:
            omics_type = OmicsType.TRANSCRIPTOMICS
            
        # Override with parser's omics type if specified
        if self.omics_type:
            omics_type = self.omics_type
            
        return OmicsExperimentContext(
            omics_type=omics_type,
            disease=metadata.get('disease', ''),
            tissue=metadata.get('tissue', ''),
            cell_type=metadata.get('cell_type', ''),
            treatment=metadata.get('treatment', ''),
            control=metadata.get('control', ''),
            organism=metadata.get('organism', 'human'),
            comparison_description=metadata.get('comparison_description', 
                f"{metadata.get('treatment', 'treatment')} vs {metadata.get('control', 'control')}"),
            platform=metadata.get('platform'),
            analysis_method=metadata.get('analysis_method'),
            normalization=metadata.get('normalization'),
            additional_info=metadata.get('additional_info', {})
        )
    
    def get_summary_stats(self, features: List[OmicsFeature]) -> Dict[str, Any]:
        """Get summary statistics for parsed features."""
        if not features:
            return {}
            
        total_features = len(features)
        significant_features = sum(1 for f in features if f.is_significant)
        upregulated = sum(1 for f in features if f.is_significant and f.is_upregulated)
        downregulated = sum(1 for f in features if f.is_significant and f.is_downregulated)
        
        # Get omics-specific info
        omics_type = features[0].omics_type if features else OmicsType.TRANSCRIPTOMICS
        feature_type_names = {
            OmicsType.TRANSCRIPTOMICS: "genes",
            OmicsType.PROTEOMICS: "proteins", 
            OmicsType.METABOLOMICS: "metabolites",
            OmicsType.GENOMICS: "variants",
            OmicsType.METAGENOMICS: "microbes",
            OmicsType.EPIGENOMICS: "genomic regions",
            OmicsType.LIPIDOMICS: "lipids"
        }
        
        return {
            'total_features': total_features,
            'significant_features': significant_features,
            'upregulated': upregulated,
            'downregulated': downregulated,
            'omics_type': omics_type.value,
            'feature_type': feature_type_names.get(omics_type, 'features'),
            'feature_type_name': feature_type_names.get(omics_type, 'features')
        }
    
    def parse(self, de_file: str, metadata_file: Optional[str] = None) -> tuple[List[OmicsFeature], OmicsExperimentContext]:
        """Unified parsing method for convenience."""
        de_path = Path(de_file)
        
        # Parse features
        features = self.parse_data(de_path)
        
        # Parse metadata if provided
        if metadata_file:
            metadata_path = Path(metadata_file)
            context = self.parse_metadata(metadata_path)
        else:
            # Create default context based on detected omics type
            omics_type = features[0].omics_type if features else OmicsType.TRANSCRIPTOMICS
            context = OmicsExperimentContext(
                omics_type=omics_type,
                disease="Unknown",
                tissue="Unknown",
                cell_type="Unknown",
                treatment="Treatment",
                control="Control",
                organism="human",
                comparison_description="Treatment vs Control"
            )
        
        return features, context