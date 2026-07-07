import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from scipy import interpolate
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Raw data processor for gas pipeline network fluid dynamics data.
    
    Handles:
    - Loading CSV files from train/test directories
    - Parsing boundary conditions and equipment outputs
    - Time series alignment (30-min boundary → 1-min predictions)
    - Data cleaning and validation
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize DataProcessor.
        
        Args:
            data_dir: Path to data directory containing dataset folder
        """
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "dataset"
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        
        # Equipment file types and their expected column counts (excluding TIME)
        self.equipment_info = {
            'B': 2058,     # Ball valves
            'C': 161,      # Compressors  
            'H': 192,      # Pipeline segments
            'N': 1716,     # Nodes
            'P': 1610,     # Pipelines
            'R': 50,       # Control valves
            'T&E': 387     # Gas sources and distribution points
        }
        
        # Total prediction dimensions: 6,174
        self.total_prediction_dims = sum(self.equipment_info.values())
        
        # Boundary conditions: 539 total columns (including TIME), 538 data columns
        self.boundary_dims = 538  # Non-TIME columns only

        # Static equipment parameters
        self.static_equipment_data = None
        self.pipe_features = None
        self.compressor_features = None

        # Common time index for 1-minute equipment series (00:01-23:59)
        self.full_minute_range = pd.date_range(
            start='2025-01-01 00:01:00',
            end='2025-01-01 23:59:00',
            freq='1min'
        )
        self.equipment_missing_fill = -1.0

        self._validate_directories()
        self._load_static_equipment_data()
        self._boundary_column_order: List[str] = []
        self._equipment_column_order: Dict[str, List[str]] = {}
    
    def _standardize_time_column(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize TIME column to 2025/1/1 format for consistent processing.
        
        Args:
            df: DataFrame with TIME column
            data_type: "boundary" (30-min intervals) or "equipment" (1-min intervals)
            
        Returns:
            DataFrame with standardized TIME column
        """
        if 'TIME' not in df.columns:
            return df
            
        df = df.copy()
        
        if data_type == "boundary":
            # Boundary: 30-min intervals from 00:00 to 23:30, extend to 23:59
            expected_times = pd.date_range(
                start='2025-01-01 00:00:00', 
                end='2025-01-01 23:30:00', 
                freq='30min'
            )
            
            # First, ensure numeric columns are properly converted
            numeric_cols = [col for col in df.columns if col != 'TIME']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Create standardized boundary data by reindexing to expected times
            df_with_std_time = df.copy()
            df_with_std_time['TIME'] = expected_times[:len(df)]
            
            # Extend boundary data from 23:30 to 23:59 by copying 23:30 values  
            extend_times = pd.date_range(
                start='2025-01-01 23:31:00',
                end='2025-01-01 23:59:00', 
                freq='1min'
            )
            
            # Create extension data by repeating the last row's values
            if len(df_with_std_time) > 0:
                last_row = df_with_std_time.iloc[-1].copy()
                extend_data = []
                for extend_time in extend_times:
                    row_data = last_row.copy()
                    row_data['TIME'] = extend_time
                    extend_data.append(row_data)
                
                extend_df = pd.DataFrame(extend_data)
                df = pd.concat([df_with_std_time, extend_df], ignore_index=True)
            else:
                df = df_with_std_time
            
        else:  # equipment data
            # Equipment: 1-min intervals from 00:01 to 23:59
            expected_times = pd.date_range(
                start='2025-01-01 00:01:00',
                end='2025-01-01 23:59:00', 
                freq='1min'
            )
            
            # First, convert TIME column to datetime for filtering
            df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
            
            # Filter to keep only 2025-01-01 data from 00:01 to 23:59
            # Use datetime comparison instead of time comparison to avoid next day 00:01
            target_start = pd.Timestamp('2025-01-01 00:01:00')
            target_end = pd.Timestamp('2025-01-01 23:59:00')
            
            # Keep only rows within the target datetime range
            time_mask = (df['TIME'] >= target_start) & (df['TIME'] <= target_end)
            df = df[time_mask].reset_index(drop=True)
        
            # Now ensure numeric columns are properly converted
            for col in df.columns:
                if col != 'TIME':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check if we have the expected number of time points after filtering
            if len(df) != len(expected_times):
                logger.warning(f"Equipment data length {len(df)} does not match expected {len(expected_times)} after filtering")
                logger.warning(f"Time range in data: {df['TIME'].min()} to {df['TIME'].max()}")
            
            # Assign the expected time range if lengths match
            if len(df) == len(expected_times):
                df['TIME'] = expected_times
            else:
                # If lengths don't match, keep original times but warn
                logger.warning(f"Keeping original TIME values due to length mismatch")
            
        # Ensure TIME is datetime
        df['TIME'] = pd.to_datetime(df['TIME'])
        
        logger.debug(f"Standardized TIME column for {data_type}: {len(df)} time points from {df['TIME'].iloc[0]} to {df['TIME'].iloc[-1]}")
        return df
    
    def _detect_nan_values(self, df: pd.DataFrame, data_name: str) -> None:
        """
        Detect and log NaN values in DataFrame.
        
        Args:
            df: DataFrame to check
            data_name: Name for logging
        """
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            nan_cols = df.isnull().any()
            logger.warning(f"NaN found in {data_name}:")
            logger.warning(f"  - Total NaN values: {nan_count}")
            logger.warning(f"  - Columns with NaN: {nan_cols[nan_cols].index.tolist()}")
            
            # Show first few rows with NaN
            nan_rows = df.isnull().any(axis=1)
            if nan_rows.any():
                first_nan_idx = nan_rows.idxmax()
                logger.warning(f"  - First NaN at row {first_nan_idx}")

    def _validate_directories(self) -> None:
        """Validate that required directories exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def _load_static_equipment_data(self):
        """加载静态设备参数数据"""
        try:
            from .process_eq_argu import StaticEquipmentProcessor

            static_processor = StaticEquipmentProcessor(str(self.data_dir))
            self.static_equipment_data = static_processor.load_processed_data()

            self.pipe_features = self.static_equipment_data['pipe_features']
            self.compressor_features = self.static_equipment_data['compressor_features']

            logger.info(f"Loaded static equipment data:")
            logger.info(f"  - Pipe features: {self.pipe_features.shape}")
            logger.info(f"  - Compressor features: {self.compressor_features.shape}")

        except Exception as e:
            logger.warning(f"Failed to load static equipment data: {e}")
            self.static_equipment_data = None
            self.pipe_features = None
            self.compressor_features = None
    
    def get_sample_directories(self, split: str = 'train') -> List[Path]:
        """
        Get list of sample directories for given split.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            List of sample directory paths
        """
        if split == 'train':
            base_dir = self.train_dir
        elif split == 'test':
            base_dir = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        
        sample_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        sample_dirs.sort()  # Sort by name for consistent ordering
        
        logger.info(f"Found {len(sample_dirs)} {split} samples")
        return sample_dirs
    
    def load_boundary_data(self, sample_dir: Path) -> pd.DataFrame:
        """
        Load boundary conditions from Boundary.csv.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            DataFrame with boundary conditions (TIME + 539 boundary parameters)
            TIME standardized to 2025/1/1 00:01-23:59 format with 30-min intervals
        """
        boundary_file = sample_dir / "Boundary.csv"
        if not boundary_file.exists():
            raise FileNotFoundError(f"Boundary.csv not found in {sample_dir}")
        
        df = pd.read_csv(boundary_file)
        
        # Validate dimensions (including TIME column)
        expected_cols = self.boundary_dims + 1  # +1 for TIME column
        if df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} columns in {boundary_file}, got {df.shape[1]}")
        
        # Standardize TIME column to 2025/1/1 format
        df = self._standardize_time_column(df, "boundary")

        self._boundary_column_order = [col for col in df.columns if col != "TIME"]

        logger.debug(f"Loaded boundary data: {df.shape} from {sample_dir.name}")
        return df
    
    def load_equipment_data(self, sample_dir: Path, equipment_type: str) -> Optional[pd.DataFrame]:
        """
        Load equipment prediction data from specific CSV file.
        
        Args:
            sample_dir: Path to sample directory
            equipment_type: Equipment type ('B', 'C', 'H', 'N', 'P', 'R', 'T&E')
            
        Returns:
            DataFrame with equipment predictions or None if file doesn't exist (test data)
            TIME standardized to 2025/1/1 00:01-23:59 format with 1-min intervals
        """
        csv_file = sample_dir / f"{equipment_type}.csv"
        
        # Test data may not have equipment files
        if not csv_file.exists():
            return None
        
        df = pd.read_csv(csv_file)
        
        # Validate dimensions
        expected_cols = self.equipment_info[equipment_type] + 1  # +1 for TIME
        if df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} columns in {csv_file}, got {df.shape[1]}")
        
        # Standardize TIME column to 2025/1/1 format
        df = self._standardize_time_column(df, "equipment")

        value_columns = [col for col in df.columns if col != "TIME"]
        self._equipment_column_order[equipment_type] = value_columns

        logger.debug(f"Loaded {equipment_type} data: {df.shape} from {sample_dir.name}")
        return df
    
    def load_all_equipment_data(self, sample_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all equipment data for a sample.

        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Dictionary mapping equipment type to DataFrame
        """
        equipment_data = {}
        
        # Add progress bar for equipment loading
        equipment_types = list(self.equipment_info.keys())
        for equipment_type in tqdm(equipment_types, desc=f"Loading equipment data from {sample_dir.name}", unit="file", leave=False):
            data = self.load_equipment_data(sample_dir, equipment_type)
            if data is not None:
                equipment_data[equipment_type] = data

        return equipment_data

    @property
    def boundary_column_order(self) -> List[str]:
        return list(self._boundary_column_order)

    def equipment_column_order(self, equipment_type: str) -> List[str]:
        return list(self._equipment_column_order.get(equipment_type, []))
    
    def combine_equipment_data(self, equipment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all equipment data into single DataFrame.
        
        Args:
            equipment_data: Dictionary of equipment DataFrames
            
        Returns:
            Combined DataFrame with all equipment predictions
        """
        if not equipment_data:
            return pd.DataFrame()
        
        # Get TIME from first available equipment data
        time_col = None
        for df in equipment_data.values():
            if 'TIME' in df.columns:
                time_col = df['TIME'].copy()
                break
        
        # Combine all non-TIME columns
        combined_data = {'TIME': time_col} if time_col is not None else {}
        
        for equipment_type, df in equipment_data.items():
            # Add all columns except TIME
            for col in df.columns:
                if col != 'TIME':
                    combined_data[col] = df[col]
        
        combined_df = pd.DataFrame(combined_data)
        
        # Validate total dimensions
        expected_cols = self.total_prediction_dims + 1  # +1 for TIME
        if combined_df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} total columns, got {combined_df.shape[1]}")
        
        return combined_df
    
    def load_sample_data(self, sample_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load complete data for a single sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Tuple of (boundary_data, equipment_data)
            equipment_data is None for test samples
        """
        # Load boundary conditions
        boundary_data = self.load_boundary_data(sample_dir)
        
        # Load equipment data (may be None for test data)
        equipment_dict = self.load_all_equipment_data(sample_dir)
        equipment_data = self.combine_equipment_data(equipment_dict) if equipment_dict else None
        
        return boundary_data, equipment_data
    
    def validate_data_consistency(self, boundary_data: pd.DataFrame, 
                                 equipment_data: Optional[pd.DataFrame]) -> bool:
        """
        Validate consistency between boundary and equipment data.
        
        Args:
            boundary_data: Boundary conditions DataFrame
            equipment_data: Equipment predictions DataFrame (may be None)
            
        Returns:
            True if data is consistent
        """
        if equipment_data is None:
            return True  # Test data doesn't have equipment data
        
        # Check time series length
        if len(boundary_data) != len(equipment_data):
            logger.error(f"Length mismatch: boundary={len(boundary_data)}, equipment={len(equipment_data)}")
            return False
        
        # Check time alignment if both have TIME columns
        if 'TIME' in boundary_data.columns and 'TIME' in equipment_data.columns:
            if not boundary_data['TIME'].equals(equipment_data['TIME']):
                logger.error("TIME columns are not aligned between boundary and equipment data")
                return False
        
        # Expected: 1440 time steps (24 hours * 60 minutes)
        expected_timesteps = 1440
        if len(boundary_data) != expected_timesteps:
            logger.warning(f"Expected {expected_timesteps} timesteps, got {len(boundary_data)}")
        
        return True
    
    def get_data_statistics(self, split: str = 'train') -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with dataset statistics
        """
        sample_dirs = self.get_sample_directories(split)
        
        stats = {
            'split': split,
            'num_samples': len(sample_dirs),
            'boundary_dims': self.boundary_dims,
            'total_prediction_dims': self.total_prediction_dims,
            'equipment_dims': self.equipment_info.copy(),
            'expected_timesteps': 1440,
            'samples': []
        }
        
        # Check first few samples for validation
        for i, sample_dir in enumerate(sample_dirs[:3]):
            try:
                boundary_data, equipment_data = self.load_sample_data(sample_dir)
                
                sample_stats = {
                    'name': sample_dir.name,
                    'boundary_shape': boundary_data.shape,
                    'equipment_shape': equipment_data.shape if equipment_data is not None else None,
                    'valid': self.validate_data_consistency(boundary_data, equipment_data)
                }
                
                stats['samples'].append(sample_stats)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_dir.name}: {e}")
                stats['samples'].append({
                    'name': sample_dir.name,
                    'error': str(e)
                })
        
        return stats

    def interpolate_boundary_to_1min(self, boundary_df: pd.DataFrame) -> pd.DataFrame:
        """
        将已标准化的boundary数据(含23:31-23:59扩展)插值到1分钟间隔。
        
        Args:
            boundary_df: 已标准化的boundary数据, shape [N, 539+1], TIME列 + 539个boundary参数
                        TIME已从00:00扩展到23:59(每30分钟+每1分钟扩展)
            
        Returns:
            插值后的数据, shape [1439, 539+1], 每分钟从00:01-23:59的时间点
        """
        if 'TIME' not in boundary_df.columns:
            raise ValueError("Boundary data must contain TIME column")
        
        df = boundary_df.copy()
        
        # 确保TIME列是datetime格式
        df['TIME'] = pd.to_datetime(df['TIME'])
        
        # 创建目标1分钟间隔时间索引：从00:01到23:59 (1439个时间点)
        target_time_range = pd.date_range(
            start='2025-01-01 00:01:00',
            end='2025-01-01 23:59:00', 
            freq='1min'
        )
        
        # 检测原始数据NaN
        self._detect_nan_values(df, f"boundary_df before interpolation")
        
        # 设置TIME为索引并构造完整分钟级索引
        df_indexed = df.set_index('TIME')

        full_time_range = pd.date_range(
            start='2025-01-01 00:00:00',
            end='2025-01-01 23:59:00',
            freq='1min'
        )

        # 使用前向填充保持最近一次边界条件
        reindexed = df_indexed.reindex(full_time_range)
        interpolated = reindexed.ffill()

        # 如果开头仍存在缺失值，使用后向填充兜底
        interpolated = interpolated.bfill()

        # 仅保留00:01-23:59的时间范围
        interpolated = interpolated.loc[target_time_range]
        
        # 最终检查插值结果
        if interpolated.isnull().any().any():
            self._detect_nan_values(interpolated.reset_index(), "interpolated boundary data")
            logger.error("Interpolation failed - NaN values remain after fillna")
        
        # 重置索引，TIME重新作为列
        interpolated_df = interpolated.reset_index()
        interpolated_df.rename(columns={'index': 'TIME'}, inplace=True)
        
        # 验证输出时间范围
        expected_len = 1439  # 00:01 to 23:59
        if len(interpolated_df) != expected_len:
            logger.warning(f"Interpolated data has {len(interpolated_df)} time points, expected {expected_len}")
        
        logger.debug(f"Interpolated boundary data from {len(df)} to {len(interpolated_df)} time points")
        logger.debug(f"Time range: {interpolated_df['TIME'].iloc[0]} to {interpolated_df['TIME'].iloc[-1]}")
        return interpolated_df

    def _align_equipment_minute_grid(self, df: pd.DataFrame, equipment_type: str) -> pd.DataFrame:
        """
        Ensure equipment data covers every minute of the target day by padding missing timestamps.
        """
        if 'TIME' not in df.columns:
            raise ValueError(f"{equipment_type} data缺少TIME列，无法对齐到分钟网格")

        aligned = df.copy()
        aligned['TIME'] = pd.to_datetime(aligned['TIME'], errors='coerce')
        aligned = aligned.dropna(subset=['TIME'])
        aligned = aligned.sort_values('TIME')
        aligned = aligned.set_index('TIME').reindex(self.full_minute_range)
        numeric_cols = aligned.columns
        missing_before_fill = aligned[numeric_cols].isnull().sum().sum()
        if missing_before_fill > 0:
            logger.info(
                "%s 缺失 %d 个观测，使用 %.1f 填充缺口",
                equipment_type,
                int(missing_before_fill),
                self.equipment_missing_fill,
            )
        aligned[numeric_cols] = aligned[numeric_cols].fillna(self.equipment_missing_fill)
        aligned[numeric_cols] = aligned[numeric_cols].astype(np.float32)
        aligned = aligned.reset_index().rename(columns={'index': 'TIME'})
        return aligned
    
    def combine_all_data(self, boundary_df: pd.DataFrame, equipment_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并boundary和所有equipment数据为单一DataFrame。

        Args:
            boundary_df: 插值后的boundary数据, shape [1439, 539+1]
            equipment_dict: 各设备类型的数据字典, 每个shape [1439, N+1]

        Returns:
            合并后的数据, shape [1439, 6713+1], TIME列 + 6712个变量 (538 boundary + 6174 equipment)
        """
        aligned_equipment = {}
        for eq_type, eq_df in equipment_dict.items():
            aligned_equipment[eq_type] = self._align_equipment_minute_grid(eq_df, eq_type)
        equipment_dict = aligned_equipment

        # 验证所有数据的时间对齐
        expected_time_series = pd.Series(self.full_minute_range)

        # 检查boundary时间对齐
        if not boundary_df['TIME'].equals(expected_time_series):
            logger.error("Boundary data TIME column is not aligned with expected 00:01-23:59 range")

        # 检查所有equipment时间对齐
        for eq_type, eq_df in equipment_dict.items():
            if 'TIME' in eq_df.columns:
                if not eq_df['TIME'].equals(expected_time_series):
                    logger.error(f"{eq_type} equipment data TIME column is not aligned with expected range")

        # 获取TIME列作为基础
        time_col = boundary_df['TIME'].copy()

        # 加载 index_variable_mapping.csv 以确保列顺序与 attention indices 一致
        mapping_file = self.data_dir / "static" / "full" / "index_variable_mapping.csv"
        if not mapping_file.exists():
            mapping_file = self.data_dir / "index_variable_mapping.csv"
        # 从映射文件读取正确的列顺序
        mapping_df = pd.read_csv(mapping_file)
        # 按 index 列排序确保顺序正确
        mapping_df = mapping_df.sort_values('index')
        logger.debug(f"Loaded variable mapping with {len(mapping_df)} variables")

        # 收集所有可用的列数据
        all_available_data = {}

        # 添加boundary数据
        boundary_numeric = boundary_df.drop(columns=['TIME'])
        for col in boundary_numeric.columns:
            all_available_data[col] = boundary_numeric[col]

        # 添加equipment数据
        for eq_type, eq_df in equipment_dict.items():
            eq_numeric = eq_df.drop(columns=['TIME'], errors='ignore')
            for col in eq_numeric.columns:
                all_available_data[col] = eq_numeric[col]

        # 按照映射中的顺序重新组织数据，这里是列的重新组织数据
        combined_data = {'TIME': time_col}
        missing_equipment = []
        for _, row in mapping_df.iterrows():
            var_name = row['variable_name']
            if var_name in all_available_data:
                combined_data[var_name] = all_available_data[var_name]
            else:
                var_index = int(row['index']) if 'index' in row else None
                if var_index is not None and var_index < self.boundary_dims:
                    raise ValueError(f"Boundary variable {var_name} 缺失，无法继续组合数据")
                missing_equipment.append(var_name)
                combined_data[var_name] = pd.Series(
                    np.full(len(time_col), self.equipment_missing_fill, dtype=np.float32),
                    index=time_col.index
                )

        if missing_equipment:
            logger.info(
                "Filled missing equipment variables with sentinel value: %s",
                missing_equipment[:10] + (["..."] if len(missing_equipment) > 10 else [])
            )

        combined_df = pd.DataFrame(combined_data)
        
        # 使用helper检测NaN
        self._detect_nan_values(combined_df, "combined data")
        
        # 验证维度
        expected_cols = 1 + self.boundary_dims + self.total_prediction_dims  # TIME + 538 + 6174 = 6713
        if combined_df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} total columns, got {combined_df.shape[1]}")
            logger.warning(f"Boundary dims: {self.boundary_dims}, Equipment dims: {self.total_prediction_dims}")
        
        logger.debug(f"Combined data shape: {combined_df.shape}")
        return combined_df
    
    def create_prediction_mask(self) -> np.ndarray:
        """
        创建预测mask数组: boundary变量=0(不预测), equipment变量=1(需要预测)。
        
        Returns:
            mask数组, shape [6712], boundary部分为0，equipment部分为1
        """
        boundary_mask = np.zeros(self.boundary_dims, dtype=np.int32)  # 538个0
        equipment_mask = np.ones(self.total_prediction_dims, dtype=np.int32)  # 6174个1
        
        mask = np.concatenate([boundary_mask, equipment_mask])
        
        logger.debug(f"Created prediction mask: {len(mask)} total, {np.sum(mask)} predictable")
        return mask
    
    def create_causal_attention_mask(self, sequence_length: int) -> np.ndarray:
        """
        创建因果注意力mask用于decoder。
        
        Args:
            sequence_length: 时间步长 T
            
        Returns:
            attention_mask: shape [T, V], 每个时间步的变量attention mask
                - Boundary变量 (前538列): 只和自己时间步交互 (对角mask)  
                - Equipment变量 (后6174列): 因果mask，可以看到当前及之前时间步
        """
        T = sequence_length
        V = self.boundary_dims + self.total_prediction_dims  # 6712
        
        # 创建 [T, V] 的mask
        attention_mask = np.zeros((T, V), dtype=np.float32)
        
        # Boundary变量 (前538列): 对角mask，只和自己时间步交互
        for t in range(T):
            attention_mask[t, :self.boundary_dims] = 0.0  # Boundary不参与预测
        
        # Equipment变量 (后6174列): 因果mask
        for t in range(T):
            # 当前时间步可以看到当前及之前所有时间步的设备变量
            attention_mask[t, self.boundary_dims:] = 1.0  # 设备变量参与预测
        
        logger.debug(f"Created causal attention mask: [{T}, {V}]")
        return attention_mask
    
    def create_sequences(
        self,
        full_data: pd.DataFrame,
        sequence_length: int = 3,
        time_step_offset: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
        """
        从完整数据创建时序序列对 (input, target)。
        
        Args:
            full_data: 合并后的完整数据, shape [T, 6713] (TIME + 6712个变量)
            sequence_length: 序列长度，默认3分钟
            time_step_offset: 预测序列相对输入向后偏移的分钟数
            
        Returns:
            序列列表，每个元素为 (input_seq, target_seq, start_time, end_time)
            - input_seq: shape [sequence_length, 6712]
            - target_seq: shape [sequence_length, 6712] 
            - start_time, end_time: 序列的时间范围
        """
        if time_step_offset < 1:
            raise ValueError("time_step_offset must be >= 1")

        min_required = sequence_length + time_step_offset
        if len(full_data) < min_required:
            raise ValueError(
                f"Data length {len(full_data)} insufficient for sequence_length {sequence_length} "
                f"with time_step_offset {time_step_offset}"
            )
        
        sequences = []
        
        # 提取数值数据 (排除TIME列)
        numeric_data = full_data.drop(columns=['TIME']).values  # [T, 6712]
        time_data = full_data['TIME'].values
        
        # 滑动窗口生成序列对 - 添加进度条
        num_sequences = len(numeric_data) - min_required + 1
        if num_sequences <= 0:
            raise ValueError(
                f"Unable to build sequences: len={len(numeric_data)}, sequence_length={sequence_length}, "
                f"time_step_offset={time_step_offset}"
            )

        sequence_range = range(num_sequences)
        for i in tqdm(sequence_range, desc="Creating sequences", unit="seq", leave=False):
            input_seq = numeric_data[i:i+sequence_length]           # [T, 6712]
            start_target = i + time_step_offset
            target_seq = numeric_data[start_target:start_target+sequence_length]      # [T, 6712] - 向后偏移 time_step_offset 分钟
            
            start_time = time_data[i]
            end_time = time_data[i+sequence_length-1]
            
            sequences.append((input_seq, target_seq, start_time, end_time))
        
        logger.debug(
            f"Created {len(sequences)} sequences of length {sequence_length} with time_step_offset={time_step_offset}"
        )
        return sequences
    
    def load_raw_sample_data(self, sample_dir: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """
        加载单个样本的原始合并数据(不创建序列)。

        Args:
            sample_dir: 样本目录路径

        Returns:
            Tuple of (raw_data, metadata)
            - raw_data: np.ndarray of shape [1439, 6712] or None if error
            - metadata: Dict containing 'sample_id', 'start_time', 'end_time', 'has_nan'
        """
        try:
            # 加载boundary数据
            boundary_data = self.load_boundary_data(sample_dir)

            # 检测boundary数据中的NaN
            self._detect_nan_values(boundary_data, f"boundary_data from {sample_dir.name}")

            # 插值boundary数据到1分钟间隔
            boundary_1min = self.interpolate_boundary_to_1min(boundary_data)

            # 检测插值后boundary数据中的NaN
            self._detect_nan_values(boundary_1min, f"interpolated boundary_1min from {sample_dir.name}")

            # 加载所有equipment数据
            equipment_dict = self.load_all_equipment_data(sample_dir)

            # 检测equipment数据中的NaN
            for eq_type, eq_data in equipment_dict.items():
                self._detect_nan_values(eq_data, f"{eq_type} equipment data from {sample_dir.name}")

            # 验证数据存在性
            if not equipment_dict:
                # 测试数据可能没有equipment数据
                logger.info(f"No equipment data found for {sample_dir.name} (likely test data)")
                return None, {'sample_id': sample_dir.name, 'error': 'No equipment data'}

            # 合并所有数据
            combined_data = self.combine_all_data(boundary_1min, equipment_dict)

            # 提取数值数据（排除TIME列）
            numeric_data = combined_data.drop(columns=['TIME']).values  # [1439, 6712]
            time_data = combined_data['TIME'].values

            # 检查NaN
            has_nan = np.isnan(numeric_data).any()
            if has_nan:
                nan_count = np.isnan(numeric_data).sum()
                logger.warning(f"Sample {sample_dir.name} contains {nan_count} NaN values "
                             f"({nan_count / numeric_data.size * 100:.2f}%)")

            metadata = {
                'sample_id': sample_dir.name,
                'start_time': time_data[0],
                'end_time': time_data[-1],
                'has_nan': has_nan
            }

            logger.debug(f"Loaded raw sample data from {sample_dir.name}: shape {numeric_data.shape}")
            return numeric_data, metadata

        except Exception as e:
            logger.error(f"Error loading raw sample data from {sample_dir}: {e}")
            return None, {'sample_id': sample_dir.name, 'error': str(e)}

    def load_combined_sample_data(
        self,
        sample_dir: Path,
        sequence_length: int = 3,
        time_step_offset: int = 1,
    ) -> Tuple[List[Tuple], Optional[np.ndarray]]:
        """
        加载单个样本的完整合并数据并创建序列。

        Args:
            sample_dir: 样本目录路径
            sequence_length: 时序序列长度
            time_step_offset: 预测序列向后偏移的分钟数

        Returns:
            Tuple of (sequences, prediction_mask)
            - sequences: List of (input, target, start_time, end_time) tuples
            - prediction_mask: shape [6712] 预测mask数组
        """
        try:
            # 加载boundary数据
            boundary_data = self.load_boundary_data(sample_dir)

            # 检测boundary数据中的NaN
            self._detect_nan_values(boundary_data, f"boundary_data from {sample_dir.name}")

            # 插值boundary数据到1分钟间隔
            boundary_1min = self.interpolate_boundary_to_1min(boundary_data)

            # 检测插值后boundary数据中的NaN
            self._detect_nan_values(boundary_1min, f"interpolated boundary_1min from {sample_dir.name}")

            # 加载所有equipment数据
            equipment_dict = self.load_all_equipment_data(sample_dir)

            # 检测equipment数据中的NaN
            for eq_type, eq_data in equipment_dict.items():
                self._detect_nan_values(eq_data, f"{eq_type} equipment data from {sample_dir.name}")

            # 验证数据存在性
            if not equipment_dict:
                # 测试数据可能没有equipment数据
                logger.info(f"No equipment data found for {sample_dir.name} (likely test data)")
                return [], None

            # 合并所有数据
            combined_data = self.combine_all_data(boundary_1min, equipment_dict)

            # 创建时序序列
            sequences = self.create_sequences(combined_data, sequence_length, time_step_offset)

            # 检测序列中的NaN
            if sequences:
                for i, (input_seq, target_seq, start_time, end_time) in enumerate(sequences[:3]):  # 检查前3个序列
                    if np.isnan(input_seq).any():
                        nan_count_input = np.isnan(input_seq).sum()
                        logger.error(f"Sequence {i} input contains {nan_count_input} NaN values")
                        # 找到NaN位置
                        nan_positions = np.where(np.isnan(input_seq))
                        logger.error(f"NaN positions in input_seq: rows={nan_positions[0][:10]}, cols={nan_positions[1][:10]}")

                    if np.isnan(target_seq).any():
                        nan_count_target = np.isnan(target_seq).sum()
                        logger.error(f"Sequence {i} target contains {nan_count_target} NaN values")
                        # 找到NaN位置
                        nan_positions = np.where(np.isnan(target_seq))
                        logger.error(f"NaN positions in target_seq: rows={nan_positions[0][:10]}, cols={nan_positions[1][:10]}")

            # 创建预测mask
            prediction_mask = self.create_prediction_mask()

            logger.debug(f"Loaded {len(sequences)} sequences from {sample_dir.name}")
            return sequences, prediction_mask

        except Exception as e:
            logger.error(f"Error loading combined data from {sample_dir}: {e}")
            return [], None

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from TIME column.
        
        Args:
            df: DataFrame with TIME column
            
        Returns:
            DataFrame with additional time features
        """
        if 'TIME' not in df.columns:
            return df
        
        df = df.copy()
        
        # Extract time features
        df['hour'] = df['TIME'].dt.hour
        df['minute'] = df['TIME'].dt.minute
        df['day_of_week'] = df['TIME'].dt.dayofweek
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        return df
