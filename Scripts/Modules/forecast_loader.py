import os
from glob import glob
from typing import Union, List, Tuple
import xarray as xr
import pandas as pd
import numpy as np


class ForecastLoader:
    """
    Classe para carregamento de dados de previsão atmosférica em arquivos NetCDF,
    com extração da grade mais próxima ao ponto de interesse e estruturação em formato tabular.
    """

    def __init__(self, folder_path: str, lat_target: float, lon_target: float):
        self.folder_path = folder_path
        self.lat_target = lat_target
        self.lon_target = lon_target

    def _find_closest_point(self, ds: xr.Dataset) -> Tuple[int, int]:
        lat = ds['lat'].values
        lon = ds['lon'].values

        distances = np.sqrt((lat - self.lat_target)**2 + (lon - self.lon_target)**2)

        return np.unravel_index(np.argmin(distances), lat.shape)

    def _extract_from_file(self, file_path: str) -> List[dict]:
        ds = xr.open_dataset(file_path)
        i, j = self._find_closest_point(ds)

        if np.issubdtype(ds['time'].dtype, np.floating):
            ds['time'] = pd.to_datetime(ds['time'].values, unit='s')

        time_values = ds['time'].values - np.timedelta64(3, 'h')  # Ajuste UTC-3
        data_vars = list(ds.data_vars)

        records = []
        for t, time_val in enumerate(time_values):
            record = {'time': pd.to_datetime(time_val), 'file': os.path.basename(file_path)}
            for var in data_vars:
                record[var] = float(ds[var][t, i, j].values)
            records.append(record)

        return records

    def _get_month_range(self, start: str, end: str) -> List[str]:
        start_year, start_month = int(start[:4]), int(start[4:])
        end_year, end_month = int(end[:4]), int(end[4:])

        months = []
        for year in range(start_year, end_year + 1):
            m_start = start_month if year == start_year else 1
            m_end = end_month if year == end_year else 12
            for month in range(m_start, m_end + 1):
                months.append(f'{year}{str(month).zfill(2)}')

        return months

    def load_forecasts(self, months: Union[str, List[str], Tuple[str, str]], drop_duplicates: bool = False) -> pd.DataFrame:
        """
        Carrega e organiza os dados de previsão para os meses indicados.

        Parâmetros:
            months: str, lista ou tupla indicando os meses no formato 'YYYYMM'
            drop_duplicates: se True, mantém apenas a última previsão por instante de tempo

        Retorna:
            pd.DataFrame: dados em formato tabular com tempo, variáveis e metadados
        """

        if isinstance(months, str):
            months = [months]
        elif isinstance(months, tuple):
            months = self._get_month_range(months[0], months[1])

        all_data = []
        for month in months:
            pattern = os.path.join(self.folder_path, f'*{month}*.nc')
            files = sorted(glob(pattern))
            for file_path in files:
                all_data.extend(self._extract_from_file(file_path))

        if not all_data:
            raise FileNotFoundError('Nenhum arquivo .nc encontrado para os meses fornecidos.')

        df = pd.DataFrame(all_data)
        df['forecast_datetime'] = df['file'].str.extract(r'(\d{10})')
        df['forecast_datetime'] = pd.to_datetime(df['forecast_datetime'], format='%Y%m%d%H')

        if drop_duplicates:
            df = df.sort_values('forecast_datetime').drop_duplicates(subset='time', keep='last')

        # Reorganizar colunas
        cols = [col for col in df.columns if col not in ['file', 'forecast_datetime']]

        return df[cols + ['file', 'forecast_datetime']]
