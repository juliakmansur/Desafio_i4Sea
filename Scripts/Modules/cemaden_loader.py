import os
from glob import glob
from typing import Union, List, Tuple
import pandas as pd


class CemadenLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.df = None

    def _get_available_months(self) -> List[str]:
        files = glob(os.path.join(self.folder_path, 'Paranagua_*.csv'))
        return sorted([f.split('_')[-1].replace('.csv', '') for f in files])

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

    def load(self,  months: Union[str, List[str], Tuple[str, str]], station_code: str = None) -> pd.DataFrame:
        if isinstance(months, str):
            months = [months]
        elif isinstance(months, tuple):
            months = self._get_month_range(months[0], months[1])

        dataframes = []
        for m in months:
            filepath = os.path.join(self.folder_path, f'Paranagua_{m}.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, sep=';', encoding='utf-8')

            df['valorMedida'] = df['valorMedida'].str.replace(',', '.', regex=False).astype(float)
            df['latitude'] = df['latitude'].str.replace(',', '.', regex=False).astype(float)
            df['longitude'] = df['longitude'].str.replace(',', '.', regex=False).astype(float)

            df['month'] = m

            dataframes.append(df)

        if not dataframes:
            raise FileNotFoundError('Nenhum arquivo correspondete aos meses informados foi encontrado')

        self.df = pd.concat(dataframes, ignore_index=True)

        if station_code:
            self.df = self.df[self.df['codEstacao'] == station_code]

        return self.df

    def filter_by_station(self, station_name):
        if self.df is None:
            raise ValueError('Os dados ainda não foram carregados. Use o método load() primeiro.')
        return self.df[self.df['nomeEstacao'].str.contains(station_name, case=False)]

    def filter_by_period(self, start_date, end_date):
        if self.df is None:
            raise ValueError('Os dados ainda não foram carregados. Use o método load() primeiro.')
        return self.df[(self.df['datahora'] >= start_date) & (self.df['datahora'] <= end_date)]


class CemadenMerger:
    def __init__(self, obs_df, forecast_df, tolerance='15min'):
        self.obs_df = obs_df.copy()
        self.forecast_df = forecast_df.copy()
        self.tolerance = tolerance

    def merge_data(self):
        obs = self.obs_df.rename(columns={'datahora': 'time', 'valorMedida': 'precipacao_obs'})
        obs['time'] = pd.to_datetime(obs['time'])
        self.forecast_df['time'] = pd.to_datetime(self.forecast_df['time'])

        merged = pd.merge_asof(
            self.forecast_df.sort_values('time'),
            obs.sort_values('time'),
            on='time',
            direction='nearest',
            tolerance=pd.Timedelta(self.tolerance)
        )
        return merged.dropna()
