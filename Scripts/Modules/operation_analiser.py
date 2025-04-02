import pandas as pd


class OperationLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_excel(self.filepath)
        df.columns = [
            col.strip().lower().replace(' ', '_') for col in df.columns
            ]

        df['data_inicio'] = pd.to_datetime(df['data_inicio'])
        df['data_fim'] = pd.to_datetime(df['data_fim'])

        return df


class OperationAnalyzer:
    def __init__(self, df_paradas, df_forecast, target_col='parada'):
        self.df_paradas = df_paradas
        self.df_forecast = df_forecast.copy()
        self.target_col = target_col

    def marcar_paradas(self):
        self.df_forecast[self.target_col] = 0

        for _, row in self.df_paradas.iterrows():
            mask = (
                self.df_forecast['time'] >= row['data_inicio']
            ) & (self.df_forecast['time'] <= row['data_fim'])
            self.df_forecast.loc[mask, self.target_col] = 1

        return self.df_forecast

    def analisar_variaveis(self, variaveis, export_path: str = None):
        df = self.df_forecast.copy()

        # Separar dados com e sem parada
        df0 = df[df[self.target_col] == 0][variaveis].describe().T
        df1 = df[df[self.target_col] == 1][variaveis].describe().T

        # Renomear colunas para indicar grupo
        df0.columns = [f'{col}_sem_parada' for col in df0.columns]
        df1.columns = [f'{col}_com_parada' for col in df1.columns]

        # Juntar estatísticas lado a lado
        stats = pd.concat([df0, df1], axis=1)

        # Calcular diferença entre médias
        stats['diferenca_media'] = stats['mean_com_parada'] - stats['mean_sem_parada']
        stats = stats.reset_index().rename(columns={'index': 'variavel'})

        # Exportar se solicitado
        if export_path:
            stats.to_csv(export_path, index=False)

        return stats
