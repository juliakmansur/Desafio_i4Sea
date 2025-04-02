import pandas as pd
from typing import List


class OperationLoader:
    """
    Classe responsável por carregar os dados de paradas operacionais a partir de planilhas Excel.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """
        Carrega e padroniza os dados da planilha de paradas.

        Retorna:
            pd.DataFrame: dataframe com colunas padronizadas e datas convertidas
        """

        df = pd.read_excel(self.filepath)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df['data_inicio'] = pd.to_datetime(df['data_inicio'])
        df['data_fim'] = pd.to_datetime(df['data_fim'])

        return df


class OperationAnalyzer:
    """
    Classe responsável por marcar períodos de parada em dados de previsão e analisar diferenças estatísticas
    entre momentos com e sem operação.
    """

    def __init__(self, df_paradas: pd.DataFrame, df_forecast: pd.DataFrame, target_col: str = 'parada'):
        self.df_paradas = df_paradas
        self.df_forecast = df_forecast.copy()
        self.target_col = target_col

    def marcar_paradas(self) -> pd.DataFrame:
        """
        Adiciona coluna binária indicando se há parada para cada linha da previsão.

        Retorna:
            pd.DataFrame: dataframe com coluna target_col preenchida
        """

        self.df_forecast[self.target_col] = 0
        for _, row in self.df_paradas.iterrows():
            mask = (self.df_forecast['time'] >= row['data_inicio']) & (self.df_forecast['time'] <= row['data_fim'])
            self.df_forecast.loc[mask, self.target_col] = 1

        return self.df_forecast

    def analisar_variaveis(self, variaveis: List[str], export_path: str = None) -> pd.DataFrame:
        """
        Compara estatísticas descritivas entre períodos com e sem parada para as variáveis selecionadas.

        Parâmetros:
            variaveis (list): lista de nomes de colunas a serem analisadas
            export_path (str): caminho para salvar a tabela em .csv (opcional)

        Retorna:
            pd.DataFrame: tabela comparando média, std, min, max etc.
        """

        df = self.df_forecast.copy()
        df0 = df[df[self.target_col] == 0][variaveis].describe().T
        df1 = df[df[self.target_col] == 1][variaveis].describe().T

        df0.columns = [f'{col}_sem_parada' for col in df0.columns]
        df1.columns = [f'{col}_com_parada' for col in df1.columns]

        stats = pd.concat([df0, df1], axis=1)
        stats['diferenca_media'] = stats['mean_com_parada'] - stats['mean_sem_parada']
        stats = stats.reset_index().rename(columns={'index': 'variavel'})

        if export_path:
            stats.to_csv(export_path, index=False)

        return stats
