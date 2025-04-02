
import os
import joblib
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from Modules.variables_info import variables_info


class RecomendadorOperacional:
    """
    Classe para previsão de paradas operacionais com base em arquivos NetCDF de previsão e modelos treinados.
    """

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.variaveis_por_cenario = self._carregar_variaveis_por_cenario()

    def _carregar_variaveis_por_cenario(self) -> dict:
        """
        Recupera variáveis usadas em cada cenário a partir dos arquivos .csv exportados.
        """

        variaveis = {}
        for nome_arquivo in os.listdir(self.base_model_path):
            if nome_arquivo.startswith('variaveis_') and nome_arquivo.endswith('.csv'):
                partes = nome_arquivo.replace('variaveis_', '').replace('.csv', '').split('_')
                if len(partes) == 2:
                    modelo, cenario = partes
                    key = f'{modelo.lower()}_{cenario}'
                    df = pd.read_csv(os.path.join(self.base_model_path, nome_arquivo))
                    variaveis[key] = df['variavel'].tolist()

        return variaveis

    def carregar_modelo(self, modelo: str, cenario: str):
        """
        Carrega o modelo treinado (RandomForest ou LogisticRegression) salvo em disco.
        """

        nome_arquivo = f'model_{modelo.lower()}_{cenario}.pkl'
        caminho = os.path.join(self.base_model_path, nome_arquivo)
        if not os.path.exists(caminho):
            raise FileNotFoundError(f'Modelo não encontrado: {caminho}')
        return joblib.load(caminho)

    def _find_closest_point(self, ds: xr.Dataset, lat_target: float, lon_target: float):
        lat = ds['lat'].values
        lon = ds['lon'].values
        distances = np.sqrt((lat - lat_target)**2 + (lon - lon_target)**2)

        return np.unravel_index(np.argmin(distances), lat.shape)

    def extrair_variaveis_netcdf(self, path_nc: str, variaveis: list, lat: float, lon: float) -> pd.DataFrame:
        """
        Extrai as variáveis relevantes de um arquivo NetCDF para a posição mais próxima.
        """
        ds = xr.open_dataset(path_nc)
        lat_idx, lon_idx = self._find_closest_point(ds, lat, lon)

        dados = {}
        for var in variaveis:
            if var in ds:
                valores = ds[var][:, lat_idx, lon_idx].values
                dados[var] = valores
            else:
                raise ValueError(f'Variável "{var}" não encontrada no NetCDF.')

        if 'time' in ds:
            time_values = pd.to_datetime(ds['time'].values, unit='s') - np.timedelta64(3, 'h')
            dados['time'] = time_values
        else:
            raise ValueError('Variável "time" não encontrada no NetCDF.')

        return pd.DataFrame(dados)

    def prever(self, path_nc: str, cenario: str, modelo: str, lat: float, lon: float) -> pd.DataFrame:
        """
        Executa previsão binária de parada para um cenário e modelo em um arquivo NetCDF.
        """

        key = f"{modelo.lower()}_{cenario}"
        if key not in self.variaveis_por_cenario:
            raise ValueError(f"Variáveis não encontradas para modelo '{modelo}' e cenário '{cenario}'.")

        clf = self.carregar_modelo(modelo, cenario)
        variaveis_originais = self.variaveis_por_cenario[key]
        df = self.extrair_variaveis_netcdf(path_nc, variaveis_originais, lat, lon)

        thresholds = pd.read_csv("../Output/Analise_02/Tables/melhores_thresholds.csv")
        thresholds = thresholds[thresholds["variavel"].isin(variaveis_originais)].set_index("variavel")

        variaveis_bin = []
        for var in variaveis_originais:
            thresh = thresholds.loc[var, "threshold"]
            col_bin = f"{var}_bin"
            df[col_bin] = (df[var] > thresh).astype(int)
            variaveis_bin.append(col_bin)

        df["parada_recomendada"] = clf.predict(df[variaveis_bin])

        return df[['time'] + variaveis_originais + variaveis_bin + ['parada_recomendada']]


class RecomendadorVisual:
    """
    Classe para visualização dos resultados de previsão de paradas em forma de série temporal.
    """

    def __init__(self):
        self.legendas_dict = {
            var: leg.replace(')', ')\n') if ')' in leg else leg
            for var, leg in zip(variables_info['variaveis'], variables_info['legendas'])
        }

    def gerar_relatorio(self, resultado: pd.DataFrame, modelo: str, cenario: str, nome_arquivo: str = None, salvar_em: str = None) -> None:
        """
        Gera gráfico e exporta .csv com série temporal e janelas de recomendação.
        """
        variaveis_prob = [col for col in resultado.columns if 'probability' in col and not col.endswith('_bin')]
        variaveis_lwe = [col for col in resultado.columns if 'lwe' in col and not col.endswith('_bin')]

        n_subplots = int(bool(variaveis_prob)) + int(bool(variaveis_lwe))
        fig, axs = plt.subplots(n_subplots, 1, figsize=(14, 4 * n_subplots), sharex=True)
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]

        idx = 0
        if variaveis_prob:
            for var in variaveis_prob:
                axs[idx].plot(resultado['time'], resultado[var], label=self.legendas_dict.get(var, var), linewidth=2)
            axs[idx].fill_between(
                resultado['time'], 0, 1,
                where=resultado['parada_recomendada'] == 1,
                transform=axs[idx].get_xaxis_transform(),
                color='crimson', alpha=0.2, label='Parada Recomendada'
            )
            axs[idx].set_ylabel('Probabilidade (%)')
            axs[idx].set_ylim(0, 100)
            axs[idx].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
            axs[idx].grid(True)
            idx += 1

        if variaveis_lwe:
            for var in variaveis_lwe:
                axs[idx].plot(resultado['time'], resultado[var], label=self.legendas_dict.get(var, var), linewidth=2)
            axs[idx].fill_between(
                resultado['time'], 0, 1,
                where=resultado['parada_recomendada'] == 1,
                transform=axs[idx].get_xaxis_transform(),
                color='crimson', alpha=0.2, label='Parada Recomendada'
            )
            axs[idx].set_ylabel('Precipitação (mm/h)')
            axs[idx].set_ylim(bottom=0)
            axs[idx].legend(loc='upper left', bbox_to_anchor=(1.01, 1))
            axs[idx].grid(True)

        axs[-1].set_xlabel('Tempo')
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh'))
        plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45)

        nome_amigavel = {'LogisticRegression': 'Regressão Logística', 'RandomForest': 'Random Forest'}.get(modelo, modelo)
        fig.suptitle(f'Série Temporal - Parada Recomendada ({nome_amigavel} - {cenario.capitalize()})', fontsize=16, y=.94)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if salvar_em and nome_arquivo:
            os.makedirs(salvar_em, exist_ok=True)
            resultado.to_csv(os.path.join(salvar_em, f'{nome_arquivo}.csv'), index=False)
            fig.savefig(os.path.join(salvar_em, f'{nome_arquivo}.png'))

            resultado_sorted = resultado.sort_values('time')
            resultado_sorted['grupo'] = (resultado_sorted['parada_recomendada'] != resultado_sorted['parada_recomendada'].shift()).cumsum()
            grupos = resultado_sorted[resultado_sorted['parada_recomendada'] == 1].groupby('grupo')

            consolidado = grupos.agg(
                Incio_da_Parada=('time', 'min'),
                Fim_da_Parada=('time', 'max')
            ).reset_index(drop=True)
            consolidado['Recomendacao'] = 'Parar operação'
            consolidado.to_csv(os.path.join(salvar_em, f'{nome_arquivo}_recomendacoes.csv'), index=False)

            print(f'Relatório salvo em: {os.path.join(salvar_em, f"{nome_arquivo}.png")}')
        else:
            plt.show()
