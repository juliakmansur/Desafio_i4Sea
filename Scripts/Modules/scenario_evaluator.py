import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
from typing import List, Tuple, Dict


class RainScenarioModel:
    """
    Classe responsável por aplicar regras binárias para diferentes cenários operacionais,
    avaliando métricas de desempenho e gerando visualizações para tomada de decisão.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            cenarios: Dict[str, List[Tuple[str, float]]],
            base_output: str,
            legendas_dict: Dict[str, str],
            modelo_nome: str = "Modelo_01"
    ):
        self.df = df
        self.cenarios = cenarios
        self.base_output = base_output
        self.legendas_dict = legendas_dict
        self.modelo_nome = modelo_nome

    def ajustar_legenda(self, ax):
        """Ajusta a legenda para visualização com múltiplas variáveis."""

        handles, labels = ax.get_legend_handles_labels()
        labels = [l.replace(') ', ')\n') for l in labels]
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='small')

    def aplicar_cenarios(self):
        """Aplica os cenários definidos ao dataframe e gera outputs de avaliação."""

        for nome, regras in self.cenarios.items():
            if hasattr(regras, 'iterrows'):
                try:
                    regras_list = [(r['variavel'], r['threshold']) for _, r in regras.iterrows()]
                except KeyError:
                    cols = regras.columns.tolist()
                    var_col = next((c for c in cols if 'var' in c.lower()), cols[0])
                    thresh_col = next((c for c in cols if 'threshold' in c.lower()), cols[1])
                    regras_list = [(r[var_col], r[thresh_col]) for _, r in regras.iterrows()]
            else:
                regras_list = regras

            self.df[f'cenario_{nome}'] = self._avaliar_regras(regras_list)
            self._avaliar_metricas(nome, regras_list)
            self._plotar_matriz_confusao(nome)
            self._plotar_series_temporais(nome, regras_list)

    def _avaliar_regras(self, regras_list: List[Tuple[str, float]]) -> pd.Series:
        """Aplica regra binária de decisão por variável (threshold)."""
        condicoes = [(self.df[var] > thresh) for var, thresh in regras_list]
        decisao = condicoes[0]
        for c in condicoes[1:]:
            decisao &= c
        return decisao.astype(int)

    def _avaliar_metricas(self, nome: str, regras_list: List[Tuple[str, float]]):
        """Imprime as métricas de classificação para o cenário."""

        y_true = self.df['parada']
        y_pred = self.df[f'cenario_{nome}']
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f'\nCenário: {nome.upper()}')
        print('Variáveis usadas:')
        print(regras_list)
        print(f'Precision: {p:.2f} | Recall: {r:.2f} | F1 Score: {f1:.2f}')

    def _plotar_matriz_confusao(self, nome: str):
        """Gera e salva matriz de confusão para o cenário."""

        y_true = self.df['parada']
        y_pred = self.df[f'cenario_{nome}']

        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax,
            display_labels=['Sem Parada', 'Com Parada'],
            cmap='Blues'
        )
        ax.set_title(f'Matriz de Confusão - Cenário {nome.capitalize()}')
        plt.ylabel('Observação')
        plt.xlabel('Modelo')
        plt.tight_layout()

        path = os.path.join(self.base_output, 'Imagens', self.modelo_nome, 'Metricas')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'matriz_cenario_{nome}.png'))
        plt.close()

    def _plotar_series_temporais(self, nome: str, regras_list: List[Tuple[str, float]]):
        """Gera visualizações temporais para o cenário, separando por tipo de variável."""

        variaveis = [v for v, _ in regras_list]
        tipos = ['prob' if 'probability' in v else 'lwe' for v in variaveis]
        misto = 'prob' in tipos and 'lwe' in tipos

        path = os.path.join(self.base_output, 'Imagens', self.modelo_nome, 'Serie_temporais')
        os.makedirs(path, exist_ok=True)

        if misto:
            self._plotar_misto(nome, variaveis, path)
        else:
            self._plotar_simples(nome, variaveis, path)

    def _plotar_misto(self, nome: str, variaveis: List[str], path: str):
        """Plota séries temporais de probabilidade e precipitação separadas."""

        fig, axs = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

        def plot_vars(ax, vars_list, fill_column, fill_label, color, title, ylabel):
            for var in vars_list:
                axs[ax].plot(self.df['time'], self.df[var], label=self.legendas_dict.get(var, var), linewidth=1)
            axs[ax].fill_between(
                self.df['time'], 0, 1,
                where=self.df[fill_column] == 1,
                color=color, alpha=0.15,
                transform=axs[ax].get_xaxis_transform(), label=fill_label
            )
            axs[ax].set_title(title)
            axs[ax].set_ylabel(ylabel)
            axs[ax].set_ylim(bottom=0)
            axs[ax].grid(True, linestyle='--', alpha=0.3)
            axs[ax].set_xlim(self.df['time'].min(), self.df['time'].max())
            self.ajustar_legenda(axs[ax])

        prob_vars = [v for v in variaveis if 'probability' in v]
        lwe_vars = [v for v in variaveis if 'lwe' in v]

        plot_vars(0, prob_vars, 'parada', 'Parada Real', 'deepskyblue', 'Probabilidade - Parada Real', 'Probabilidade (%)')
        plot_vars(1, prob_vars, f'cenario_{nome}', 'Sugestão Modelo', 'crimson', 'Probabilidade - Sugestão Modelo', 'Probabilidade (%)')
        plot_vars(2, lwe_vars, 'parada', 'Parada Real', 'deepskyblue', 'Precipitação - Parada Real', 'Precipitação (mm/h)')
        plot_vars(3, lwe_vars, f'cenario_{nome}', 'Sugestão Modelo', 'crimson', 'Precipitação - Sugestão Modelo', 'Precipitação (mm/h)')

        axs[3].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axs[3].xaxis.get_majorticklabels(), rotation=45)
        fig.suptitle(f'Série Temporal - Parada Real x Decisão Modelo ({nome.capitalize()})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'serie_temporal_comparada_{nome}.png'))
        plt.close()

    def _plotar_simples(self, nome: str, variaveis: List[str], path: str):
        """Plota séries temporais simples (apenas um tipo de variável)."""

        fig, axs = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

        for var in variaveis:
            axs[0].plot(self.df['time'], self.df[var], label=self.legendas_dict.get(var, var), linewidth=1)
            axs[1].plot(self.df['time'], self.df[var], label=self.legendas_dict.get(var, var), linewidth=1)

        axs[0].fill_between(self.df['time'], 0, 1, where=self.df['parada'] == 1, color='deepskyblue', alpha=0.15,
                            transform=axs[0].get_xaxis_transform(), label='Parada Real')
        axs[1].fill_between(self.df['time'], 0, 1, where=self.df[f'cenario_{nome}'] == 1, color='crimson', alpha=0.15,
                            transform=axs[1].get_xaxis_transform(), label='Sugestão Modelo')

        for ax in axs:
            ylabel = 'Probabilidade (%)' if all('probability' in v for v in variaveis) else 'Precipitação (mm/h)'
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', alpha=0.3)
            self.ajustar_legenda(ax)

        axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axs[1].set_xlim(self.df['time'].min(), self.df['time'].max())
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)
        fig.suptitle(f'Série Temporal - Parada Real x Decisão Modelo ({nome.capitalize()})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'serie_temporal_comparada_{nome}.png'))
        plt.close()
