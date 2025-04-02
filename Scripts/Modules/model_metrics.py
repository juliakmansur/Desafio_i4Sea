import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class ModelPlotter:
    """
    Classe responsável por gerar gráficos de desempenho (precision, recall, F1) de modelos
    para diferentes cenários e classes com base em dados já estruturados.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.modelos_legiveis = {
            'RandomForest': 'Random Forest',
            'LogisticRegression': 'Regressão Logística'
        }

    def _preparar_dados(self, filtro_classe: str = None) -> pd.DataFrame:
        """
        Prepara o dataframe para plotagem, filtrando por classe e mapeando nomes legíveis.

        Parâmetros:
            filtro_classe (str): classe binária ('0' ou '1') para filtrar

        Retorna:
            pd.DataFrame: dataframe pronto para visualização
        """

        df_filtrado = self.df
        if filtro_classe is not None:
            df_filtrado = df_filtrado[df_filtrado['classe'] == filtro_classe]

        df_filtrado = df_filtrado.copy()
        df_filtrado['modelo_legivel'] = df_filtrado['modelo'].map(self.modelos_legiveis)
        return df_filtrado

    def plotar_metricas(
        self,
        titulo: str,
        output_path: str,
        filtro_classe: str = None,
        col: str = None,
        row: str = None,
        legenda_fora: bool = True
    ) -> None:
        """
        Gera gráfico de barras para comparação de métricas de diferentes modelos.

        Parâmetros:
            titulo (str): título do gráfico
            output_path (str): caminho de saída para salvar a imagem
            filtro_classe (str): '0' ou '1' para filtrar classe específica (opcional)
            col (str): coluna para facetar em subplots horizontais (ex: 'cenario')
            row (str): linha para facetar em subplots verticais (ex: 'classe')
            legenda_fora (bool): se True, coloca legenda fora do gráfico
        """

        df_plot = self._preparar_dados(filtro_classe)

        g = sns.catplot(
            data=df_plot,
            x='metric',
            y='value',
            hue='modelo_legivel',
            col=col,
            row=row,
            kind='bar',
            palette='Set2',
            height=4,
            aspect=1.1,
            sharey=True
        )

        if col and not row:
            g.set_titles('Cenário: {col_name}')
        elif row:
            g.set_titles('Cenário: {col_name} | Classe: {row_name}')

        g.set_axis_labels('Métrica', 'Desempenho do Modelo')
        g.set(ylim=(0, 1))

        if g._legend:
            g._legend.set_title(None)
            if legenda_fora:
                g._legend.set_bbox_to_anchor((.86, 0.5))
                g._legend.set_loc('center left')

        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(0)

        plt.suptitle(titulo, y=1.03, fontsize=14)
        g.tight_layout()
        g.savefig(output_path)
        plt.close()
