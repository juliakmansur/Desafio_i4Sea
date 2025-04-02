import seaborn as sns
import matplotlib.pyplot as plt


class ModelPlotter:
    def __init__(self, df):
        self.df = df.copy()
        self.modelos_legiveis = {
            'RandomForest': 'Random Forest',
            'LogisticRegression': 'Regressão Logística'
        }

    def _preparar_dados(self, filtro_classe=None):
        df_filtrado = self.df
        if filtro_classe is not None:
            df_filtrado = df_filtrado[df_filtrado['classe'] == filtro_classe]

        df_filtrado = df_filtrado.copy()
        df_filtrado['modelo_legivel'] = df_filtrado['modelo'].map(self.modelos_legiveis)
        return df_filtrado

    def plotar_metricas(self, titulo, output_path, filtro_classe=None, col=None, row=None, legenda_fora=True):
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
