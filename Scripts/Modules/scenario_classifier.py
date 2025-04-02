import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple


class RainScenarioClassifier:
    """
    Classe responsável por aplicar classificadores supervisionados (Random Forest e Regressão Logística)
    sobre variáveis binárias geradas a partir de thresholds definidos em diferentes cenários.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cenarios: Dict[str, List[Tuple[str, float]]],
        base_output: str,
        legendas_dict: Dict[str, str]
    ):
        self.df = df
        self.cenarios = cenarios
        self.base_output = base_output
        self.legendas_dict = legendas_dict
        self.metricas_modelos = []

    def aplicar_classificacao(self, modelo_nome: str = "Modelo_01") -> None:
        """
        Aplica classificadores RandomForest e LogisticRegression para cada cenário,
        avalia desempenho e salva modelos e gráficos gerados.
        """

        for nome_cenario, regras in self.cenarios.items():
            print(f"\n=== CENÁRIO: {nome_cenario.upper()} ===")

            if hasattr(regras, 'iterrows'):
                regras = list(regras[['variavel', 'threshold']].itertuples(index=False, name=None))

            for var, thresh in regras:
                self.df[f"{var}_bin"] = (self.df[var] > thresh).astype(int)

            variaveis_binarias = [f"{v}_bin" for v, _ in regras]
            X = self.df[variaveis_binarias]
            y = self.df['parada']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=42
            )

            model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_rf.fit(X_train, y_train)

            model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
            model_lr.fit(X_train, y_train)

            self._avaliar_modelo(model_rf, X_test, y_test, nome_cenario, 'RandomForest')
            self._avaliar_modelo(model_lr, X_test, y_test, nome_cenario, 'LogisticRegression')

            # Salvar modelos
            model_dir = os.path.join(self.base_output, "Modelos", modelo_nome)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model_rf, os.path.join(model_dir, f"model_randomforest_{nome_cenario}.pkl"))
            joblib.dump(model_lr, os.path.join(model_dir, f"model_logisticregression_{nome_cenario}.pkl"))
            print(f"Modelos salvos em {model_dir}")

            # Salvar variáveis utilizadas
            df_vars = pd.DataFrame({
                'variavel': [v for v, _ in regras],
                'variavel_bin': variaveis_binarias
            })
            for m in ['logisticregression', 'randomforest']:
                df_vars.to_csv(os.path.join(model_dir, f'variaveis_{m}_{nome_cenario}.csv'), index=False)

            # Importância das variáveis
            self._plot_importancia_variaveis(
                model_rf.feature_importances_, X.columns,
                nome_cenario, modelo_nome, tipo='rf'
            )
            importancias_lr = np.abs(model_lr.coef_[0])
            self._plot_importancia_variaveis(
                importancias_lr, X.columns,
                nome_cenario, modelo_nome, tipo='lr'
            )

        # Salvar métricas globais
        metricas_df = pd.DataFrame(self.metricas_modelos)
        os.makedirs(f"{self.base_output}/Tables", exist_ok=True)
        metricas_df.to_csv(f"{self.base_output}/Tables/metricas_{modelo_nome.lower()}_por_cenario.csv", index=False)

    def _avaliar_modelo(self, modelo, X_test: pd.DataFrame, y_test: pd.Series, nome_cenario: str, nome_modelo: str) -> None:
        """
        Avalia e armazena métricas de classificação do modelo.
        """

        y_pred = modelo.predict(X_test)
        print(f"\n{nome_modelo}")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

        report = classification_report(y_test, y_pred, output_dict=True)
        for classe in ['0', '1']:
            for metric in ['precision', 'recall', 'f1-score']:
                self.metricas_modelos.append({
                    'cenario': nome_cenario,
                    'modelo': nome_modelo,
                    'classe': classe,
                    'metric': metric,
                    'value': report[classe][metric]
                })

    def _plot_importancia_variaveis(
        self, importancias: np.ndarray, variaveis: List[str],
        nome_cenario: str, modelo_nome: str, tipo: str = 'rf'
    ) -> None:
        """
        Plota gráfico de barras horizontal com importâncias das variáveis.
        """

        nomes = [self.legendas_dict.get(v.replace('_bin', ''), v) for v in variaveis]
        importancias_series = pd.Series(importancias, index=nomes)

        fig, ax = plt.subplots(figsize=(12, 6))
        importancias_series.sort_values().plot(kind='barh', ax=ax)

        ax.set_title(f"Importância das Variáveis - Cenário {nome_cenario.capitalize()} ({tipo.upper()})")
        ax.set_xlim(0, .7 if tipo == 'rf' else 3)
        plt.tight_layout()
        path = os.path.join(self.base_output, "Imagens", "Metricas", modelo_nome)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'importancia_variaveis_{tipo}_{nome_cenario}.png'))
        plt.close()
