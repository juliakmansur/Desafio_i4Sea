import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class RainScenarioClassifier:
    def __init__(self, df, cenarios, base_output, legendas_dict):
        self.df = df
        self.cenarios = cenarios
        self.base_output = base_output
        self.legendas_dict = legendas_dict
        self.metricas_modelos = []

    def aplicar_classificacao(self, modelo_nome="Modelo_01"):
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

            modelo_rf_path = f"{self.base_output}/Modelos/{modelo_nome}/model_randomforest_{nome_cenario}.pkl"
            modelo_lr_path = f"{self.base_output}/Modelos/{modelo_nome}/model_logisticregression_{nome_cenario}.pkl"
            joblib.dump(model_rf, modelo_rf_path)
            joblib.dump(model_lr, modelo_lr_path)
            print(f"Modelos salvos: {modelo_rf_path} e {modelo_lr_path}")

            df_vars = pd.DataFrame({
                'variavel': [v for v, _ in regras],
                'variavel_bin': variaveis_binarias
            })
            for m in ['logisticregression', 'randomforest']:
                var_path = f"{self.base_output}/Modelos/{modelo_nome}/variaveis_{m}_{nome_cenario}.csv"
                df_vars.to_csv(var_path, index=False)

            self._plot_importancia_variaveis(
                model_rf.feature_importances_, X.columns,
                nome_cenario, modelo_nome, tipo='rf'
                )
            coef = model_lr.coef_[0]
            importancias_lr = np.abs(coef)
            self._plot_importancia_variaveis(
                importancias_lr, X.columns,
                nome_cenario, modelo_nome, tipo='lr'
                )

        metricas_df = pd.DataFrame(self.metricas_modelos)
        os.makedirs(f"{self.base_output}/Tables", exist_ok=True)
        metricas_df.to_csv(f"{self.base_output}/Tables/metricas_{modelo_nome.lower()}_por_cenario.csv", index=False)

    def _avaliar_modelo(self, modelo, X_test, y_test, nome_cenario, nome_modelo):
        y_pred = modelo.predict(X_test)
        print(f"\n{nome_modelo}")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão:", confusion_matrix(y_test, y_pred))

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

    def _plot_importancia_variaveis(self, importancias, variaveis, nome_cenario, modelo_nome, tipo='rf'):
        nomes = [self.legendas_dict.get(v.replace('_bin', ''), v) for v in variaveis]
        importancias_series = pd.Series(importancias, index=nomes)

        fig, ax = plt.subplots(figsize=(12, 6))
        importancias_series.sort_values().plot(kind='barh', ax=ax)

        ax.set_title(f"Importância das Variáveis - Cenário {nome_cenario.capitalize()} ({tipo.upper()})")
        ax.set_xlim(0, .7 if tipo == 'rf' else 3)
        plt.tight_layout()
        os.makedirs(f"{self.base_output}/Imagens/Metricas/{modelo_nome}", exist_ok=True)
        plt.savefig(f"{self.base_output}/Imagens/Metricas/{modelo_nome}/importancia_variaveis_{tipo}_{nome_cenario}.png")
        plt.close()
