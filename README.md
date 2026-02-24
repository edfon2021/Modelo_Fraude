# Dashboard de Detección de Fraude (Streamlit)

Dashboard interactivo para visualización y evaluación de métricas de un modelo de detección de fraude en transacciones, desarrollado con **Streamlit** y conectado a **PostgreSQL**.

## Características
- Visualización de datos y métricas del modelo
- Matriz de confusión y métricas derivadas
- Curvas ROC / PR (si están implementadas)
- Conexión a base de datos PostgreSQL
- Estructura modular (`views/`, `evaluation/`, etc.)
---

## Requisitos previos

Antes de ejecutar el proyecto, asegurate de tener instalado:

- **Python** (recomendado: 3.14.0)
- **Git**
- Acceso a la **base de datos PostgreSQL** (si el dashboard consulta datos reales)

> En Windows, podés verificar Python con:
```bash
python --version

## Clonar Repositorio

A cotinuacion se muestra el proceso de descarga e instalacion del proyecto.

git clone https://github.com/edfon2021/Modelo_Fraude.git
cd Modelo_Fraude

## Creamos el entorno virtual.
python -m venv modelo

# Lo activamos e instalamos los requerimientos.
modelo\Scripts\activate
pip install -r requirements.txt

## Ejecutamos el app usando Strimlite.
streamlit run app.py

Nota: Se requiere python 3.14.0 para que corra correctamente.
