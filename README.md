A cotinuacion se muestra el proceso de descarga e instalacion del proyecto.

git clone https://github.com/tuusuario/tu-repo.git
cd tu-repo

python -m venv .venv

# Windows
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
