A cotinuacion se muestra el proceso de descarga e instalacion del proyecto.

git clone https://github.com/tuusuario/tu-repo.git
cd tu-repo

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt

# Configurar credenciales (crear config.json a partir de config.example.json)

streamlit run app.py
