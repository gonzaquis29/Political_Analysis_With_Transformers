import pytest
from fastapi.testclient import TestClient
from main import app  # Suponiendo que tu archivo principal se llama main.py
import json

# Usamos TestClient de FastAPI para simular las peticiones HTTP
client = TestClient(app)

@pytest.fixture(scope="module")
def setup():
    """Fixture para configurar recursos compartidos si es necesario."""
    # Puedes configurar algo aquí si es necesario (por ejemplo, crear mock data)
    yield
    # Aquí puedes agregar la limpieza si es necesario

def test_analyze_text_success(setup):
    """Prueba el análisis de texto exitoso con la API."""

    # Datos de prueba
    test_text = {
        "text": "El gobierno debe respetar las libertades individuales, pero también necesita regular la economía para asegurar el bienestar de todos."
    }

    # Realizamos una solicitud POST al endpoint /analyze_text
    response = client.post("/analyze_text", json=test_text)

    # Comprobamos que la respuesta tiene un código de estado 200
    assert response.status_code == 200

    # Comprobamos que el cuerpo de la respuesta contiene la estructura esperada
    data = response.json()
    
    assert "predictions" in data
    assert "global_metrics" in data
    
    # Verificar que los puntajes de las predicciones son enteros y entre -1, 0, 1
    for prediction in data["predictions"]:
        assert prediction["personal_liberty"] in [-1, 0, 1]
        assert prediction["economic_liberty"] in [-1, 0, 1]
    
    # Verificar que las métricas globales sean numéricas
    assert isinstance(data["global_metrics"]["avg_personal_score"], (int, float))
    assert isinstance(data["global_metrics"]["avg_economic_score"], (int, float))
    assert data["global_metrics"]["nolan_category"] in ["Libertarian", "Liberal", "Conservative", "Authoritarian"]

def test_analyze_text_empty_input(setup):
    """Prueba de análisis de texto con una cadena vacía."""
    
    # Datos de prueba (texto vacío)
    test_text = {
        "text": ""
    }

    # Realizamos una solicitud POST al endpoint /analyze_text
    response = client.post("/analyze_text", json=test_text)

    # Comprobamos que la respuesta tiene un código de estado 200
    assert response.status_code == 200

    # Comprobamos que la respuesta contiene predicciones vacías
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 0
    
    # Verificar que las métricas globales estén definidas
    assert "global_metrics" in data
    assert isinstance(data["global_metrics"]["avg_personal_score"], (int, float))
    assert isinstance(data["global_metrics"]["avg_economic_score"], (int, float))

def test_analyze_text_missing_model(setup):
    """Prueba de análisis de texto cuando el modelo no está cargado."""

    # Forzamos que el modelo no se cargue, simulando un fallo en la inicialización
    from main import model
    model = None  # Simulamos que el modelo no se ha cargado

    test_text = {
        "text": "Un texto de prueba."
    }

    response = client.post("/analyze_text", json=test_text)

    assert response.status_code == 500
    assert response.json() == {"detail": "Model not loaded"}

def test_health_check(setup):
    """Prueba para verificar la salud del servicio."""

    # Realizamos una solicitud GET al endpoint /health
    response = client.get("/health")

    # Comprobamos que la respuesta tiene un código de estado 200
    assert response.status_code == 200

    # Verificamos que la respuesta contenga la información esperada
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert data["status"] in ["healthy", "model not loaded"]
