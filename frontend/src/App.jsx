import React, { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    setSelectedFile(file)
    setError(null)
    setPrediction(null)

    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
    } else {
      setPreview(null)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    
    if (!selectedFile) {
      setError('Veuillez sélectionner une image')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post('/api/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000
      })

      setPrediction(response.data)
    } catch (err) {
      console.error('Erreur:', err)
      setError(
        err.response?.data?.error || 
        err.message || 
        'Erreur lors de la prédiction'
      )
    } finally {
      setLoading(false)
    }
  }

  const getPredictionColor = (prediction) => {
    if (!prediction) return '#333'
    
    const confidence = prediction.confidence || 0
    if (prediction.predicted_class === 'PNEUMONIA') {
      return confidence > 0.8 ? '#dc3545' : '#fd7e14'
    } else {
      return confidence > 0.8 ? '#28a745' : '#ffc107'
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🫁 Détection de Pneumonie</h1>
        <p>Téléchargez une radiographie thoracique pour analyse</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-input-container">
            <label htmlFor="file-input" className="file-input-label">
              {selectedFile ? selectedFile.name : 'Choisir une image...'}
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
            />
          </div>

          <button 
            type="submit" 
            disabled={!selectedFile || loading}
            className="submit-button"
          >
            {loading ? '🔄 Analyse en cours...' : '🔍 Analyser'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        <div className="results-container">
          {preview && (
            <div className="preview-container">
              <h3>Image sélectionnée :</h3>
              <img src={preview} alt="Aperçu" className="preview-image" />
            </div>
          )}

          {prediction && (
            <div className="prediction-container">
              <h3>Résultat de l'analyse :</h3>
              <div 
                className="prediction-result"
                style={{ borderColor: getPredictionColor(prediction) }}
              >
                <div className="prediction-class">
                  <strong>Diagnostic : </strong>
                  <span 
                    className={`prediction-label ${prediction.predicted_class?.toLowerCase()}`}
                    style={{ color: getPredictionColor(prediction) }}
                  >
                    {prediction.predicted_class === 'PNEUMONIA' ? '🦠 Pneumonie détectée' : '✅ Normal'}
                  </span>
                </div>
                
                {prediction.confidence && (
                  <div className="prediction-confidence">
                    <strong>Confiance : </strong>
                    <span style={{ color: getPredictionColor(prediction) }}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                )}

                {prediction.probabilities && (
                  <div className="prediction-probabilities">
                    <h4>Probabilités détaillées :</h4>
                    <div className="prob-bars">
                      {Object.entries(prediction.probabilities).map(([cls, prob]) => (
                        <div key={cls} className="prob-bar-container">
                          <span className="prob-label">{cls}:</span>
                          <div className="prob-bar">
                            <div 
                              className="prob-fill"
                              style={{ 
                                width: `${prob * 100}%`,
                                backgroundColor: cls === 'PNEUMONIA' ? '#dc3545' : '#28a745'
                              }}
                            />
                            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="App-footer">
        <p>© 2025 PneumoScan – Intelligent Pneumonia Detection Interface</p>
        <div className="footer-links">
          <a href="/mlflow" target="_blank" rel="noopener noreferrer">
            All rights reserved
          </a>
        </div>
      </footer>
    </div>
  )
}

export default App