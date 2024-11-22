import React, { useState } from 'react';
import TextInput from '../components/TextInput';
import FileUpload from '../components/FileUpload';
import NolanChart from '../components/NolanChart';
import AnalysisResults from '../components/AnalysisResults';
import Summary from '../components/Summary';
import { analyzeText } from '../services/AnalysisService';

const Home = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [politicalFreedom, setPoliticalFreedom] = useState(0);
  const [economicFreedom, setEconomicFreedom] = useState(0);
  
  const handleTextAnalysis = async (text) => {
    try {
      setLoading(true);
      setError(null);
      
      const results = await analyzeText(text);
      setAnalysisResults(results);
      
      // Convert scores from -1,0,1 range to 0-100 range for the Nolan chart
      const personalScore = (results.global_metrics.avg_personal_score + 1) * 50;
      const economicScore = (results.global_metrics.avg_economic_score + 1) * 50;
      
      setPoliticalFreedom(personalScore);
      setEconomicFreedom(economicScore);
    } catch (err) {
      setError('Error analyzing text. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home">
      <div className='title'>
        <h1>Detector de espectro político para discursos</h1>
      </div>
      
      <div className="text-section">
        <h3>Texto original</h3>
        <TextInput onAnalyze={handleTextAnalysis} />
      </div>
      
      {loading && <div>Cargando análisis...</div>}
      {error && <div style={{color: 'red'}}>{error}</div>}
        
      {analysisResults && (
        <div className="analysis-section">
          <NolanChart 
            politicalFreedom={politicalFreedom} 
            economicFreedom={economicFreedom} 
          />
          
          <Summary 
            politicalFreedom={politicalFreedom} 
            economicFreedom={economicFreedom} 
          />
        </div>
      )}
    </div>
  );
};

export default Home;