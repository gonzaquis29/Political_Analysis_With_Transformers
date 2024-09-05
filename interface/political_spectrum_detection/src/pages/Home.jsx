import React from 'react';
import TextInput from '../components/TextInput';
import FileUpload from '../components/FileUpload';
import { useState } from 'react'
import NolanChart from '../components/NolanChart';
import AnalysisResults from '../components/AnalysisResults';
import Summary from '../components/Summary';

const Home = () => {
  const [politicalFreedom, setPoliticalFreedom] = useState(75); // Placeholder
  const [economicFreedom, setEconomicFreedom] = useState(60); // Placeholder

  return (
    <div className="home">
      <div className='title'>
      <h1>Detector de espectro político para discursos</h1>
      </div>
      
      <div className="text-section">
        <h3>Texto original</h3>
        <TextInput />
      </div>
        
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
    </div>
  );
};

export default Home;