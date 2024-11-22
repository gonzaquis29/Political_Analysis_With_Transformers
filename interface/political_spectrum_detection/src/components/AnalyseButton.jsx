// src/components/FileUpload.jsx
import React from 'react';

const AnalyseButton = ({ onClick, disabled }) => {
  return (
    <div className="button-area">
      <button 
        onClick={onClick} 
        disabled={disabled}
      >
        Analizar Contenido
      </button>
    </div>
  );
};

export default AnalyseButton;