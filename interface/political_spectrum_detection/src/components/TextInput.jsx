// src/components/TextInput.jsx
import React, { useState } from 'react';
import FileUpload from './FileUpload';
import AnalyseButton from './AnalyseButton';

const TextInput = ({ onAnalyze }) => {
  const [text, setText] = useState('');
  
  const handleChange = (e) => {
    setText(e.target.value);
  };
  
  const handleAnalyze = () => {
    if (text.trim()) {
      onAnalyze(text);
    }
  };
  
  return (
    <div className="text-input">
      <textarea 
        placeholder="Escribe o pega un discurso..."
        value={text}
        onChange={handleChange}
        rows="15"
      />
      <div className="footer">
        <span>Conteo de palabras: {text.split(' ').filter(w => w !== '').length}</span>
        <div className="button-group">
          <FileUpload />
          <AnalyseButton onClick={handleAnalyze} disabled={!text.trim()} />
        </div>
      </div>
    </div>
  );
};

export default TextInput;