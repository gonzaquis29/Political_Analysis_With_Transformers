import React from 'react';

const Summary = ({ politicalFreedom, economicFreedom }) => {
  return (
    <div className="summary">
      <h3>Resumen del Análisis</h3>
      <div className="summary-bar">
        <div className="political-freedom" style={{ width: `${politicalFreedom}%` }}>
          Libertad Política: {politicalFreedom}%
        </div>
        <div className="economic-freedom" style={{ width: `${economicFreedom}%` }}>
          Libertad Económica: {economicFreedom}%
        </div>
      </div>
    </div>
  );
};

export default Summary;