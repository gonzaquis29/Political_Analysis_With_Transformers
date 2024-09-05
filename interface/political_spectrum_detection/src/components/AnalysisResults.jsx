import React from 'react';

const AnalysisResults = ({ politicalFreedom, economicFreedom }) => {
  return (
    <div className="analysis-results">
      <div className="result-item">
        <h3>Libertad Política: {politicalFreedom}%</h3>
      </div>
      <div className="result-item">
        <h3>Libertad Económica: {economicFreedom}%</h3>
      </div>
    </div>
  );
};

export default AnalysisResults;