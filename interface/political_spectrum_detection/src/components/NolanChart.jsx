import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,Legend } from 'recharts';

const NolanChart = ({ politicalFreedom, economicFreedom }) => {
    // Datos para los dos ejes
    const data = [
      { subject: 'Libertario', A: politicalFreedom, fullMark: 100 },
      { subject: 'Conservador', A:100 - economicFreedom, fullMark: 100 },
      { subject: 'Autoritario', A: 100 - politicalFreedom, fullMark: 100 },
      { subject: 'Liberal', A: economicFreedom, fullMark: 100 }
    ];
  
    return (
      <div style={{ alignContent: 'center' }}>
        <h3>Espectro de Nolan: Libertad Económica y Política</h3>
        <RadarChart
        cx={500}
        cy={250}
        outerRadius={190}
        width={1000}
        height={500}
        data={data}
      >
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" />
        <PolarRadiusAxis angle={0} domain={[0, 100]} />
        <Radar name="Resultado" dataKey="A" stroke="#84aad8" fill="#84aad8" fillOpacity={0.7} />
¿      </RadarChart>
        
      </div>
    );
  };
  
  export default NolanChart;