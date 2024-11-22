export const analyzeText = async (text) => {
  try {
    const response = await fetch('http://127.0.0.1:8000/analyze_text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      // Intenta obtener más detalles del error del servidor
      const errorBody = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorBody}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Detailed error analyzing text:', error);
    throw error;
  }
};