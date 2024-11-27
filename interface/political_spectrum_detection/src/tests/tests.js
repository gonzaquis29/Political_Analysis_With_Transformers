import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Home from '../Home';
import { analyzeText } from '../../services/AnalysisService';

// Mock the AnalysisService
jest.mock('../../services/AnalysisService');

describe('Home Component', () => {
    // Mock analysis results
    const mockAnalysisResults = {
        global_metrics: {
            avg_personal_score: 0.5,
            avg_economic_score: 0.3
        }
    };

    beforeEach(() => {
        // Clear all mocks before each test
        jest.clearAllMocks();
    });

    test('renders main title correctly', () => {
        render(<Home />);
        const titleElement = screen.getByText(/Detector de espectro político para discursos/i);
        expect(titleElement).toBeInTheDocument();
    });

    test('renders text input section', () => {
        render(<Home />);
        const textInputLabel = screen.getByText(/Texto original/i);
        expect(textInputLabel).toBeInTheDocument();
    });

    test('shows loading state when analysis is in progress', async () => {
        // Mock a slow analysis
        analyzeText.mockImplementation(() => new Promise(() => { }));

        render(<Home />);
        const textInput = screen.getByRole('textbox');

        // Simulate user typing and submitting
        await userEvent.type(textInput, 'Test text');
        const analyzeButton = screen.getByRole('button', { name: /analizar/i });
        fireEvent.click(analyzeButton);

        // Check for loading message
        const loadingMessage = screen.getByText(/Cargando análisis.../i);
        expect(loadingMessage).toBeInTheDocument();
    });

    test('displays analysis results and Nolan chart', async () => {
        // Mock successful analysis
        analyzeText.mockResolvedValue(mockAnalysisResults);

        render(<Home />);
        const textInput = screen.getByRole('textbox');

        // Simulate user typing and submitting
        await userEvent.type(textInput, 'Test text for analysis');
        const analyzeButton = screen.getByRole('button', { name: /analizar/i });
        fireEvent.click(analyzeButton);

        // Wait for analysis results to appear
        await waitFor(() => {
            // Check for Nolan chart rendering
            const nolanChart = screen.getByTestId('nolan-chart');
            expect(nolanChart).toBeInTheDocument();

            // Check for Summary component
            const summaryComponent = screen.getByTestId('summary-component');
            expect(summaryComponent).toBeInTheDocument();
        });
    });

    test('handles analysis error', async () => {
        // Mock analysis error
        analyzeText.mockRejectedValue(new Error('Analysis failed'));

        render(<Home />);
        const textInput = screen.getByRole('textbox');

        // Simulate user typing and submitting
        await userEvent.type(textInput, 'Test text');
        const analyzeButton = screen.getByRole('button', { name: /analizar/i });
        fireEvent.click(analyzeButton);

        // Wait for error message
        await waitFor(() => {
            const errorMessage = screen.getByText(/Error analyzing text\. Please try again\./i);
            expect(errorMessage).toBeInTheDocument();
        });
    });
});