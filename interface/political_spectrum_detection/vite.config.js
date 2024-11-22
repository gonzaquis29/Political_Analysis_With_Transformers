import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Opcional: si quieres abrir automáticamente en el navegador
    open: true
  }
})