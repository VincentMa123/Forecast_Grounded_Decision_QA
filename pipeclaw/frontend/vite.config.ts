import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // <--- 添加这行，设置为 true 也可以
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
      '/assets': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      }
    }
  }
})
