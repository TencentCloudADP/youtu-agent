import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // Exclude KaTeX fonts from the build
      external: ['katex/dist/fonts/*'],
    },
  },
  // Use a custom CSS build configuration to exclude KaTeX fonts
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `
          @import 'katex/dist/katex.css';
        `,
      },
    },
  },
});
