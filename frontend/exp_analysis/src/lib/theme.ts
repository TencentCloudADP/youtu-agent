// lib/theme.ts
import { create } from 'zustand';

interface ThemeState {
    isDarkMode: boolean;
    toggleTheme: () => void;
}

export const useThemeStore = create<ThemeState>((set) => {
    let initialDarkMode = false;
    if (typeof window !== 'undefined') {
        const savedTheme = localStorage.getItem("theme");
        if (savedTheme) {
            initialDarkMode = savedTheme === "dark";
        } else {
            initialDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        }
        document.body.className = initialDarkMode ? "dark-mode" : "light-mode";
        document.documentElement.dataset.theme = initialDarkMode ? "dark" : "light";
    }

    return {
        isDarkMode: initialDarkMode,
        toggleTheme: () => set((state) => {
            const newMode = !state.isDarkMode;
            if (typeof window !== 'undefined') {
                localStorage.setItem("theme", newMode ? "dark" : "light");
                document.body.className = newMode ? "dark-mode" : "light-mode";
                document.documentElement.dataset.theme = newMode ? "dark" : "light";
            }
            return { isDarkMode: newMode };
        }),
    };
});