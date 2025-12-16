// app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Script from "next/script";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "LLM Agent Evaluations",
    description: "Evaluate your LLM Agent performance",
};

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" suppressHydrationWarning>
        <head>
            <Script id="theme-script" strategy="beforeInteractive">
                {`
                    (function() {
                        const savedTheme = localStorage.getItem("theme");
                        const systemPrefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                        const initialDarkMode = savedTheme ? savedTheme === "dark" : systemPrefersDark;
                        
                        document.body.className = initialDarkMode ? "dark-mode" : "light-mode";
                        document.documentElement.dataset.theme = initialDarkMode ? "dark" : "light";
                    })();
                `}
            </Script>
        </head>
        <body className={inter.className}>
        {children}
        </body>
        </html>
    );
}