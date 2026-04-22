import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Fraunces } from "next/font/google";
import "./globals.css";
import { UpdateBanner } from "@/components/update-banner";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  weight: ["400", "500"],
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: "MO Trading — Trading Journal",
  description: "CANSLIM trading journal and analytics platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} ${fraunces.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        {/* Prevent flash: apply saved theme before React hydrates */}
        <script dangerouslySetInnerHTML={{ __html: `
          try {
            const t = localStorage.getItem('mo-theme');
            if (t === 'dark') document.documentElement.classList.add('dark');
          } catch(e) {}
        ` }} />
      </head>
      <body className="min-h-full" style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}>
        {children}
        <UpdateBanner />
      </body>
    </html>
  );
}
