import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-powered Document Intelligence Pipeline",
  description: "Extracts and structures document content for AI-ready consumption.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">{children}</body>
    </html>
  );
}
