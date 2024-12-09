import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jokekook",
  description: "Small joke generator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
