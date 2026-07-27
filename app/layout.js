import "./globals.css";

export const metadata = {
  title: "Hydra-Pix",
  description: "Hydra-Pix status page",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

