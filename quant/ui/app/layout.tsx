import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LlamaPajamas Quantization UI',
  description: 'Simple UI for model quantization and evaluation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
