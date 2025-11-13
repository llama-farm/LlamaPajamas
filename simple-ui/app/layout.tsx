import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LlamaPajamas - Simple UI',
  description: 'Quantize, evaluate, and run models',
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
